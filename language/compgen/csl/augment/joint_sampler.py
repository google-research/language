# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to sample sources and targets in both induced QCFG and target CFG.

The implementation makes the assumption that the target side of every QCFG
rule can be parsed given the target CFG. While this is not gauranteed in
general, it can be enforced by providing the target CFG during induction.

The implementation then finds the set of possible target derivation for every
QCFG rule, storing a one-to-many mapping from QCFG rules to the set of target
CFG nonterminals that are allowed in the target CFG.
"""

import collections
import dataclasses
import random


from language.compgen.csl.cky import cfg_converter
from language.compgen.csl.cky import cfg_parser
from language.compgen.csl.cky import cfg_rule
from language.compgen.csl.common import json_utils
from language.compgen.csl.model.data import parsing_utils
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.targets import target_grammar

NT_PLACEHOLDER = "NT_PLACEHOLDER_%s"
JOINT_NT = "<NT>"
MAX_RECURSION_TOL = 5


@dataclasses.dataclass(frozen=True, order=True)
class JointRule:
  """Represents a QCFG rule with type annotations based on target CFG."""
  qcfg_rule: Union[qcfg_rule.QCFGRule, str]
  # Tuple contains LHS nt then any RHS nts.
  cfg_nts_set: FrozenSet[Tuple[str, Ellipsis]]

  def __str__(self):
    return "%s: %s" % (self.qcfg_rule, self.cfg_nts_set)

  def __repr__(self):
    return self.__str__()

  def get_lhs_cfg_nts(self):
    return set([cfg_nts[0] for cfg_nts in self.cfg_nts_set])

  def get_rhs_cfg_nts(self, lhs_nts, nt_idx):
    nts = set()
    for cfg_nts in self.cfg_nts_set:
      lhs_cfg_nt = cfg_nts[0]
      rhs_cfg_nts = cfg_nts[1:]
      if lhs_cfg_nt in lhs_nts:
        nts.add(rhs_cfg_nts[nt_idx])
    return nts

  def get_arity(self):
    if isinstance(self.qcfg_rule, str):
      raise ValueError("Root rule does not have arity.")
    return self.qcfg_rule.arity

  @classmethod
  def load_from_dict(cls, joint_rule):
    if joint_rule["qcfg_rule"] == parsing_utils.ROOT_RULE_KEY:
      rule = joint_rule["qcfg_rule"]
    else:
      rule = qcfg_rule.rule_from_string(joint_rule["qcfg_rule"])
    cfg_nt_sets = set()
    for cfg_nts in joint_rule["cfg_nts_set"]:
      cfg_nt_sets.add(tuple(cfg_nts))
    return cls(rule, frozenset(cfg_nt_sets))

  def save_to_dict(self):
    return {
        "qcfg_rule": str(self.qcfg_rule),
        "cfg_nts_set": list(self.cfg_nts_set)
    }


@dataclasses.dataclass
class ParseNode:
  rule: cfg_rule.CFGRule
  children: List["ParseNode"]


def _get_cfg_nts(nonterminals_to_ids, rhs_nt_rules, parse_node, num_nts):
  """Return tuple of NTs corresponding to parse."""
  ids_to_nonterminals = {v: k for k, v in nonterminals_to_ids.items()}
  lhs_nt = ids_to_nonterminals[parse_node.rule.lhs]
  node_stack = [parse_node]

  rhs_nts = [None] * num_nts
  while node_stack:
    node = node_stack.pop()
    if node.rule.idx in rhs_nt_rules:
      (nt_idx, target_nt) = rhs_nt_rules[node.rule.idx]
      # QCFG NT indexes are 1-indexed.
      rhs_nts[nt_idx - 1] = target_nt
    for child in node.children:
      node_stack.append(child)
  for nt in rhs_nts:
    if not nt:
      raise ValueError("Bad rhs_nts: %s" % rhs_nts)
  return tuple([lhs_nt] + rhs_nts)


def _rearrange_nts(cfg_nts, qcfg_idxs):
  """Rearrange cfg_nts to match qcfg_idxs."""
  lhs_nt = cfg_nts[0]
  rhs_nts = cfg_nts[1:]
  qcfg_idx_to_rhs_nt = {}
  for rhs_nt, qcfg_idx in zip(rhs_nts, qcfg_idxs):
    if qcfg_idx in qcfg_idx_to_rhs_nt:
      if qcfg_idx_to_rhs_nt[qcfg_idx] != rhs_nt:
        return None
    qcfg_idx_to_rhs_nt[qcfg_idx] = rhs_nt
  new_rhs_nts = list()
  for i in sorted(set(qcfg_idxs)):
    new_rhs_nts.append(qcfg_idx_to_rhs_nt[i])
  return tuple([lhs_nt] + new_rhs_nts)


def _convert_to_qcfg(nested_rule):
  """Convert nested JointRule to QCFG source and target."""
  sources = []
  targets = []
  rule = nested_rule[0].qcfg_rule
  idx_to_source = {}
  idx_to_target = {}
  for nt_idx, child_rule in enumerate(nested_rule[1:]):
    source, target = _convert_to_qcfg(child_rule)
    idx_to_source[nt_idx + 1] = source
    idx_to_target[nt_idx + 1] = target
  for symbol in rule.source:
    if qcfg_rule.is_nt_fast(symbol):
      index = qcfg_rule.get_nt_index(symbol)
      sources.extend(idx_to_source[index])
    else:
      sources.append(symbol)
  for symbol in rule.target:
    if qcfg_rule.is_nt_fast(symbol):
      index = qcfg_rule.get_nt_index(symbol)
      targets.extend(idx_to_target[index])
    else:
      targets.append(symbol)
  return sources, targets


class JointRuleConverter(object):
  """Class to parse QCFG rules given target CFG."""

  def __init__(self,
               target_grammar_rules,
               max_num_nts=20,
               max_single_nt_applications=1):
    # Note that `max_num_nts` is an upper bound on NTs in targets, including
    # those that share an index.
    self.max_num_nts = max_num_nts
    self.max_single_nt_applications = max_single_nt_applications
    self.converter = cfg_converter.CFGRuleConverter()
    self.parser_rules = []
    rule_idx = 0
    for rule_idx, rule in enumerate(target_grammar_rules):
      parser_rule = self.converter.convert_to_cfg_rule(
          lhs=rule.lhs,
          rhs=rule.rhs.split(" "),
          rule_idx=rule_idx,
          nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX)
      if parser_rule:
        self.parser_rules.append(parser_rule)
        rule_idx += 1

    # Add rules for every target nonterminal from placeholder NT.
    self.rhs_nt_rules = {}
    target_nts = set(self.converter.nonterminals_to_ids.keys())
    for target_nt in target_nts:
      # NT placeholders are 1-indexed.
      for nt_idx in range(1, max_num_nts + 1):
        # Create rule for TARGET_CFG_NT => NT_X.
        qcfg_nt = NT_PLACEHOLDER % nt_idx
        if qcfg_nt in target_nts:
          raise ValueError("%s overlaps with target CFG NTs: %s" %
                           (qcfg_nt, target_nts))
        rhs = ["%s%s" % (target_grammar.NON_TERMINAL_PREFIX, qcfg_nt)]
        parser_rule = self.converter.convert_to_cfg_rule(
            lhs=target_nt,
            rhs=rhs,
            rule_idx=rule_idx,
            nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX)
        self.rhs_nt_rules[rule_idx] = (nt_idx, target_nt)
        self.parser_rules.append(parser_rule)
        rule_idx += 1

  def convert(self, induced_rule, verbose=False):
    """Convert QCFGRule to JointRule."""
    tokens = induced_rule.target
    input_symbols = []
    terminal_ids = set()
    qcfg_idxs = []
    rhs = []
    num_nts = 0
    for token in tokens:
      if qcfg_rule.is_nt(token):
        qcfg_idx = qcfg_rule.get_nt_index(token)
        qcfg_idxs.append(qcfg_idx)
        # NT placeholders are 1-indexed.
        qcfg_nt = NT_PLACEHOLDER % (num_nts + 1)
        num_nts += 1
        rhs.append(JOINT_NT)
        idx = self.converter.nonterminals_to_ids[qcfg_nt]
        input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.NON_TERMINAL))
      else:
        if token not in self.converter.terminals_to_ids:
          raise ValueError(
              "token `%s` not in `converter.terminals_to_ids`: %s" %
              (token, self.converter.terminals_to_ids))
        rhs.append(token)
        idx = self.converter.terminals_to_ids[token]
        terminal_ids.add(idx)
        input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL))

    # Filter rules that contain terminals not in the input.
    def should_include(parser_rule):
      for symbol in parser_rule.rhs:
        if symbol.type == cfg_rule.TERMINAL and symbol.idx not in terminal_ids:
          return False
      return True

    filtered_rules = [
        rule for rule in self.parser_rules if should_include(rule)
    ]
    if verbose:
      print("filtered_rules:")
      for rule in filtered_rules:
        print(rule)

    def populate_fn(unused_span_begin, unused_span_end, parser_rule, children):
      return [ParseNode(parser_rule, children)]

    nonterminals = set(self.converter.nonterminals_to_ids.values())
    parses = cfg_parser.parse_symbols(
        input_symbols,
        filtered_rules,
        nonterminals,
        nonterminals,
        populate_fn,
        postprocess_fn=None,
        max_single_nt_applications=self.max_single_nt_applications,
        verbose=verbose)
    if not parses:
      print("Could not parse: %s" % (tokens,))
      return None

    # Extract cfg_nts from parses.
    cfg_nts_set = set()
    for parse_node in parses:
      cfg_nts = _get_cfg_nts(self.converter.nonterminals_to_ids,
                             self.rhs_nt_rules, parse_node, num_nts)
      cfg_nts = _rearrange_nts(cfg_nts, qcfg_idxs)
      if cfg_nts:
        cfg_nts_set.add(cfg_nts)

    return JointRule(induced_rule, frozenset(cfg_nts_set))


def _uniform_score_fn(unused_parent_rule, unused_nt_idx, unused_child_rule):
  return 1


class JointSampler(object):
  """Class to sample sources and targets."""

  def __init__(self,
               nonterminals_to_t_rules,
               nonterminals_to_nt_rules,
               max_recursion=10,
               min_recursion=1,
               verbose=False):
    """Sample source and target in both QCFG and target CFG."""
    self.nonterminals_to_t_rules = nonterminals_to_t_rules
    self.nonterminals_to_nt_rules = nonterminals_to_nt_rules
    self.max_recursion = max_recursion
    self.min_recursion = min_recursion
    self.verbose = verbose

  @classmethod
  def from_rules(cls,
                 target_grammar_rules,
                 qcfg_rules,
                 max_num_nts=20,
                 max_recursion=10,
                 min_recursion=1,
                 max_single_nt_applications=1,
                 verbose=False):
    """Init JointSampler from QCFG rules and Target CFG rules.

    Args:
      target_grammar_rules: A list of TargetCfgRule instance.
      qcfg_rules: A list of QCFGRule instance.
      max_num_nts: An upper bound on NTs in targets, including those that share
        an index.
      max_recursion: Attempt to limit the derivation tree depth to this number.
        There are cases where this number may not be a strict bound, e.g. if
        certain nonterminals cannot be expanded by a rule with no RHS
        nonterminals.
      min_recursion: Minimum recursion depth.
      max_single_nt_applications: The maximum number of single NT applications
        for parsing induced QCFG rules using target CFG rules.
      verbose: Whether to log rule expansion probabilities (very slow).

    Returns:
      A JointSampler Instance.
    """
    converter = JointRuleConverter(target_grammar_rules, max_num_nts,
                                   max_single_nt_applications)
    joint_rules = []
    for i, rule in enumerate(qcfg_rules):
      joint_rule = converter.convert(rule)
      if joint_rule:
        joint_rules.append(joint_rule)
      if verbose:
        print("Converting rule %d." % i)

    return cls.from_joint_rules(
        joint_rules,
        max_recursion=max_recursion,
        min_recursion=min_recursion,
        verbose=verbose)

  @classmethod
  def from_joint_rules(cls,
                       joint_rules,
                       max_recursion=10,
                       min_recursion=1,
                       verbose=False):
    """Init JointSampler from QCFG rules and Target CFG rules.

    Args:
      joint_rules: A list of JointRule instances.
      max_recursion: Attempt to limit the derivation tree depth to this number.
        There are cases where this number may not be a strict bound, e.g. if
        certain nonterminals cannot be expanded by a rule with no RHS
        nonterminals.
      min_recursion: Minimum recursion depth.
      verbose: Whether to log rule expansion probabilities (very slow).

    Returns:
      A JointSampler Instance.
    """
    # Dict of NT symbol to joint_rule.
    nonterminals_to_nt_rules = collections.defaultdict(list)
    nonterminals_to_t_rules = collections.defaultdict(list)
    for joint_rule in joint_rules:
      for cfg_nts in joint_rule.cfg_nts_set:
        lhs_cfg_nt = cfg_nts[0]
        if joint_rule.get_arity() > 0:
          nonterminals_to_nt_rules[lhs_cfg_nt].append(joint_rule)
        else:
          nonterminals_to_t_rules[lhs_cfg_nt].append(joint_rule)
    return cls(
        nonterminals_to_t_rules=nonterminals_to_t_rules,
        nonterminals_to_nt_rules=nonterminals_to_nt_rules,
        max_recursion=max_recursion,
        min_recursion=min_recursion,
        verbose=verbose)

  @classmethod
  def from_file(cls,
                filename,
                max_recursion=10,
                min_recursion=1,
                verbose=False):
    """Init JointSampler from JSON file."""
    save_dict = json_utils.json_file_to_dict(filename)
    nonterminals_to_nt_rules = collections.defaultdict(list)
    nonterminals_to_t_rules = collections.defaultdict(list)
    for nt, rules in save_dict["nonterminals_to_nt_rules"].items():
      for rule in rules:
        nonterminals_to_nt_rules[nt].append(JointRule.load_from_dict(rule))
    for nt, rules in save_dict["nonterminals_to_t_rules"].items():
      for rule in rules:
        nonterminals_to_t_rules[nt].append(JointRule.load_from_dict(rule))
    print("Loaded JointSampler from %s." % filename)
    return cls(
        nonterminals_to_t_rules=nonterminals_to_t_rules,
        nonterminals_to_nt_rules=nonterminals_to_nt_rules,
        max_recursion=max_recursion,
        min_recursion=min_recursion,
        verbose=verbose)

  def save(self, filename):
    """Save JointSampler to JSON file."""
    nonterminals_to_nt_rules = collections.defaultdict(list)
    nonterminals_to_t_rules = collections.defaultdict(list)

    for nt, rules in self.nonterminals_to_nt_rules.items():
      for rule in rules:
        nonterminals_to_nt_rules[nt].append(rule.save_to_dict())
    for nt, rules in self.nonterminals_to_t_rules.items():
      for rule in rules:
        nonterminals_to_t_rules[nt].append(rule.save_to_dict())
    save_dict = {
        "nonterminals_to_nt_rules": nonterminals_to_nt_rules,
        "nonterminals_to_t_rules": nonterminals_to_t_rules
    }
    json_utils.dict_to_json_file(save_dict, filename)

  def __str__(self):
    rep_str = ["nonterminals_to_nt_rules"]
    for nt, rules in self.nonterminals_to_nt_rules.items():
      rep_str.append("%s: %s" % (nt, rules))
    rep_str.append("nonterminals_to_t_rules")
    for nt, rules in self.nonterminals_to_t_rules.items():
      rep_str.append("%s: %s" % (nt, rules))
    return "\n".join(rep_str)

  def __repr__(self):
    return self.__str__()

  def _sample_rule(self, nt_symbols, score_fn, parent_rule, nt_idx, recursions):
    """Sample a rule."""

    if recursions > self.max_recursion + MAX_RECURSION_TOL:
      raise ValueError("Exceed the maximum recursion.")

    rules_t = set()
    rules_nt = set()
    for nt in nt_symbols:
      rules_t |= set(self.nonterminals_to_t_rules[nt])
      rules_nt |= set(self.nonterminals_to_nt_rules[nt])

    rules = list(rules_nt) + list(rules_t)
    weights = [
        score_fn(parent_rule.qcfg_rule, nt_idx, rule.qcfg_rule)
        for rule in rules
    ]

    if recursions >= self.max_recursion and rules_nt:
      weights = [0] * len(rules_nt) + weights[len(rules_nt):]
    if recursions < self.min_recursion and rules_t:
      weights = weights[:len(rules_nt)] + [0] * len(rules_t)

    if self.verbose:
      # Print out top 10 rule probabilities and current sampling context.
      print("========\ncontext: %s (%s)" % (parent_rule.qcfg_rule, nt_idx))
      rules_and_weights = list(zip(rules, weights))
      rules_and_weights.sort(key=lambda x: x[1], reverse=True)
      for rule, weight in rules_and_weights[:10]:
        print("%s - %.2f" % (rule.qcfg_rule, weight))

    return random.choices(rules, weights=weights)[0]

  def _expand(self, nt_symbols, score_fn, parent_rule, nt_idx, recursions):
    """Recursively expand `nt_symbol`."""
    rule = self._sample_rule(nt_symbols, score_fn, parent_rule, nt_idx,
                             recursions)
    outputs = [rule]
    lhs_nts = rule.get_lhs_cfg_nts().intersection(nt_symbols)
    for nt_idx in range(rule.get_arity()):
      nts = rule.get_rhs_cfg_nts(lhs_nts, nt_idx)
      outputs.append(self._expand(nts, score_fn, rule, nt_idx, recursions + 1))
    return outputs

  def sample(self, score_fn=None):
    """Sample a source and target pair.

    Args:
      score_fn: A scoring function that takes parent_rule (QCFGRule), nt_idx,
        child_rule (QCFGRule) and returns s score.

    Returns:
      A tuple of source tokens and target tokens.
    """
    if not score_fn:
      score_fn = _uniform_score_fn
    root_rule = JointRule(parsing_utils.ROOT_RULE_KEY, frozenset())
    outputs = None
    while not outputs:
      try:
        outputs = self._expand({target_grammar.ROOT_SYMBOL},
                               score_fn,
                               root_rule,
                               0,
                               recursions=0)
      except ValueError as e:
        print(e)
    sources, targets = _convert_to_qcfg(outputs)
    return sources, targets

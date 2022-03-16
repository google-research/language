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
"""Utilities for QCFG target parsing by extending a general CFG parser."""

from language.compgen.csl.cky import cfg_converter
from language.compgen.csl.cky import cfg_parser
from language.compgen.csl.cky import cfg_rule
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.targets import target_grammar


def _convert_qcfg_nt(nt):
  """Convert QCFG non-terminal to TargetCFG non-terminal."""
  return "%s%s" % (target_grammar.NON_TERMINAL_PREFIX, nt)


def parse(tokens, rules, node_fn, postprocess_fn, verbose=False):
  """Run bottom up parser on QCFG target using target CFG.

  Args:
    tokens: List of strings for input.
    rules: List of TargetCfgRule instances.
    node_fn: Function with input arguments (span_begin, span_end, rule,
      children) and returns a list of "node".
    postprocess_fn: Function from a list of "nodes" to "nodes".
    verbose: Print debug output if True.

  Returns:
    A List of "node" objects for completed parses.
  """
  if verbose:
    print("tokens: %s" % (tokens,))
    print("rules:")
    for rule in rules:
      print(str(rule))
  terminals = [
      token for token in tokens
      if not token.startswith(qcfg_rule.NON_TERMINAL_PREFIX)
  ]

  # Convert rules.
  converter = cfg_converter.CFGRuleConverter()
  parser_rules = []
  idx_to_rule = {}
  rule_idx = 0
  for rule in rules:
    parser_rule = converter.convert_to_cfg_rule(
        lhs=rule.lhs,
        rhs=rule.rhs.split(" "),
        rule_idx=rule_idx,
        nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX,
        allowed_terminals=set(terminals))
    if parser_rule:
      parser_rules.append(parser_rule)
      idx_to_rule[rule_idx] = rule
      rule_idx += 1

  # Add rules for every target nonterminal and QCFG nonterminal
  target_nts = set(converter.nonterminals_to_ids.keys())
  qcfg_nts = set(qcfg_rule.get_nts(tokens))
  for target_nt in target_nts:
    for qcfg_nt in qcfg_nts:
      rule = target_grammar.TargetCfgRule(target_nt, _convert_qcfg_nt(qcfg_nt))
      parser_rule = converter.convert_to_cfg_rule(
          lhs=rule.lhs,
          rhs=rule.rhs.split(" "),
          rule_idx=rule_idx,
          nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX)
      parser_rules.append(parser_rule)
      idx_to_rule[rule_idx] = rule
      rule_idx += 1

  input_symbols = []
  for token in tokens:
    if qcfg_rule.is_nt(token):
      if token not in converter.nonterminals_to_ids:
        return []
      idx = converter.nonterminals_to_ids[token]
      input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.NON_TERMINAL))
    else:
      if token not in converter.terminals_to_ids:
        return []
      idx = converter.terminals_to_ids[token]
      input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL))

  # Wrap node_fn to pass original Rule instead of CFGRule.
  def populate_fn(span_begin, span_end, parser_rule, children):
    rule = idx_to_rule[parser_rule.idx]
    nodes = node_fn(span_begin, span_end, rule, children)
    return nodes

  nonterminals = set(converter.nonterminals_to_ids.values())
  if verbose:
    print("parser_rules: %s" % parser_rules)

  parses = cfg_parser.parse_symbols(
      input_symbols,
      parser_rules,
      nonterminals,
      nonterminals,
      populate_fn,
      postprocess_fn,
      max_single_nt_applications=0,
      verbose=verbose)
  return parses


def can_parse(target, rules, verbose=False):
  """Return True if target can be derived given rules using parser.

  Args:
    target: Target string (can contain non-terminals or terminals).
    rules: List of TargetCFGRule instances.
    verbose: Print debug output if True.

  Returns:
    True if target can be derived.
  """

  def node_fn(unused_span_begin, unused_span_end, rule, children):
    """Represent node as applied rules."""
    rules = [rule]
    for child in children:
      rules.extend(child)
    return [rules]

  def postprocess_fn(nodes):
    """Filter invalid nodes."""
    new_nodes = []
    for node in nodes:
      qcfg_to_target = {}
      valid_node = True
      for rule in node:
        if qcfg_rule.is_nt(rule.rhs[len(target_grammar.NON_TERMINAL_PREFIX):]):
          qcfg_to_target.setdefault(rule.rhs, rule.lhs)
          # Filter out node with QCFG NT matches to more than one target NTs.
          if qcfg_to_target[rule.rhs] != rule.lhs:
            valid_node = False
            break
      if valid_node:
        new_nodes.append(node)
    return new_nodes

  tokens = target.split(" ")
  outputs = parse(
      tokens,
      rules,
      verbose=verbose,
      node_fn=node_fn,
      postprocess_fn=postprocess_fn)
  if outputs:
    return True
  return False


PLACEHOLDER_NT = "ANYNT"


class TargetChecker(object):
  """Faster version of `can_parse` above but less strict for repeated NTs."""

  def __init__(self, target_grammar_rules):
    # Convert rules.
    self.converter = cfg_converter.CFGRuleConverter()
    self.parser_rules = []
    rule_idx = 0
    for rule in target_grammar_rules:
      parser_rule = self.converter.convert_to_cfg_rule(
          lhs=rule.lhs,
          rhs=rule.rhs.split(" "),
          rule_idx=rule_idx,
          nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX)
      if parser_rule:
        self.parser_rules.append(parser_rule)
        rule_idx += 1

    # Add rules for every target nonterminal from placeholder NT.
    target_nts = set(self.converter.nonterminals_to_ids.keys())
    placeholder_rhs = ["%s%s" % (target_grammar.NON_TERMINAL_PREFIX,
                                 PLACEHOLDER_NT)]
    for target_nt in target_nts:
      parser_rule = self.converter.convert_to_cfg_rule(
          lhs=target_nt,
          rhs=placeholder_rhs,
          rule_idx=rule_idx,
          nonterminal_prefix=target_grammar.NON_TERMINAL_PREFIX)
      self.parser_rules.append(parser_rule)
      rule_idx += 1

  def can_parse(self, tokens, verbose=False):
    """Return True if can be parsed given target CFG."""
    input_symbols = []
    terminal_ids = set()
    for token in tokens:
      if qcfg_rule.is_nt(token):
        idx = self.converter.nonterminals_to_ids[PLACEHOLDER_NT]
        input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.NON_TERMINAL))
      else:
        if token not in self.converter.terminals_to_ids:
          if verbose:
            print("token `%s` not in `converter.terminals_to_ids`: %s" % (
                token, self.converter.terminals_to_ids))
          return False
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
        rule for rule in self.parser_rules if should_include(rule)]
    if verbose:
      print("filtered_rules:")
      for rule in filtered_rules:
        print(rule)

    def populate_fn(unused_span_begin, unused_span_end,
                    unused_parser_rule, unused_children):
      return [True]

    nonterminals = set(self.converter.nonterminals_to_ids.values())
    parses = cfg_parser.parse_symbols(
        input_symbols,
        filtered_rules,
        nonterminals,
        nonterminals,
        populate_fn,
        postprocess_fn=None,
        max_single_nt_applications=2,
        verbose=verbose)
    if parses:
      return True
    else:
      return False

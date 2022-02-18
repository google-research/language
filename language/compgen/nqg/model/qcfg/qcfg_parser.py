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
"""Utilities for QCFG parsing by extending a general CFG parser."""

from language.compgen.nqg.common.cky import cfg_parser
from language.compgen.nqg.common.cky import cfg_rule

from language.compgen.nqg.model.qcfg import qcfg_rule


def _convert_rhs(rule, nt_idx, tokens_to_input_ids):
  """Convert rule to `rhs` argument for CFGRule."""
  rhs = []
  for token in rule.source:
    if token == qcfg_rule.NT_1:
      rhs.append(cfg_rule.CFGSymbol(idx=nt_idx, type=cfg_rule.NON_TERMINAL))
    elif token == qcfg_rule.NT_2:
      rhs.append(cfg_rule.CFGSymbol(idx=nt_idx, type=cfg_rule.NON_TERMINAL))
    else:
      if token not in tokens_to_input_ids:
        # Rule contains tokens not in the input so can be ignored for parsing.
        return None
      else:
        token_id = tokens_to_input_ids[token]
        rhs.append(cfg_rule.CFGSymbol(idx=token_id, type=cfg_rule.TERMINAL))
  return tuple(rhs)


def parse(tokens, rules, node_fn, postprocess_cell_fn, verbose=False):
  """Run bottom up parser.

  Args:
    tokens: List of strings for input.
    rules: List of QCFGRule instances.
    node_fn: Function with input arguments (span_begin, span_end, rule,
      children) and returns a "node".
    postprocess_cell_fn: Function from a list of "nodes" to "nodes".
    verbose: Print debug output if True.

  Returns:
    A List of "node" objects for completed parses.
  """
  if verbose:
    print("tokens: %s" % (tokens,))
    print("rules:")
    for rule in rules:
      print(str(rule))

  # Convert tokens to integer IDs.
  tokens_to_input_ids = {}
  input_ids_to_tokens = {}
  for idx, token in enumerate(set(tokens)):
    input_ids_to_tokens[idx] = token
    tokens_to_input_ids[token] = idx
  input_ids = [tokens_to_input_ids[token] for token in tokens]

  # Our QCFG grammars always use a single NT symbol.
  nt_idx = 0

  # Convert to ParserRule format.
  idx_to_rule = {}
  parser_rules = []
  rule_idx = 0
  for rule in rules:
    rhs = _convert_rhs(rule, nt_idx, tokens_to_input_ids)
    if rhs is None:
      continue
    parser_rule = cfg_rule.CFGRule(idx=rule_idx, lhs=nt_idx, rhs=rhs)
    parser_rules.append(parser_rule)
    idx_to_rule[rule_idx] = rule
    rule_idx += 1

  # Wrap node_fn to pass original Rule instead of CFGRule.
  def populate_fn(span_begin, span_end, parser_rule, children):
    rule = idx_to_rule[parser_rule.idx]
    return node_fn(span_begin, span_end, rule, children)

  nonterminals = {nt_idx}
  start_idx = nt_idx

  if verbose:
    print("parser_rules: %s" % parser_rules)

  parses = cfg_parser.parse(
      input_ids,
      parser_rules,
      nonterminals,
      start_idx,
      populate_fn,
      postprocess_cell_fn,
      verbose=verbose)

  return parses


def can_parse(source, target, rules, verbose=False):
  """Return True if source and target can be derived given rules using parser.

  Args:
    source: Source string (cannot contain non-terminals).
    target: Target string (cannot contain non-terminals).
    rules: List of QCFGRule instances.
    verbose: Print debug output if True.

  Returns:
    True if source and target can be derived.
  """

  def node_fn(unused_span_begin, unused_span_end, rule, children):
    """Represent nodes as target strings."""
    return qcfg_rule.apply_target(rule, children)

  def postprocess_cell_fn(nodes):
    """Filter and merge generated nodes."""
    new_nodes = []
    for node in nodes:
      # Discard targets that are not substrings of the gold target.
      if node in target:
        new_nodes.append(node)
    return list(set(new_nodes))

  tokens = source.split(" ")
  outputs = parse(
      tokens,
      rules,
      verbose=verbose,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_cell_fn)

  if outputs and target in outputs:
    return True
  else:
    return False

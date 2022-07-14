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

import typing

from language.compgen.csl.cky import cfg_converter
from language.compgen.csl.cky import cfg_parser
from language.compgen.csl.cky import cfg_rule
from language.compgen.csl.qcfg import qcfg_rule

NT_IDX = "0"
NON_TERMINAL_PREFIX = "NT_"
NT = NON_TERMINAL_PREFIX + NT_IDX


def _convert_nt(source):
  """Convert rule to `rhs` argument for CFGRule."""
  return [NT if qcfg_rule.is_nt_fast(s) else s for s in source]


def parse(tokens,
          rules,
          node_fn,
          postprocess_cell_fn,
          max_single_nt_applications=1,
          verbose=False):
  """Run bottom up parser.

  Args:
    tokens: List of strings for input (terminals or nonterminals).
    rules: List of QCFGRule instances.
    node_fn: Function with input arguments (span_begin, span_end, rule,
      children) and returns a "node".
    postprocess_cell_fn: Function from a list of "nodes" to "nodes".
    max_single_nt_applications: The maximum number of times a rule where the RHS
      is a single nonterminal symbol can be applied consecutively.
    verbose: Print debug output if True.

  Returns:
    A List of "node" objects for completed parses.
  """
  if verbose:
    print("tokens: %s" % (tokens,))
    print("rules:")
    for rule in rules:
      print(str(rule))

  # Our QCFG grammars always use a single NT symbol.
  nt_idx = 0

  # Convert to ParserRule format.
  converter = cfg_converter.CFGRuleConverter()
  idx_to_rule = {}
  parser_rules = []
  rule_idx = 0

  allowed_terminals = set(tokens)
  for rule in rules:
    if not qcfg_rule.is_allowed(rule.source, allowed_terminals):
      continue
    rhs = _convert_nt(rule.source)
    parser_rule = converter.convert_to_cfg_rule(
        lhs=NT_IDX,
        rhs=rhs,
        rule_idx=rule_idx,
        nonterminal_prefix=NON_TERMINAL_PREFIX)
    parser_rules.append(parser_rule)
    idx_to_rule[rule_idx] = rule
    rule_idx += 1

  for token in tokens:
    if not qcfg_rule.is_nt(token) and token not in converter.terminals_to_ids:
      if verbose:
        print("Input token does not appear in rules: %s" % token)
      return []

  input_symbols = []
  for token in tokens:
    if qcfg_rule.is_nt(token):
      input_symbols.append(cfg_rule.CFGSymbol(nt_idx, cfg_rule.NON_TERMINAL))
    else:
      idx = converter.terminals_to_ids[token]
      input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL))

  # Wrap node_fn to pass original Rule instead of CFGRule.
  def populate_fn(span_begin, span_end, parser_rule, children):
    rule = idx_to_rule[parser_rule.idx]
    node = node_fn(span_begin, span_end, rule, children)
    return [node]

  nonterminals = {nt_idx}
  start_idx = nt_idx

  if verbose:
    print("parser_rules: %s" % parser_rules)

  parses = cfg_parser.parse_symbols(
      input_symbols,
      parser_rules,
      nonterminals, {start_idx},
      populate_fn,
      postprocess_cell_fn,
      max_single_nt_applications=max_single_nt_applications,
      verbose=verbose)

  return parses


def can_parse(source,
              target,
              rules,
              max_single_nt_applications=1,
              verbose=False):
  """Return True if source and target can be derived given rules using parser.

  Args:
    source: Source string (cannot contain non-terminals).
    target: Target string (cannot contain non-terminals).
    rules: List of QCFGRule instances.
    max_single_nt_applications: The maximum number of times a rule where the RHS
      is a single nonterminal symbol can be applied consecutively.
    verbose: Print debug output if True.

  Returns:
    True if source and target can be derived.
  """

  def node_fn(unused_span_begin, unused_span_end, rule, children):
    """Represent nodes as target strings."""
    return qcfg_rule.apply_target(rule, children)

  def postprocess_cell_fn(nodes):
    """Filter and merge generated nodes."""
    new_nodes = list(set(nodes).intersection(target))
    return new_nodes

  tokens = source.split(" ")
  outputs = parse(
      tokens,
      rules,
      max_single_nt_applications=max_single_nt_applications,
      verbose=verbose,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_cell_fn)

  if outputs and target in outputs:
    return True
  else:
    return False

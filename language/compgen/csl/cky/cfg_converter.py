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
"""Tools for converting grammars to CFGRule format."""

import collections
import typing

from language.compgen.csl.cky import cfg_rule


class CFGRuleConverter(object):
  """Converts string-based rules to CFGRule format."""

  def __init__(self):
    self.terminals_to_ids = {}
    self.nonterminals_to_ids = {}

  def _get_terminal_id(self, symbol):
    if symbol not in self.terminals_to_ids:
      self.terminals_to_ids[symbol] = len(self.terminals_to_ids)
    return self.terminals_to_ids[symbol]

  def _get_nonterminal_id(self, symbol):
    if symbol not in self.nonterminals_to_ids:
      self.nonterminals_to_ids[symbol] = len(self.nonterminals_to_ids)
    return self.nonterminals_to_ids[symbol]

  def convert_to_cfg_rule(self,
                          lhs,
                          rhs,
                          rule_idx,
                          nonterminal_prefix,
                          allowed_terminals=None):
    """Convert symbol strings to CFGRule.

    Args:
      lhs: String symbol for LHS.
      rhs: List of string symbols for RHS.
      rule_idx: Integer index for rule.
      nonterminal_prefix: String prefix for nonterminal symbols in `rhs`.
      allowed_terminals: If set, returns None if rhs contains terminals not in
        this set.

    Returns:
      A CFGRule.
    """
    rhs_symbols = []
    lhs_idx = self._get_nonterminal_id(lhs)
    for token in rhs:
      if token.startswith(nonterminal_prefix):
        symbol_idx = self._get_nonterminal_id(token[len(nonterminal_prefix):])
        rhs_symbols.append(
            cfg_rule.CFGSymbol(idx=symbol_idx, type=cfg_rule.NON_TERMINAL))
      else:
        if allowed_terminals and token not in allowed_terminals:
          return None
        symbol_idx = self._get_terminal_id(token)
        rhs_symbols.append(
            cfg_rule.CFGSymbol(idx=symbol_idx, type=cfg_rule.TERMINAL))
    rule = cfg_rule.CFGRule(idx=rule_idx, lhs=lhs_idx, rhs=rhs_symbols)
    return rule


def expand_unit_rules(rules, lhs_fn, rhs_fn, rule_init_fn, nonterminal_prefix):
  """Removes unit rules, i.e.

  X -> Y where X and Y are non-terminals.

  Args:
    rules: List of rule objects.
    lhs_fn: Returns `lhs` of a rule.
    rhs_fn: Returns `rhs` of a rule.
    rule_init_fn: Function that takes (lhs, rule) and returns a copy of `rule`
      but with the lhs symbol set to `lhs`.
    nonterminal_prefix: String prefix for nonterminal symbols in `rhs`.

  Returns:
    Set of tuples of (lhs, rhs, original rule_idx of rhs) to add.
  """
  # List of 2-tuple of non-terminal strings.
  unit_tuples = set()
  # List of rule tuples.
  other_rules = []

  for rule in rules:
    tokens = rhs_fn(rule).split(" ")
    if len(tokens) == 1 and tokens[0].startswith(nonterminal_prefix):
      unit_rhs = tokens[0][len(nonterminal_prefix):]
      unit_tuples.add((lhs_fn(rule), unit_rhs))
    else:
      other_rules.append(rule)

  # Identify any chains of unit rules.
  # For example, if we have X -> Y and Y -> Z, then we need to consider X -> Z.
  unit_lhs_to_rhs_set = collections.defaultdict(set)
  for unit_lhs, unit_rhs in unit_tuples:
    unit_lhs_to_rhs_set[unit_lhs].add(unit_rhs)

  derived_unit_tuples = set()
  for unit_lhs_start in unit_lhs_to_rhs_set.keys():
    stack = [unit_lhs_start]
    visited_lhs = []

    while stack:
      unit_lhs = stack.pop()
      # Check for cycles so that we don't loop indefinitely.
      if unit_lhs in visited_lhs:
        raise ValueError("There exists an unallowed cycle in unit rules: %s" %
                         visited_lhs)
      visited_lhs.append(unit_lhs)
      for unit_rhs in unit_lhs_to_rhs_set.get(unit_lhs, {}):
        stack.append(unit_rhs)
      # Add derived unit rule following transitive chain.
      derived_unit_tuples.add((unit_lhs_start, unit_rhs))
  unit_tuples |= derived_unit_tuples

  # Add derived rules based on unit rules.
  # For example, if we have X -> Y and Y -> foo, then we add X -> foo.
  new_rules = []
  for unit_lhs, unit_rhs in unit_tuples:
    for rule in other_rules:
      if lhs_fn(rule) == unit_rhs:
        new_rules.append(rule_init_fn(unit_lhs, rule))
  return other_rules + new_rules

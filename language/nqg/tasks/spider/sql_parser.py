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
"""Utilities for parsing SQL to tree using CFG.

The CFG defined in this file covers covers only the subset of SQL in the
Spider-SSP dataset, but gaurantees a single unique parse for every
query in that dataset.
"""

import collections

from language.nqg.common.cky import cfg_parser
from language.nqg.common.cky import cfg_rule

from language.nqg.tasks.spider import sql_tokenizer

# Used for string formatting of CFG rules.
NON_TERMINAL_PREFIX = "##"
ARROW = "=>"


class Rule(object):
  """Represents a string-formatted CFG rule."""

  def __init__(self, lhs, rhs):
    self.lhs = lhs  # String.
    self.rhs = rhs  # String.

  def __str__(self):
    return "%s %s %s" % (self.lhs, ARROW, self.rhs)

  def __repr__(self):
    return str(self)

  @classmethod
  def from_string(cls, rule_string):
    symbols = rule_string.split(" ")
    if symbols[1] != ARROW:
      raise ValueError("Invalid rule_string: %s." % rule_string)
    lhs = symbols[0]
    rhs = " ".join(symbols[2:])
    return cls(lhs, rhs)


class Node(object):
  """Represents a chart cell entry."""

  def __init__(self, lhs, rhs, children):
    self.lhs = lhs
    self.rhs = rhs
    self.children = children

  def __str__(self):
    return "%s => %s" % (self.lhs, self.rhs)

  def __repr__(self):
    return str(self)


def _parse_to_str_inner(node, indent):
  node_strs = []
  node_strs.append("-" * indent + str(node))
  for child in node.children:
    node_strs.extend(_parse_to_str_inner(child, indent + 1))
  return node_strs


def parse_to_str(node):
  return "\n".join(_parse_to_str_inner(node, indent=0))


def _get_populate_fn(idx_to_rule):

  def populate_fn(unused_span_begin, unused_span_end, parser_rule, children):
    rule = idx_to_rule[parser_rule.idx]
    return Node(rule.lhs, rule.rhs, children)

  return populate_fn


def _convert_to_rhs(rule, terminals_to_ids, nonterminals_to_ids):
  """Convert rule to `rhs` argument for CFGRule."""
  rhs = []
  for token in rule.rhs.split(" "):
    if token.startswith(NON_TERMINAL_PREFIX):
      symbol_idx = nonterminals_to_ids[token[len(NON_TERMINAL_PREFIX):]]
      rhs.append(cfg_rule.CFGSymbol(idx=symbol_idx, type=cfg_rule.NON_TERMINAL))
    else:
      if token not in terminals_to_ids:
        return None
      symbol_idx = terminals_to_ids[token]
      rhs.append(cfg_rule.CFGSymbol(idx=symbol_idx, type=cfg_rule.TERMINAL))
  rhs = tuple(rhs)
  return rhs


def _postprocess_fn(nodes):
  """Grammar should be unambiguous, so nodes should never be > 1."""
  if len(nodes) > 1:
    raise ValueError(
        "Grammar is ambiguous for sub-tree with %s derivations:\n\n%s" %
        (len(nodes), "\n\n".join([parse_to_str(node) for node in nodes])))
  return nodes


def expand_unit_rules(rules):
  """Removes unit rules, i.e. X -> Y where X and Y are non-terminals."""
  # List of 2-tuple of non-terminal strings.
  unit_rules = set()
  # List of Rules.
  other_rules = []

  for rule in rules:
    tokens = rule.rhs.split(" ")
    if len(tokens) == 1 and tokens[0].startswith(NON_TERMINAL_PREFIX):
      unit_rhs = tokens[0][len(NON_TERMINAL_PREFIX):]
      unit_rules.add((rule.lhs, unit_rhs))
    else:
      other_rules.append(rule)

  # Identify any chains of unit rules.
  # For example, if we have X -> Y and Y -> Z, then we need to consider X -> Z.
  unit_lhs_to_rhs_set = collections.defaultdict(set)
  for unit_lhs, unit_rhs in unit_rules:
    unit_lhs_to_rhs_set[unit_lhs].add(unit_rhs)

  derived_unit_rules = set()
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
      derived_unit_rules.add((unit_lhs_start, unit_rhs))
  unit_rules |= derived_unit_rules

  # Add derived rules based on unit rules.
  # For example, if we have X -> Y and Y -> foo, then we add X -> foo.
  new_rules = []
  for unit_lhs, unit_rhs in unit_rules:
    for rule in other_rules:
      if rule.lhs == unit_rhs:
        new_rules.append(Rule(unit_lhs, rule.rhs))

  return other_rules + new_rules


def _run_parser(tokens, rules, verbose):
  """Run bottom up parser."""
  # Expand and eliminate unit rules.
  rules = expand_unit_rules(rules)

  # Convert tokens to integer IDs.
  terminals_to_ids = {}
  for idx, token in enumerate(set(tokens)):
    terminals_to_ids[token] = idx
  input_ids = [terminals_to_ids[token] for token in tokens]

  # Generate non-terminal IDs.
  nonterminals_to_ids = {}
  nt_idx = 0
  for rule in rules:
    if rule.lhs not in nonterminals_to_ids:
      nonterminals_to_ids[rule.lhs] = nt_idx
      nt_idx += 1
  nonterminals = nonterminals_to_ids.values()
  start_idx = nonterminals_to_ids["ROOT"]

  # Convert to ParserRule format.
  idx_to_rule = {}
  parser_rules = []
  rule_idx = 0
  for rule in rules:
    rhs = _convert_to_rhs(rule, terminals_to_ids, nonterminals_to_ids)
    if rhs is None:
      continue
    lhs = nonterminals_to_ids[rule.lhs]
    parser_rule = cfg_rule.CFGRule(idx=rule_idx, lhs=lhs, rhs=rhs)
    parser_rules.append(parser_rule)
    idx_to_rule[rule_idx] = rule
    rule_idx += 1

  populate_fn = _get_populate_fn(idx_to_rule)
  parses = cfg_parser.parse(
      input_ids,
      parser_rules,
      nonterminals,
      start_idx,
      populate_fn=populate_fn,
      postprocess_fn=_postprocess_fn,
      verbose=verbose)

  return parses


KEYWORDS = frozenset([
    "select",
    "from",
    "where",
    "group by",
    "order by",
    "limit",
    "intersect",
    "union",
    "except",
    "join",
    "on",
    "as",
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
    "none",
    "-",
    "+",
    "*",
    "/",
    "none",
    "max",
    "min",
    "count",
    "sum",
    "avg",
    "and",
    "or",
    "intersect",
    "union",
    "except",
    "desc",
    "asc",
    "distinct",
    ",",
    ".",
    "not",
    "(",
    ")",
])

# CFG rules containing <= 2 RHS non-terminals.
RULES = [
    "ROOT => ##ROOT union ##ROOT",
    "ROOT => ##ROOT intersect ##ROOT",
    "ROOT => ##ROOT except ##ROOT",
    "ROOT => ##SELECT",
    "SELECT => select ##SELECT_EXPR from ##FROM_EXPR",
    "SELECT => select distinct ##SELECT_EXPR from ##FROM_EXPR",
    "SELECT => ##SELECT where ##EXPR",
    "SELECT => ##SELECT order by ##ORDER_EXPR",
    "SELECT => ##SELECT having ##EXPR",
    "SELECT => ##SELECT group by ##GROUP_EXPR",
    "SELECT => ##SELECT limit ##T",
    "SELECT_EXPR => ##SELECT_ATOM",
    "SELECT_EXPR => ##SELECT_CONJ",
    "SELECT_ATOM => ##T",
    "SELECT_ATOM => ##T - ##T",
    "SELECT_ATOM => ##T + ##T",
    "SELECT_ATOM => ##T / ##T",
    "SELECT_ATOM => ##T * ##T",
    "SELECT_ATOM => ( ##ROOT )",
    "SELECT_ATOM => ( ##SELECT_ATOM )",
    "SELECT_ATOM => count ( ##SELECT_ATOM )",
    "SELECT_ATOM => count ( distinct ##SELECT_ATOM )",
    "SELECT_ATOM => sum ( ##SELECT_ATOM )",
    "SELECT_ATOM => avg ( ##SELECT_ATOM )",
    "SELECT_ATOM => max ( ##SELECT_ATOM )",
    "SELECT_ATOM => min ( ##SELECT_ATOM )",
    "SELECT_CONJ => ##SELECT_ATOM , ##SELECT_ATOM",
    "SELECT_CONJ => ##SELECT_CONJ , ##SELECT_ATOM",
    "FROM_EXPR => ( ##ROOT )",
    "FROM_EXPR => ##T",
    "FROM_EXPR => ##FROM_EXPR join ##T",
    "FROM_EXPR => ##FROM_EXPR join ##JOIN",
    "JOIN => ##T on ##JOIN_ATOM",
    "JOIN => ##T on ##JOIN_CONJ",
    "JOIN_ATOM => ##T = ##T",
    "JOIN_CONJ => ##JOIN_ATOM and ##JOIN_ATOM",
    "JOIN_CONJ => ##JOIN_CONJ and ##JOIN_ATOM",
    "GROUP_EXPR => ##T",
    "GROUP_EXPR => ##GROUP_EXPR , ##T",
    "EXPR => ##EXPR_ATOM",
    "EXPR => ##EXPR_CONJ",
    "EXPR_CONJ => ##EXPR_ATOM or ##EXPR_ATOM",
    "EXPR_CONJ => ##EXPR_ATOM and ##EXPR_ATOM",
    "EXPR_CONJ => ##EXPR_ATOM , ##EXPR_ATOM",
    "EXPR_CONJ => ##EXPR_CONJ or ##EXPR_ATOM",
    "EXPR_CONJ => ##EXPR_CONJ and ##EXPR_ATOM",
    "EXPR_CONJ => ##EXPR_CONJ , ##EXPR_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM like ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM not like ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM in ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM not in ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM = ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM != ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM > ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM < ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM >= ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM <= ##SELECT_ATOM",
    "EXPR_ATOM => ##SELECT_ATOM ##BETWEEN",
    "BETWEEN => between - ##SELECT_ATOM and ##SELECT_ATOM",
    "BETWEEN => between ##SELECT_ATOM and ##SELECT_ATOM",
    "T => *",
    "T => ##T as ##T",
    "T => ##T . ##T",
    "ORDER_EXPR => ##ORDER_CONJ",
    "ORDER_EXPR => ##ORDER_ATOM",
    "ORDER_CONJ => ##ORDER_CONJ , ##ORDER_ATOM",
    "ORDER_CONJ => ##ORDER_ATOM , ##ORDER_ATOM",
    "ORDER_ATOM => ##SELECT_ATOM",
    "ORDER_ATOM => ##SELECT_ATOM desc",
    "ORDER_ATOM => ##SELECT_ATOM asc",
    "ORDER_ATOM => ##SELECT_ATOM > ##SELECT_ATOM",
    "ORDER_ATOM => ##SELECT_ATOM >= ##SELECT_ATOM",
    "ORDER_ATOM => ##SELECT_ATOM < ##SELECT_ATOM",
    "ORDER_ATOM => ##SELECT_ATOM <= ##SELECT_ATOM",
]


def parse_sql(sql, verbose=False):
  """Parse SQL string and return Node representing parse tree."""
  tokens = sql_tokenizer.tokenize_sql(sql)

  rules = []
  for rule_string in RULES:
    rules.append(Rule.from_string(rule_string))

  # Add rule `T -> token` for every non-keyword token in input.
  for token in set(tokens):
    if token not in KEYWORDS:
      rules.append(Rule("T", token))

  # Re-split on spaces.
  tokens = " ".join(tokens).split(" ")
  if verbose:
    print("Parsing SQL:")
    for idx, token in enumerate(tokens):
      print("%s - %s" % (idx, token))

  nodes = _run_parser(tokens, rules, verbose=verbose)
  if not nodes:
    raise ValueError("No parse for SQL: %s" % sql)
  elif len(nodes) > 1:
    raise ValueError("Multiple parses for SQL: %s (%s)" % (sql, nodes))
  return nodes[0]

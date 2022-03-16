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
"""Module to validate that targets are properly constructed.

The input is a CFG defining valid target constructions for a given task.
This can be viewed as a loose check that the target would be executable
for a given formalism and database.

This can be useful for NQG, which can otherwise over-generate syntactically
invalid targets as the grammars are restricted to a single non-terminal symbol.
"""

from language.compgen.csl.cky import cfg_converter
from language.compgen.csl.cky import cfg_parser
from language.compgen.csl.cky import cfg_rule
from tensorflow.io import gfile

# Used for string formatting.
NON_TERMINAL_PREFIX = "##"
ARROW = "=>"

# Root non-terminal symbol.
ROOT_SYMBOL = "ROOT"


class TargetCfgRule(object):
  """Represents a rule."""

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
    if len(symbols) < 2 or symbols[1] != ARROW:
      raise ValueError("Invalid rule_string: `%s`." % rule_string)
    lhs = symbols[0]
    rhs = " ".join(symbols[2:])
    return cls(lhs, rhs)


def rules_to_txt_file(rules, filename):
  """Write rules to txt file."""
  with gfile.GFile(filename, "w") as rule_file:
    for rule in rules:
      rule_file.write("%s\n" % str(rule))
  print("Wrote %s rules to %s." % (len(rules), filename))


def load_rules_from_file(filename):
  """Load list of TargetCfgRules from txt file."""
  rules = []
  with gfile.GFile(filename, "r") as rule_file:
    for line in rule_file:
      line = line.rstrip()
      # Allow blank lines and comment lines in grammar files starting with '#'.
      if line and not line.startswith("#"):
        rule = TargetCfgRule.from_string(line)
        rules.append(rule)
  print("Loaded %s rules from %s." % (len(rules), filename))
  return rules


def _populate_fn(unused_span_begin, unused_span_end, unused_parser_rule,
                 unused_children):
  # We are only interested in the presence of a parse, not the parse itself.
  # So, we use `True` to simply indicate the presence of some parse.
  return [True]


def _postprocess_fn(nodes):
  """Merge any nodes."""
  if nodes:
    return [True]
  else:
    return []


def can_parse(target_string,
              rules,
              max_single_nt_applications=2,
              verbose=False):
  """Returns True if there exists >=1 parse of target_string given rules."""
  tokens = target_string.split(" ")

  # Convert rules.
  converter = cfg_converter.CFGRuleConverter()
  parser_rules = []
  for rule_idx, rule in enumerate(rules):
    parser_rule = converter.convert_to_cfg_rule(
        lhs=rule.lhs,
        rhs=rule.rhs.split(" "),
        rule_idx=rule_idx,
        nonterminal_prefix=NON_TERMINAL_PREFIX,
        allowed_terminals=set(tokens))
    if parser_rule:
      parser_rules.append(parser_rule)

  start_idx = converter.nonterminals_to_ids[ROOT_SYMBOL]
  nonterminals = converter.nonterminals_to_ids.values()

  input_symbols = []
  for token in tokens:
    if token.startswith(NON_TERMINAL_PREFIX):
      idx = converter.nonterminals_to_ids[token[len(NON_TERMINAL_PREFIX):]]
      input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.NON_TERMINAL))
    else:
      if token not in converter.terminals_to_ids:
        return False
      idx = converter.terminals_to_ids[token]
      input_symbols.append(cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL))

  # Run parser.
  parses = cfg_parser.parse_symbols(
      input_symbols,
      parser_rules,
      nonterminals, {start_idx},
      _populate_fn,
      _postprocess_fn,
      verbose=verbose,
      max_single_nt_applications=max_single_nt_applications)

  if parses:
    return True
  else:
    return False

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

from language.nqg.common.cky import cfg_parser
from language.nqg.common.cky import cfg_rule

from tensorflow.io import gfile


# Used for string formatting.
NON_TERMINAL_PREFIX = "##"
ARROW = "=>"

# Root non-terminal symbol.
ROOT_SYMBOL = "ROOT"

# Special non-terminal that can match any terminal sequence.
ANYTHING = "ANYTHING"


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
    if symbols[1] != ARROW:
      raise ValueError("Invalid rule_string: %s." % rule_string)
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
      # Allow blank lines and comment lines in grammar files starting with '#'.
      if line and not line.startswith("#"):
        line = line.rstrip()
        rule = TargetCfgRule.from_string(line)
        rules.append(rule)
  print("Loaded %s rules from %s." % (len(rules), filename))
  return rules


def _convert_to_parser_rule(rule, terminals_to_ids, nonterminals_to_ids,
                            rule_idx):
  """Convert Rule to CFGRule."""
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
  lhs = nonterminals_to_ids[rule.lhs]
  parser_rule = cfg_rule.CFGRule(idx=rule_idx, lhs=lhs, rhs=rhs)
  return parser_rule


def _populate_fn(unused_span_begin, unused_span_end, unused_parser_rule,
                 unused_children):
  # We are only interested in the presence of a parse, not the parse itself.
  # So, we use `True` to simply indicate the presence of some parse.
  return True


def _postprocess_fn(nodes):
  """Merge any nodes."""
  if nodes:
    return [True]
  else:
    return []


def can_parse(target_string, rules, verbose=False):
  """Returns True if there exists >=1 parse of target_string given rules."""
  tokens = target_string.split(" ")

  # Add a rule for every span in target_string with lhs `ANYTHING`.
  anything_rules = []
  for start_idx in range(len(tokens)):
    for end_idx in range(start_idx + 1, len(tokens) + 1):
      rhs = " ".join(tokens[start_idx:end_idx])
      anything_rules.append(TargetCfgRule(ANYTHING, rhs))

  # Convert tokens to integer IDs.
  terminals_to_ids = {}
  for idx, token in enumerate(set(tokens)):
    terminals_to_ids[token] = idx
  input_ids = [terminals_to_ids[token] for token in tokens]

  # Generate non-terminal IDs.
  nonterminals_to_ids = {}
  nt_idx = 0
  for rule in rules + anything_rules:
    if rule.lhs not in nonterminals_to_ids:
      nonterminals_to_ids[rule.lhs] = nt_idx
      nt_idx += 1
  nonterminals = nonterminals_to_ids.values()
  start_idx = nonterminals_to_ids[ROOT_SYMBOL]

  # Convert rules.
  parser_rules = []
  for rule_idx, rule in enumerate(rules + anything_rules):
    parser_rule = _convert_to_parser_rule(rule, terminals_to_ids,
                                          nonterminals_to_ids, rule_idx)
    if parser_rule:
      parser_rules.append(parser_rule)

  # Run parser.
  parses = cfg_parser.parse(
      input_ids,
      parser_rules,
      nonterminals,
      start_idx,
      _populate_fn,
      _postprocess_fn,
      verbose=verbose)

  if parses:
    return True
  else:
    return False

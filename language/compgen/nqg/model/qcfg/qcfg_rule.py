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
"""Data structures for representing Quasi-Synchronous CFG (QCFG) rules.

Currently, both terminal and non-terminal symbols are simply represented
as strings, with special strings reserved for non-terminals.

QCFG rules used by NQG follow the following restrictions:
- There is only one non-terminal symbol, `NT`
- The only allowed non-terminal indexes are 1 and 2.
Therefore, we only need to reserve two strings to represent indexed
non-terminals.

We also expect all rules to be normalized as follows: a non-terminal with index
2 should never appear before a non-terminal with index 1 in the source
sequence.

Note that this data structure could potentially be improved:
1. A more flexible representation for terminal and non-terminal symbols
would avoid possible collisions between terminal and non-terminal symbols,
and allow for representing QCFGs that do not conform to the restrictions above.
2. Representing symbols as integers rather than strings may have computational
benefits for various operations over QCFG rules.
"""

import collections

# Represents the non-terminal symbol `NT` with linked index 1.
NT_1 = "NT_1"
# Represents the non-terminal symbol `NT` with linked index 2.
NT_2 = "NT_2"
# All other strings are assumed to represent terminal symbols.

# The LHS non-terminal is always assumed to be `NT` so is not represented.
QCFGRuleParent = collections.namedtuple(
    "QCFGRuleParent",
    [
        "source",  # Tuple of source symbols (strings).
        "target",  # Tuple of target symbols (strings).
        "arity",  # The number of unique non-terminal indexes (0, 1, or 2).
    ])

# Used for separating source and target sequences for string formatting.
SEPARATOR = "###"


# Define sub-class to override __str__ and __repr__ for easier debugging.
class QCFGRule(QCFGRuleParent):

  def __str__(self):
    return "%s %s %s" % (" ".join(self.source), SEPARATOR, " ".join(
        self.target))

  def __repr__(self):
    return str(self)


def _get_arity(source):
  if NT_1 in source and NT_2 in source:
    return 2
  if NT_1 in source:
    return 1
  if NT_2 in source:
    raise ValueError("Source is unnormalized: %s" % source)
  return 0


def rule_from_string(rule_str):
  """Parse rule in format 'source SEPARATOR target'."""
  splits = rule_str.split(SEPARATOR)
  if len(splits) != 2:
    raise ValueError("Invalid rule string: %s" % rule_str)
  source_str, target_str = splits
  source = source_str.strip().split()
  target = target_str.strip().split()
  arity = _get_arity(source)
  return QCFGRule(tuple(source), tuple(target), arity)


def apply_target(rule, substitutions):
  """Return target string with non-terminals replaced with substitutions."""
  if rule.arity != len(substitutions):
    raise ValueError
  output = []
  for token in rule.target:
    if token == NT_1:
      output.append(substitutions[0])
    elif token == NT_2:
      output.append(substitutions[1])
    else:
      output.append(token)
  return " ".join(output)

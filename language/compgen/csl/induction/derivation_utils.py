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
"""Utilities for derivations."""

from language.compgen.csl.qcfg import qcfg_parser
from language.compgen.csl.qcfg import qcfg_rule


class ChartNode(object):
  """Represents node in chart."""

  def __init__(self, source, span_begin, span_end, rule, children):
    self.span_begin = span_begin
    self.span_end = span_end
    self.children = children
    self.source_string = " ".join(source)

    child_targets = []
    new_rules = set()
    cur_span_begin = 0
    for child in children:
      cur_span_end = child.span_begin - span_begin
      span = source[cur_span_begin:cur_span_end]
      child_targets.extend(qcfg_rule.get_nts(span))
      child_targets.append(child.target_string)
      new_rules |= child.rules
      cur_span_begin = child.span_end - span_begin
    span = source[cur_span_begin:]
    child_targets.extend(qcfg_rule.get_nts(span))

    new_rules.add(rule)
    self.target_string = qcfg_rule.apply_target(rule, child_targets)
    self.rules = new_rules

  def __str__(self):
    return "(%d, %d) %s ### %s" % (self.span_begin, self.span_end,
                                   self.source_string, self.target_string)

  def __repr__(self):
    return self.__str__()


def generate_derivation(config, goal_rule, current_rules, verbose=False):
  """Attempt to find derivation of goal_rule given current_rules.

  Args:
    config: Config dict.
    goal_rule: Goal QCFGRule to derive.
    current_rules: A set of QCFGRule instances.
    verbose: Print debug output if True.

  Returns:
    Set of QCFGRule representing a derivation or None.
  """
  # Don't use the goal_rule to derive itself if included in current_rules.
  rules = current_rules - {goal_rule}
  source = " ".join(goal_rule.source)
  target = " ".join(goal_rule.target)

  def node_fn(span_begin, span_end, rule, children):
    node_source = goal_rule.source[span_begin:span_end]
    return ChartNode(node_source, span_begin, span_end, rule, children)

  def postprocess_cell_fn(nodes):
    """Filter and merge generated nodes."""
    targets_to_rules = {}
    for node in nodes:
      # Discard targets that are not substrings of the gold target.
      if node.target_string in target:
        # Keep one arbitrary derivation per target.
        # TODO(petershaw): Revisit this.
        targets_to_rules[node.target_string] = node
    new_nodes = list(targets_to_rules.values())
    return new_nodes

  tokens = source.split(" ")
  max_single_nt_applications = config.get("max_single_nt_applications", 2)
  outputs = qcfg_parser.parse(
      tokens,
      rules,
      verbose=verbose,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_cell_fn,
      max_single_nt_applications=max_single_nt_applications)

  for output in outputs:
    if output.target_string == target:
      return output.rules

  return None

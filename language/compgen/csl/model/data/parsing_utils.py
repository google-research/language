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
"""Utilities for generating parse forests for model training."""

import collections

from language.compgen.csl.qcfg import qcfg_parser
from language.compgen.csl.qcfg import qcfg_rule


# Special "rule" for scoring the root of the tree.
ROOT_RULE_KEY = "ROOT_RULE_KEY"


def _get_order(children):
  if not children:
    return 0
  return max([child.order for child in children]) + 1


class ForestNode(object):
  """Parent class representing a node in parse forest."""

  def __init__(self, span_begin, span_end, target_string, rule, children):
    self.span_begin = span_begin
    self.span_end = span_end
    self.target_string = target_string
    self.rule = rule
    self.children = children  # List of ForestNode.
    # The "order" is used for serializing nodes. We ensure that a parent node
    # has higher "order" than any of its children.
    self.order = _get_order(children)

  def __str__(self):
    node_type = self.__class__.__name__
    return "%s (%s, %s): %s, %s" % (node_type, self.span_begin, self.span_end,
                                    self.target_string, self.rule)

  def __repr__(self):
    return self.__str__()


def _unaggregate(nodes):
  """Return a list of unaggregated nodes."""
  new_nodes = []
  for node in nodes:
    if isinstance(node, AggregationNode):
      new_nodes.extend(node.children)
    else:
      new_nodes.append(node)
  return new_nodes


class AggregationNode(ForestNode):
  """Represents an aggregation over multiple nodes."""

  def __init__(self, children):
    target_string = children[0].target_string
    span_begin = children[0].span_begin
    span_end = children[0].span_end
    rule = children[0].rule
    # Avoid nested AggregationNodes.
    children = _unaggregate(children)
    # All nodes should have the same span, target_string, and rule.
    for node in children:
      if ((node.target_string, node.span_begin, node.span_end, node.rule) !=
          (target_string, span_begin, span_end, rule)):
        raise ValueError("Cannot aggreagate different spans or targets: %s" %
                         children)
    super(AggregationNode, self).__init__(span_begin, span_end, target_string,
                                          rule, children)


class RuleApplicationNode(ForestNode):
  """Represents an anchored rule application."""

  def __init__(self, rule, children, span_begin, span_end, target_string):
    super(RuleApplicationNode, self).__init__(span_begin, span_end,
                                              target_string, rule, children)


def _fingerprint(node):
  # We cannot merge nodes with two different rules because the rule determines
  # which embedding is used for scoring this node.
  return (node.target_string, node.rule)


def _aggregate(nodes, fingerprint_fn):
  """Returns list of nodes aggregated by target string."""
  fingerprints_to_nodes = collections.OrderedDict()
  aggregated_nodes = []
  for node in nodes:
    fingerprint = fingerprint_fn(node)
    if fingerprint not in fingerprints_to_nodes:
      fingerprints_to_nodes[fingerprint] = []
    fingerprints_to_nodes[fingerprint].append(node)
  for _, nodes in fingerprints_to_nodes.items():
    if len(nodes) > 1:
      aggregated_node = AggregationNode(nodes)
      aggregated_nodes.append(aggregated_node)
    else:
      aggregated_nodes.append(nodes[0])
  return aggregated_nodes


def filter_nodes(nodes, target_string):
  new_nodes = []
  for node in nodes:
    if node.target_string not in target_string:
      continue
    new_nodes.append(node)
  return new_nodes


def get_target_node(source, target, rules, max_single_nt_applications=1):
  """Return node corresponding to parses for target, or None."""
  tokens = source.split(" ")

  def node_fn(span_begin, span_end, rule, children):
    target_string = qcfg_rule.apply_target(
        rule, [node.target_string for node in children])
    return RuleApplicationNode(rule, children, span_begin, span_end,
                               target_string)

  def postprocess_fn(nodes):
    nodes = filter_nodes(nodes, target)
    return _aggregate(nodes, _fingerprint)

  nodes = qcfg_parser.parse(
      tokens,
      rules,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_fn,
      max_single_nt_applications=max_single_nt_applications)

  # Filter for nodes where target_string matches target exactly.
  ret_nodes = []
  for node in nodes:
    if node.target_string == target:
      ret_nodes.append(node)

  if not ret_nodes:
    return None

  # Add application and aggregation nodes for root.
  if len(ret_nodes) == 1:
    return RuleApplicationNode(ROOT_RULE_KEY, [ret_nodes[0]], 0, len(tokens),
                               target)
  elif len(ret_nodes) > 1:
    root_children = []
    for ret_node in ret_nodes:
      root_children.append(
          RuleApplicationNode(ROOT_RULE_KEY, [ret_node], 0, len(tokens),
                              target))
    return AggregationNode(root_children)
  else:
    raise ValueError

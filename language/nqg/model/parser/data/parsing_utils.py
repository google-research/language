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

from language.nqg.model.qcfg import qcfg_parser
from language.nqg.model.qcfg import qcfg_rule


class ForestNode(object):
  """Parent class representing a node in parse forest."""

  def __init__(self, span_begin, span_end, target_string, rule, children):
    self.span_begin = span_begin
    self.span_end = span_end
    self.target_string = target_string
    self.rule = rule  # Can be None for AggregationNode.
    self.children = children  # List of ForestNode.

  def __str__(self):
    node_type = self.__class__.__name__
    return "%s (%s, %s): %s, %s" % (node_type, self.span_begin,
                                    self.span_end,
                                    self.target_string,
                                    self.rule)

  def __repr__(self):
    return self.__str__()


class AggregationNode(ForestNode):
  """Represents an aggregation over multiple nodes."""

  def __init__(self, children):
    target_string = children[0].target_string
    span_begin = children[0].span_begin
    span_end = children[0].span_end
    # All nodes should have the same span and target_string.
    for node in children:
      if ((node.target_string, node.span_begin, node.span_end) !=
          (target_string, span_begin, span_end)):
        raise ValueError("Cannot aggreagate different spans or targets: %s" %
                         children)
    super(AggregationNode, self).__init__(span_begin, span_end, target_string,
                                          None, children)


class RuleApplicationNode(ForestNode):
  """Represents an anchored rule application."""

  def __init__(self, rule, children, span_begin, span_end, target_string):
    super(RuleApplicationNode, self).__init__(span_begin, span_end,
                                              target_string, rule, children)


def _fingerprint(node):
  return node.target_string


def _aggregate(nodes):
  """Returns list of nodes aggregated by target string."""
  fingerprints_to_nodes = collections.OrderedDict()
  aggregated_nodes = []
  for node in nodes:
    fingerprint = _fingerprint(node)
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


def get_target_node(source, target, rules):
  """Return node corresponding to parses for target, or None."""
  tokens = source.split(" ")

  def node_fn(span_begin, span_end, rule, children):
    target_string = qcfg_rule.apply_target(
        rule, [node.target_string for node in children])
    return RuleApplicationNode(rule, children, span_begin, span_end,
                               target_string)

  def postprocess_fn(nodes):
    nodes = filter_nodes(nodes, target)
    return _aggregate(nodes)

  nodes = qcfg_parser.parse(
      tokens, rules, node_fn=node_fn, postprocess_cell_fn=postprocess_fn)

  # Filter for nodes where target_string matches target exactly.
  ret_nodes = []
  for node in nodes:
    if node.target_string == target:
      ret_nodes.append(node)

  if not ret_nodes:
    return None

  if len(ret_nodes) > 1:
    raise ValueError

  return ret_nodes[0]


def get_merged_node(source, rules):
  """Return node corresponding to all parses."""
  tokens = source.split(" ")

  def node_fn(span_begin, span_end, rule, children):
    # Target string is ignored for this case.
    target_string = None
    return RuleApplicationNode(rule, children, span_begin, span_end,
                               target_string)

  def postprocess_fn(nodes):
    if len(nodes) > 1:
      return [AggregationNode(nodes)]
    else:
      return nodes

  nodes = qcfg_parser.parse(
      tokens, rules, node_fn=node_fn, postprocess_cell_fn=postprocess_fn)
  if len(nodes) != 1:
    raise ValueError("example `%s` len(nodes) != 1: %s" % (source, nodes))

  return nodes[0]

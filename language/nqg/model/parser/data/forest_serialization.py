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
"""Utilities for generating input tensors for parse forests.

The output of the QCFG parser used for pre-processing is a forest
representation of a set of parses. This representation factors common sub-trees
to represent exponentially many trees in an effecient manner.

In our TensorFlow graph, we want to sum over scores for the given set of parse
trees, using dynamic programming over the forest representation for effeciency.

Therefore, this module serializes the forest into a set of integer lists that
collectively represent a sequence of nodes, with child nodes always preceding
their parents. We create new nodes as necessary so that no node has more than
2 children.
"""

import collections

from language.nqg.model.parser.data import data_constants
from language.nqg.model.parser.data import parsing_utils


def _get_node_fingerprint(node):
  return id(node)


def _get_span_to_nodes_maps(root_node):
  """Return maps of span indexes to nodes."""
  node_stack = [root_node]
  seen_fingerprints = set()
  span_to_production_nodes = collections.defaultdict(list)
  span_to_aggregation_nodes = collections.defaultdict(list)
  while node_stack:
    node = node_stack.pop()
    fingerprint = _get_node_fingerprint(node)
    if fingerprint in seen_fingerprints:
      continue
    seen_fingerprints.add(fingerprint)

    if isinstance(node, parsing_utils.AggregationNode):
      for child in node.children:
        node_stack.append(child)
      span_to_aggregation_nodes[(node.span_begin, node.span_end)].append(node)
    elif isinstance(node, parsing_utils.RuleApplicationNode):
      for child in node.children:
        node_stack.append(child)
      span_to_production_nodes[(node.span_begin, node.span_end)].append(node)
    else:
      raise ValueError("Unexpected node type.")
  return span_to_production_nodes, span_to_aggregation_nodes


def get_forest_lists(root_node, num_tokens, application_idx_fn):
  """Get integer lists for serialized forest.

  Args:
    root_node: Root parsing_utils.ForestNode for parse forest.
    num_tokens: Number of tokens in input.
    application_idx_fn: Takes (span_begin, span_end, rule) and returns a idx.

  Returns:
    A tuple (node_type_list, node_1_idx_list, node_2_idx_list,
    application_idx_list, num_nodes). All of these are lists of integers
    with length equal to the number of nodes in the forest, except for num_nodes
    which is the integer number of nodes in the forest. The lists include
    the following information:
        node_type_list: Where node is of type AGGREGATION or RULE_APPLICATION.
        node_1_idx_list: If node has >= 1 children, this is the index of its
            first child. A node index refers to its index in these lists.
            If node has no children, will be -1.
        node_2_idx_list: If node has 2 children, this is the index of its
            second child, otherwise will be -1.
        application_idx_list: If node is of type RULE_APPLICATION, this is
            the index of the anchored rule application, where indexing is
            defined by application_idx_fn.
  """
  (span_to_production_nodes,
   span_to_aggregation_nodes) = _get_span_to_nodes_maps(root_node)

  # Setup empty lists.
  node_type_list = []
  node_1_idx_list = []
  node_2_idx_list = []
  application_idx_list = []

  # Map of fingerprints to index.
  fingerprint_to_idx = {}
  current_index = 0

  # Iterate through chart.
  for span_end in range(1, num_tokens + 1):
    for span_begin in range(span_end - 1, -1, -1):
      if (span_begin, span_end) in span_to_production_nodes:
        for node in span_to_production_nodes[(span_begin, span_end)]:
          fingerprint = _get_node_fingerprint(node)
          fingerprint_to_idx[fingerprint] = current_index
          current_index += 1

          if not isinstance(node, parsing_utils.RuleApplicationNode):
            raise ValueError

          node_type_list.append(data_constants.RULE_APPLICATION)

          if not node.children:
            node_1_idx_list.append(-1)
            node_2_idx_list.append(-1)
          elif len(node.children) == 1:
            node_1_idx_list.append(fingerprint_to_idx[_get_node_fingerprint(
                node.children[0])])
            node_2_idx_list.append(-1)
          elif len(node.children) == 2:
            node_1_idx_list.append(fingerprint_to_idx[_get_node_fingerprint(
                node.children[0])])
            node_2_idx_list.append(fingerprint_to_idx[_get_node_fingerprint(
                node.children[1])])
          else:
            raise ValueError

          application_idx_list.append(
              application_idx_fn(node.span_begin, node.span_end, node.rule))

        for node in span_to_aggregation_nodes[(span_begin, span_end)]:

          if not isinstance(node, parsing_utils.AggregationNode):
            raise ValueError

          node_type_list.append(data_constants.AGGREGATION)
          application_idx_list.append(-1)

          # Compute sum of first 2 nodes.
          node_1_fingerprint = _get_node_fingerprint(node.children[0])
          node_1_idx = fingerprint_to_idx[node_1_fingerprint]
          node_1_idx_list.append(node_1_idx)
          node_2_fingerprint = _get_node_fingerprint(node.children[1])
          node_2_idx = fingerprint_to_idx[node_2_fingerprint]
          node_2_idx_list.append(node_2_idx)
          current_index += 1

          # Sum the remaining.
          for idx in range(2, len(node.children)):
            node_type_list.append(data_constants.AGGREGATION)
            application_idx_list.append(-1)

            node_1_idx_list.append(current_index - 1)
            node_2_idx = fingerprint_to_idx[_get_node_fingerprint(
                node.children[idx])]
            node_2_idx_list.append(node_2_idx)
            current_index += 1

          # Point to last node for index.
          fingerprint = _get_node_fingerprint(node)
          fingerprint_to_idx[fingerprint] = current_index - 1

  num_nodes = current_index

  return (node_type_list, node_1_idx_list, node_2_idx_list,
          application_idx_list, num_nodes)

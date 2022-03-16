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

from language.compgen.csl.model import data_constants
from language.compgen.csl.model.data import parsing_utils


def _get_node_fingerprint(node):
  return id(node)


def _get_sorted_nodes(root_node):
  """Returns list of nodes sorted by order (ties broken by traversal order)."""
  node_stack = [root_node]
  seen_fingerprints = set()
  order_to_nodes_map = collections.defaultdict(list)
  while node_stack:
    node = node_stack.pop()
    fingerprint = _get_node_fingerprint(node)
    if fingerprint in seen_fingerprints:
      continue
    seen_fingerprints.add(fingerprint)
    order_to_nodes_map[node.order].append(node)
    for child in node.children:
      node_stack.append(child)

  orders = sorted(order_to_nodes_map.keys())
  sorted_nodes = []
  for order in orders:
    sorted_nodes.extend(order_to_nodes_map[order])
  return sorted_nodes


def get_forest_lists(root_node,
                     rhs_emb_idx_map,
                     lhs_emb_idx_map,
                     max_num_nts=2):
  """Get integer lists for serialized forest.

  Args:
    root_node: Root parsing_utils.ForestNode for parse forest.
    rhs_emb_idx_map: Map from (rule, nt_idx) to rhs_emb_idx.
    lhs_emb_idx_map: Map from rule to lhs_emb_idx.
    max_num_nts: The maximum number of nonterminals.

  Returns:
    A tuple (node_type_list, node_idx_list,
             rhs_emb_idx_list, lhs_emb_idx_list, num_nodes).
  """
  sorted_nodes = _get_sorted_nodes(root_node)

  # Setup empty lists.
  node_type_list = []
  node_idx_list = []
  rhs_emb_idx_list = []
  lhs_emb_idx_list = []

  # Map of fingerprints to index.
  fingerprint_to_idx = {}
  current_index = 0

  # Iterate through chart.
  for node in sorted_nodes:
    if isinstance(node, parsing_utils.RuleApplicationNode):
      fingerprint = _get_node_fingerprint(node)
      fingerprint_to_idx[fingerprint] = current_index
      current_index += 1

      node_type_list.append(data_constants.RULE_APPLICATION)

      num_children = len(node.children) if node.children else 0
      node_idx = [-1] * max_num_nts
      lhs_emb_idx = [-1] * max_num_nts
      rhs_emb_idx = [-1] * max_num_nts
      for nt_idx in range(num_children):
        child_fingerprint = _get_node_fingerprint(node.children[nt_idx])
        if child_fingerprint not in fingerprint_to_idx:
          raise ValueError("Child of %s not in `fingerprint_to_idx`: %s" %
                           (node, node.children[nt_idx]))
        node_idx[nt_idx] = fingerprint_to_idx[child_fingerprint]
        rhs_emb_idx[nt_idx] = rhs_emb_idx_map[(node.rule, nt_idx)]
        lhs_emb_idx[nt_idx] = lhs_emb_idx_map[node.children[nt_idx].rule]
      node_idx_list.append(node_idx)
      rhs_emb_idx_list.append(rhs_emb_idx)
      lhs_emb_idx_list.append(lhs_emb_idx)
    elif isinstance(node, parsing_utils.AggregationNode):
      node_1_fingerprint = _get_node_fingerprint(node.children[0])
      node_1_idx = fingerprint_to_idx[node_1_fingerprint]

      for idx in range(1, len(node.children)):
        node_type_list.append(data_constants.AGGREGATION)
        node_2_idx = fingerprint_to_idx[_get_node_fingerprint(
            node.children[idx])]

        node_idx = [-1] * max_num_nts
        node_idx[0] = node_1_idx
        node_idx[1] = node_2_idx
        node_idx_list.append(node_idx)

        lhs_emb_idx = [-1] * max_num_nts
        lhs_emb_idx_list.append(lhs_emb_idx)
        rhs_emb_idx = [-1] * max_num_nts
        rhs_emb_idx_list.append(rhs_emb_idx)

        node_1_idx = current_index
        current_index += 1

      # Point to last node for index.
      fingerprint = _get_node_fingerprint(node)
      fingerprint_to_idx[fingerprint] = current_index - 1
    else:
      raise ValueError

  num_nodes = current_index
  return (node_type_list, node_idx_list, rhs_emb_idx_list, lhs_emb_idx_list,
          num_nodes)

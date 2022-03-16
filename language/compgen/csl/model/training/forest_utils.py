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
"""Utilities for iterating over serialized parse forests in TensorFlow."""

from language.compgen.csl.model import data_constants
import tensorflow as tf


def get_forest_score_function(config, verbose=False):
  """Return forest_score_function."""
  # TODO(petershaw): In order to use TPU, it is likely necessary to consider
  # max_num_nodes as another input argument to initialize the arrays and
  # while loop.
  # However, this appears to still be insufficient for TPU compilation,
  # so this requires further investigation.

  @tf.function
  def forest_score_function(application_scores, num_nodes, node_type_list,
                            node_idx_list, rhs_emb_idx_list, lhs_emb_idx_list):
    """Iterate over nodes in forest and return score for root.

    Note that the returned score is not exponentiated, i.e. it is equivalent to
    the log of the sum of the exponentiated scores for individual parses in
    the forest:

    log(sum over parses(exp(sum over applications in parse(application score))))

    This function benefits from dynamic programming to compute this sum more
    efficiently.

    Args:
      application_scores: <float>[num_lhs_emb, num_rhs_emb].
      num_nodes: Integer number of nodes. By convention, the final non-padding
        node is the root node and should correspond to the `num_nodes - 1` index
        of the 3 `node_x` input tensors below.
      node_type_list: <int>[max_num_nodes].
      node_idx_list: <int>[max_num_nts, max_num_nodes].
      rhs_emb_idx_list: <int>[max_num_nts, max_num_nodes].
      lhs_emb_idx_list: <int>[max_num_nts, max_num_nodes].


    Returns:
      Score for root node (see description above).
    """

    # Write once / read array storing scores for each node.
    # Note that the scores are not exponentiated.
    node_array = tf.TensorArray(
        tf.float32,
        size=num_nodes,
        dynamic_size=False,
        clear_after_read=False,
        element_shape=[])
    max_num_nts = config["max_num_nts"]

    # Return early, i.e. iterate only for num_nodes not max_num_nodes.
    for idx in tf.range(num_nodes):
      node_type = node_type_list[idx]
      node_idx = node_idx_list[:, idx]
      rhs_emb_idx = rhs_emb_idx_list[:, idx]
      lhs_emb_idx = lhs_emb_idx_list[:, idx]

      if verbose:
        tf.print("idx:", idx)
        tf.print("node_type:", node_type)
        for i in range(max_num_nts):
          tf.print(f"node_idx[{i}]:", node_idx[i])
          tf.print(f"lhs_emb_idx[{i}]:", lhs_emb_idx[i])
          tf.print(f"rhs_emb_idx[{i}]:", rhs_emb_idx[i])

      if node_type == data_constants.RULE_APPLICATION:
        score = 0.0

        # Additionally, we add the scores for each child.
        for nonterminal_idx in range(max_num_nts):
          nt_node_idx = node_idx[nonterminal_idx]
          if nt_node_idx != -1:
            nt_lhs_emb_idx = lhs_emb_idx[nonterminal_idx]
            nt_rhs_emb_idx = rhs_emb_idx[nonterminal_idx]
            nt_score = application_scores[nt_lhs_emb_idx, nt_rhs_emb_idx]
            score += nt_score

            score += node_array.read(nt_node_idx)

        node_array = node_array.write(idx, score)

        if verbose:
          tf.print("Write RULE_APPLICATION node: ", idx, score)

      elif node_type == data_constants.AGGREGATION:
        # Merge nodes for common sub-trees.
        node_1_score = node_array.read(node_idx[0])
        node_2_score = node_array.read(node_idx[1])
        # Use logsumexp trick for stable calculation.
        score = tf.math.reduce_logsumexp(
            tf.stack([node_1_score, node_2_score]), axis=0)

        node_array = node_array.write(idx, score)

        if verbose:
          tf.print("Write AGGREGATION node: ", idx, score)

    # Reduce the final score using the `root` embedding.
    root_score = node_array.read(num_nodes - 1)
    return root_score

  return forest_score_function

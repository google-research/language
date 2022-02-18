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

from language.compgen.nqg.model.parser.data import data_constants

import tensorflow as tf


def get_forest_score_function(verbose=False):
  """Return forest_score_function."""
  # TODO(petershaw): In order to use TPU, it is likely necessary to consider
  # max_num_nodes as another input argument to initialize the arrays and
  # while loop.
  # However, this appears to still be insufficient for TPU compilation,
  # so this requires further investigation.

  @tf.function
  def forest_score_function(application_scores, num_nodes, node_type_list,
                            node_1_idx_list, node_2_idx_list,
                            node_application_idx_list):
    """Iterate over nodes in forest and return score for root.

    Note that the returned score is not exponentiated, i.e. it is equivalent to
    the log of the sum of the exponentiated scores for individual parses in
    the forest:

    log(sum over parses(exp(sum over applications in parse(application score))))

    This function benefits from dynamic programming to compute this sum more
    efficiently.

    Also note that input arguments should not be batched. This function could
    potentially be made more efficient by implementing a batched version of
    this computation. However, the computation in this function is limited to:
      1. Control flow (while loop) and TensorArray read/write operations
      2. Gather operations over application_scores
      2. Summation and logsumexp
    So the overall amount of computation should be small relative to
    large encoders and computation of application_scores. Using an
    implementation that is not batched also allows for returning early
    for examples where the number of nodes is less than the maximum limit.

    Args:
      application_scores: <float>[max_num_applications] of raw scores (not
        exponentiated) for anchored rule applications.
      num_nodes: Integer number of nodes. By convention, the final non-padding
        node is the root node and should correspond to the `num_nodes - 1` index
        of the 4 `node_x` input tensors below.
      node_type_list: <int>[max_num_nodes].
      node_1_idx_list: <int>[max_num_nodes].
      node_2_idx_list: <int>[max_num_nodes].
      node_application_idx_list: <int>[max_num_nodes].

    Returns:
      Score for root node (see description above).
    """
    if verbose:
      tf.print("application_scores:", application_scores, summarize=1000)

    # Write once / read array storing scores for each node.
    # Note that the scores are not exponentiated.
    node_array = tf.TensorArray(
        tf.float32,
        size=num_nodes,
        dynamic_size=False,
        clear_after_read=False,
        element_shape=[])

    # Return early, i.e. iterate only for num_nodes not max_num_nodes.
    for idx in tf.range(num_nodes):
      node_type = node_type_list[idx]
      node_1_idx = node_1_idx_list[idx]
      node_2_idx = node_2_idx_list[idx]
      node_application_idx = node_application_idx_list[idx]

      if verbose:
        tf.print("idx:", idx)
        tf.print("node_type:", node_type)
        tf.print("node_1_idx:", node_1_idx)
        tf.print("node_2_idx:", node_2_idx)
        tf.print("node_application_idx:", node_application_idx)

      if node_type == data_constants.RULE_APPLICATION:
        score = 0.0

        # All rule application nodes are associated with some application
        # score.
        score += application_scores[node_application_idx]

        # Additionally, we add the scores for any children.
        if node_1_idx != -1:
          score += node_array.read(node_1_idx)
        if node_2_idx != -1:
          score += node_array.read(node_2_idx)

        node_array = node_array.write(idx, score)

        if verbose:
          tf.print("Write RULE_APPLICATION node: ", idx, score)

      elif node_type == data_constants.AGGREGATION:
        # Merge nodes for common sub-trees.
        node_1_score = node_array.read(node_1_idx)
        node_2_score = node_array.read(node_2_idx)
        # Use logsumexp trick for stable calculation.
        score = tf.math.reduce_logsumexp(tf.stack([node_1_score, node_2_score]))

        node_array = node_array.write(idx, score)

        if verbose:
          tf.print("Write AGGREGATION node: ", idx, score)

    # Return final score (note that it is not exponentiated).
    return node_array.read(num_nodes - 1)

  return forest_score_function

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
"""Tests for forest_utils."""

import math

from language.compgen.nqg.model.parser.data import data_constants
from language.compgen.nqg.model.parser.training import forest_utils

import tensorflow as tf


class ForestUtilsTest(tf.test.TestCase):

  def test_forest_score(self):
    r"""Test computation of score for hypothetical parse forest.

    Parse forest consists of two parses:

         0
        / \
       1   2
          / \
         3   4

         0
        / \
       5   2
          / \
         3   4

    Where the integers represent the node_application_idx associated with the
    node. Below is the parse forest representing the above two parses:

           6(0)
          /    \
         /      \
       5(A)     2(2)
       / \       / \
      /   \     /   \
    4(1) 3(5) 1(3) 0(4)

    Where each node is drawn as X(Y), where:
      X - node_idx
      Y - node_application_idx or A for aggregation node
    """
    node_type = tf.constant(
        [
            data_constants.RULE_APPLICATION,  # node_idx=0
            data_constants.RULE_APPLICATION,  # node_idx=1
            data_constants.RULE_APPLICATION,  # node_idx=2
            data_constants.RULE_APPLICATION,  # node_idx=3
            data_constants.RULE_APPLICATION,  # node_idx=4
            data_constants.AGGREGATION,  # node_idx=5
            data_constants.RULE_APPLICATION,  # node_idx=6
        ],
        dtype=tf.int32)
    node_application_idx = tf.constant(
        [
            4,  # node_idx=0
            3,  # node_idx=1
            2,  # node_idx=2
            5,  # node_idx=3
            1,  # node_idx=4
            -1,  # node_idx=5
            0,  # node_idx=6
        ],
        dtype=tf.int32)
    node_1_idx = tf.constant(
        [
            -1,  # node_idx=0
            -1,  # node_idx=1
            1,  # node_idx=2
            -1,  # node_idx=3
            -1,  # node_idx=4
            4,  # node_idx=5
            5,  # node_idx=6
        ],
        dtype=tf.int32)
    node_2_idx = tf.constant(
        [
            -1,  # node_idx=0
            -1,  # node_idx=1
            0,  # node_idx=2
            -1,  # node_idx=3
            -1,  # node_idx=4
            3,  # node_idx=5
            2,  # node_idx=6
        ],
        dtype=tf.int32)

    application_scores_list = [
        1.0,  # node_application_idx=0
        2.0,  # node_application_idx=1
        3.0,  # node_application_idx=2
        -1.0,  # node_application_idx=3
        -2.0,  # node_application_idx=4
        -3.0,  # node_application_idx=5
    ]
    application_scores = tf.constant(application_scores_list, dtype=tf.float32)

    num_nodes = 7

    # Manually compute expected scores without dynamic programming.
    parse_1_application_idxs = [0, 1, 2, 3, 4]
    parse_2_application_idxs = [0, 5, 2, 3, 4]
    parse_1_score = math.exp(
        sum([application_scores_list[idx] for idx in parse_1_application_idxs]))
    parse_2_score = math.exp(
        sum([application_scores_list[idx] for idx in parse_2_application_idxs]))
    expected_forest_score = math.log(parse_1_score + parse_2_score)
    print("expected_forest_score: %s" % expected_forest_score)

    forest_score_function = forest_utils.get_forest_score_function(verbose=True)
    forest_score = forest_score_function(application_scores, num_nodes,
                                         node_type, node_1_idx, node_2_idx,
                                         node_application_idx)
    print("forest_score: %s" % forest_score)
    self.assertAlmostEqual(expected_forest_score, forest_score)


if __name__ == "__main__":
  tf.test.main()

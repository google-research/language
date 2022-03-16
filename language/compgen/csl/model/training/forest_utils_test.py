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

from language.compgen.csl.model import data_constants
from language.compgen.csl.model.training import forest_utils
import tensorflow as tf


class ForestUtilsTest(tf.test.TestCase):

  def test_forest_score(self):
    """Test computation of score for hypothetical parse forest."""
    # num_lhs_emb = 4
    # num_rhs_emb = 5
    application_scores = tf.constant(
        [[1.0, -1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0, 1.0]],
        dtype=tf.float32)
    node_type_list = tf.constant(
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

    node_idx_list = tf.stack([node_1_idx, node_2_idx], axis=0)

    rhs_emb_idx_list = tf.constant(
        [[1, 2, 3, 1, 2, 3, 1], [1, 2, 3, 1, 2, 3, 1]], dtype=tf.int32)
    lhs_emb_idx_list = tf.constant(
        [[1, 2, 3, 1, 2, 3, 1], [1, 2, 3, 1, 2, 3, 1]], dtype=tf.int32)

    num_nodes = 7

    config = {"model_dims": 2, "max_num_nts": 2}
    forest_score_function = forest_utils.get_forest_score_function(
        config, verbose=True)
    forest_score = forest_score_function(application_scores, num_nodes,
                                         node_type_list, node_idx_list,
                                         rhs_emb_idx_list, lhs_emb_idx_list)
    forest_score = forest_score.numpy()
    print("forest_score: %s" % forest_score)

    self.assertIsNotNone(forest_score)


if __name__ == "__main__":
  tf.test.main()

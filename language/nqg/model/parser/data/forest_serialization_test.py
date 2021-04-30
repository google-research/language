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
"""Tests for forest_serialization."""

from language.nqg.model.parser.data import forest_serialization
from language.nqg.model.parser.data import parsing_utils

from language.nqg.model.qcfg import qcfg_rule

import tensorflow as tf


def _mock_application_idx_fn(unused_span_begin, unused_span_end, unused_rule):
  return 1


class ForestSerializationTest(tf.test.TestCase):

  def test_get_node_tensors(self):
    source = "foo bar"
    target = "foo bar"
    rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
    ]
    target_node = parsing_utils.get_target_node(source, target, rules)

    num_tokens = 2
    (node_type_list, node_1_idx_list, node_2_idx_list, application_idx_list,
     num_nodes) = forest_serialization.get_forest_lists(
         target_node, num_tokens, _mock_application_idx_fn)

    print(node_type_list, "node_type_list")
    print(node_1_idx_list, "node_1_idx_list")
    print(node_2_idx_list, "node_2_idx_list")
    print(application_idx_list, "application_idx_list")
    print(num_nodes, "num_nodes")

    self.assertLen(node_type_list, num_nodes)
    self.assertLen(node_1_idx_list, num_nodes)
    self.assertLen(node_2_idx_list, num_nodes)
    self.assertLen(application_idx_list, num_nodes)

    self.assertEqual(num_nodes, 4)
    self.assertEqual(node_type_list, [1, 1, 1, 2])
    self.assertEqual(application_idx_list, [1, 1, 1, -1])
    self.assertEqual(node_1_idx_list, [-1, 0, -1, 2])
    self.assertEqual(node_2_idx_list, [-1, -1, -1, 1])


if __name__ == "__main__":
  tf.test.main()

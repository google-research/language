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
"""Tests for example_converter."""

from language.compgen.csl.model import test_utils
from language.compgen.csl.model.data import example_converter
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


class ExampleUtilsTest(tf.test.TestCase):

  def test_example_converter(self):
    config = test_utils.get_test_config()

    rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
        qcfg_rule.rule_from_string("foo bar ### baz"),
    ]

    converter = example_converter.ExampleConverter(rules, config)

    example = ("foo bar", "foo bar")
    tf_example = converter.convert(example)
    print(tf_example)

    feature_dict = tf_example.feature_lists.feature_list
    expected_keys = {
        "node_type_list",
        "node_idx_list",
        "rhs_emb_idx_list",
        "lhs_emb_idx_list",
        "num_nodes",
    }

    self.assertEqual(feature_dict.keys(), expected_keys)

    for key in expected_keys:
      for feature in feature_dict[key].feature:
        print("%s: %s" % (key, feature.int64_list.value))

    self.assertEqual(feature_dict["num_nodes"].feature[0].int64_list.value, [6])
    self.assertEqual(feature_dict["node_type_list"].feature[0].int64_list.value,
                     [1, 1, 1, 1, 1, 2, 0, 0])
    self.assertEqual(
        feature_dict["rhs_emb_idx_list"].feature[0].int64_list.value,
        [-1, -1, 1, 0, 0, -1, 0, 0])
    self.assertEqual(
        feature_dict["rhs_emb_idx_list"].feature[1].int64_list.value,
        [-1, -1, -1, -1, -1, -1, 0, 0])
    self.assertEqual(
        feature_dict["lhs_emb_idx_list"].feature[0].int64_list.value,
        [-1, -1, 0, 3, 1, -1, 0, 0])
    self.assertEqual(
        feature_dict["lhs_emb_idx_list"].feature[1].int64_list.value,
        [-1, -1, -1, -1, -1, -1, 0, 0])
    self.assertEqual(feature_dict["node_idx_list"].feature[0].int64_list.value,
                     [-1, -1, 0, 1, 2, 3, 0, 0])
    self.assertEqual(feature_dict["node_idx_list"].feature[1].int64_list.value,
                     [-1, -1, -1, -1, -1, 4, 0, 0])


if __name__ == "__main__":
  tf.test.main()

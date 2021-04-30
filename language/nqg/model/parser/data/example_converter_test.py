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

from language.nqg.model.parser import test_utils
from language.nqg.model.parser.data import example_converter
from language.nqg.model.qcfg import qcfg_rule

import tensorflow as tf


class ExampleUtilsTest(tf.test.TestCase):

  def test_example_converter(self):
    tokenizer = test_utils.MockTokenizer()
    config = test_utils.get_test_config()

    rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
        qcfg_rule.rule_from_string("foo bar ### baz"),
    ]

    converter = example_converter.ExampleConverter(rules, tokenizer, config)

    example = ("foo bar", "foo bar")
    tf_example = converter.convert(example)
    print(tf_example)

    feature_dict = tf_example.features.feature
    expected_keys = {
        "wordpiece_ids",
        "num_wordpieces",
        "application_span_begin",
        "application_span_end",
        "application_rule_idx",
        "nu_node_type",
        "nu_node_1_idx",
        "nu_node_2_idx",
        "nu_application_idx",
        "nu_num_nodes",
        "de_node_type",
        "de_node_1_idx",
        "de_node_2_idx",
        "de_application_idx",
        "de_num_nodes",
    }
    self.assertEqual(feature_dict.keys(), expected_keys)

    for key in expected_keys:
      print("%s: %s" % (key, feature_dict[key].int64_list.value))

    self.assertEqual(feature_dict["num_wordpieces"].int64_list.value, [4])
    self.assertEqual(feature_dict["wordpiece_ids"].int64_list.value,
                     [1, 5, 6, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["application_rule_idx"].int64_list.value,
                     [1, 4, 3, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["application_span_begin"].int64_list.value,
                     [1, 1, 1, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["application_span_end"].int64_list.value,
                     [2, 2, 2, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["nu_num_nodes"].int64_list.value, [4])
    self.assertEqual(feature_dict["nu_node_type"].int64_list.value,
                     [1, 1, 1, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["nu_application_idx"].int64_list.value,
                     [3, 0, 2, -1, 0, 0, 0, 0])
    self.assertEqual(feature_dict["nu_node_1_idx"].int64_list.value,
                     [-1, 0, -1, 2, 0, 0, 0, 0])
    self.assertEqual(feature_dict["nu_node_2_idx"].int64_list.value,
                     [-1, -1, -1, 1, 0, 0, 0, 0])
    self.assertEqual(feature_dict["de_num_nodes"].int64_list.value, [6])
    self.assertEqual(feature_dict["de_node_type"].int64_list.value,
                     [1, 1, 1, 1, 2, 2, 0, 0])
    self.assertEqual(feature_dict["de_application_idx"].int64_list.value,
                     [3, 0, 1, 2, -1, -1, 0, 0])
    self.assertEqual(feature_dict["de_node_1_idx"].int64_list.value,
                     [-1, 0, -1, -1, 3, 4, 0, 0])
    self.assertEqual(feature_dict["de_node_2_idx"].int64_list.value,
                     [-1, -1, -1, -1, 2, 1, 0, 0])


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for codelength_utils."""

from language.nqg.model.induction import codelength_utils

from language.nqg.model.qcfg import qcfg_rule

import tensorflow as tf


class CodelengthUtilsTest(tf.test.TestCase):

  def test_dataset_encoding_delta_1(self):
    current_ruleset = {
        qcfg_rule.rule_from_string("bar bar foo ### bar bar foo"),
        qcfg_rule.rule_from_string("foo ### foo"),
        qcfg_rule.rule_from_string("bar foo ### bar foo"),
        qcfg_rule.rule_from_string("bar bar NT_1 ### bar bar NT_1"),
    }

    dataset = [
        ("bar bar foo", "bar bar foo"),
        ("bar foo", "bar foo"),
        ("foo", "foo"),
    ]

    candidate_rule_to_add = qcfg_rule.rule_from_string("bar NT_1 ### bar NT_1")
    candidate_rules_to_remove = [
        qcfg_rule.rule_from_string("bar bar NT_1 ### bar bar NT_1")
    ]

    sample_size = 2
    dataset_encoding_delta = codelength_utils.get_dataset_encoding_delta(
        sample_size,
        dataset,
        current_ruleset,
        candidate_rule_to_add,
        candidate_rules_to_remove,
        verbose=True)
    self.assertEqual(dataset_encoding_delta, 0.0)

  def test_dataset_encoding_delta_2(self):
    current_ruleset = {
        qcfg_rule.rule_from_string("bar foo ### bar foo"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo ### foo"),
        qcfg_rule.rule_from_string("foo bar ### bar foo"),
    }

    dataset = [
        ("bar foo", "bar foo"),
        ("bar", "bar"),
        ("foo", "foo"),
        ("foo bar", "bar foo"),
    ]

    candidate_rule_to_add = qcfg_rule.rule_from_string(
        "NT_1 NT_2 ### NT_1 NT_2")
    candidate_rules_to_remove = [
        qcfg_rule.rule_from_string("bar foo ### bar foo"),
    ]

    sample_size = 4
    dataset_encoding_delta = codelength_utils.get_dataset_encoding_delta(
        sample_size,
        dataset,
        current_ruleset,
        candidate_rule_to_add,
        candidate_rules_to_remove,
        verbose=True)
    self.assertGreater(dataset_encoding_delta, 0.0)

  def test_dataset_encoding_delta_3(self):
    current_ruleset = {
        qcfg_rule.rule_from_string("NT_1 after NT_2 ### NT_2 NT_1"),
        qcfg_rule.rule_from_string("run ### I_RUN"),
        qcfg_rule.rule_from_string(
            "NT_1 right thrice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1"
        ),
        qcfg_rule.rule_from_string("NT_1 thrice ### NT_1 NT_1 NT_1"),
        qcfg_rule.rule_from_string("NT_1 thrice ### NT_1 I_WALK I_WALK"),
        qcfg_rule.rule_from_string("NT_1 thrice ### NT_1 I_RUN I_RUN"),
    }

    dataset = [
        ("run after run right thrice",
         "I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_RUN")
    ]

    candidate_rule_to_add = qcfg_rule.rule_from_string(
        "NT_1 right ### I_TURN_RIGHT NT_1")
    candidate_rules_to_remove = [
        qcfg_rule.rule_from_string(
            "NT_1 right thrice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1"
        ),
    ]

    sample_size = None
    dataset_encoding_delta = codelength_utils.get_dataset_encoding_delta(
        sample_size,
        dataset,
        current_ruleset,
        candidate_rule_to_add,
        candidate_rules_to_remove,
        verbose=True)
    self.assertAlmostEqual(dataset_encoding_delta, 2.1699250)


if __name__ == "__main__":
  tf.test.main()

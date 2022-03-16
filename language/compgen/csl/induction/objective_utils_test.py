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
"""Tests for objective_utils."""

from language.compgen.csl.induction import objective_utils
from language.compgen.csl.qcfg import qcfg_rule

import tensorflow as tf


def _get_config():
  return {
      "non_terminal_coef": 1.0,
      "terminal_coef": 8.0,
      "source_given_target_coef": 0.0,
      "target_given_source_coef": 1.0,
  }


class ObjectiveUtilsTest(tf.test.TestCase):

  def test_dataset_encoding_delta_1(self):
    dataset = [
        ("bar foo", "bar foo"),
        ("foo", "other"),
    ]

    candidate_rule = qcfg_rule.rule_from_string("foo ### foo")
    rules_to_remove = [qcfg_rule.rule_from_string("bar foo ### bar foo")]

    objective_calculator = objective_utils.ObjectiveCalculator(
        dataset, _get_config())

    size_delta = objective_calculator.get_candidate_size_delta(
        candidate_rule, rules_to_remove)
    self.assertEqual(size_delta, 16.0)

    penalty_delta = objective_calculator.get_candidate_penalty_delta(
        candidate_rule, rules_to_remove)
    self.assertEqual(penalty_delta, -1.0)


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for derivation_utils."""

from language.compgen.csl.induction import derivation_utils
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


def _get_config():
  return {}


class DerivationUtilsTest(tf.test.TestCase):

  def test_generate_derivation_1(self):
    goal_rule = qcfg_rule.rule_from_string("foo bar ### foo bar")
    rules = {
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
    }
    derivations = derivation_utils.generate_derivation(_get_config(), goal_rule,
                                                       rules)
    self.assertEqual(derivations, rules)

  def test_generate_derivation_2(self):
    goal_rule = qcfg_rule.rule_from_string("foo bar and NT_1 ### foo bar NT_1")
    rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
    ]
    derivations = derivation_utils.generate_derivation(_get_config(), goal_rule,
                                                       set(rules))
    self.assertEqual(derivations, {rules[0], rules[1], rules[2]})

  def test_generate_derivation_3(self):
    goal_rule = qcfg_rule.rule_from_string(
        "list NT_1 borders NT_2 ### "
        "answer ( intersection ( NT_2 , next_to_1 ( NT_1 ) ) )")
    rules = [
        qcfg_rule.rule_from_string("NT_1 ### answer ( NT_1 )"),
        qcfg_rule.rule_from_string(
            "NT_1 borders NT_2 ### intersection ( NT_2 , next_to_1 ( NT_1 ) )"),
        qcfg_rule.rule_from_string("list NT_1 ### NT_1"),
        qcfg_rule.rule_from_string(
            "list NT_1 borders NT_2 ### answer ( NT_1 , NT_2 )"),
    ]
    derivations = derivation_utils.generate_derivation(_get_config(), goal_rule,
                                                       set(rules))
    self.assertEqual(derivations, {rules[0], rules[1], rules[2]})

  def test_generate_derivation_4(self):
    goal_rule = qcfg_rule.rule_from_string(
        "foo foo bar NT_1 ### foo foo bar NT_1")
    rules = {
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
    }
    derivations = derivation_utils.generate_derivation(_get_config(), goal_rule,
                                                       rules)
    self.assertIsNone(derivations)


if __name__ == "__main__":
  tf.test.main()

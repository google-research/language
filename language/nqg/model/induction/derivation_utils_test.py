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

from language.nqg.model.induction import derivation_utils

from language.nqg.model.qcfg import qcfg_rule

import tensorflow as tf


class DerivationUtilsTest(tf.test.TestCase):

  def test_can_derive_1(self):
    goal_rule = qcfg_rule.rule_from_string("foo bar ### foo bar")
    rules = {
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_2(self):
    goal_rule = qcfg_rule.rule_from_string(
        "foo foo bar NT_1 ### foo foo bar NT_1")
    rules = {
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar NT_1 ### bar NT_1"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(
        goal_rule, rules, derived_rules, verbose=True)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_false_1(self):
    goal_rule = qcfg_rule.rule_from_string(
        "foo foo bar NT_1 ### foo foo bar NT_1")
    rules = {
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertFalse(can_derive)

  def test_can_derive_and_1(self):
    goal_rule = qcfg_rule.rule_from_string(
        "NT_1 and run twice ### NT_1 I_RUN I_RUN")
    rules = {
        qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1"),
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("run ### I_RUN"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_after_1(self):
    goal_rule = qcfg_rule.rule_from_string(
        "NT_1 after jump thrice ### I_JUMP I_JUMP I_JUMP NT_1")
    rules = {
        qcfg_rule.rule_from_string("NT_1 thrice ### NT_1 NT_1 NT_1"),
        qcfg_rule.rule_from_string("NT_1 after NT_2 ### NT_2 NT_1"),
        qcfg_rule.rule_from_string("jump ### I_JUMP"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_after_2(self):
    goal_rule = qcfg_rule.rule_from_string(
        "run after jump thrice ### I_JUMP I_JUMP I_JUMP I_RUN")
    rules = {
        qcfg_rule.rule_from_string("NT_1 thrice ### NT_1 NT_1 NT_1"),
        qcfg_rule.rule_from_string("NT_1 after NT_2 ### NT_2 NT_1"),
        qcfg_rule.rule_from_string("jump ### I_JUMP"),
        qcfg_rule.rule_from_string("run ### I_RUN"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_and_2(self):
    goal_rule = qcfg_rule.rule_from_string(
        "look and jump left ### I_LOOK I_TURN_LEFT I_JUMP")
    rules = {
        qcfg_rule.rule_from_string("NT_1 left ### I_TURN_LEFT NT_1"),
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("jump ### I_JUMP"),
        qcfg_rule.rule_from_string("look ### I_LOOK"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertTrue(can_derive)

  def test_can_derive_false_2(self):
    goal_rule = qcfg_rule.rule_from_string(
        "walk and run around right ### I_WALK I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN"
    )
    rules = {
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("NT_1 and run ### NT_1 I_RUN"),
        qcfg_rule.rule_from_string("walk ### I_WALK"),
        qcfg_rule.rule_from_string("NT_1 right ### I_TURN_RIGHT NT_1"),
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertFalse(can_derive)

  def test_can_derive_false_3(self):
    goal_rule = qcfg_rule.rule_from_string(
        "what are the major cities in the NT_1 through which the major river in NT_2 runs ### answer ( intersection ( major , intersection ( city , loc_2 ( intersection ( NT_1 , traverse_1 ( intersection ( major , intersection ( river , loc_2 ( NT_2 ) ) ) ) ) ) ) ) )"
    )
    rules = {
        qcfg_rule.rule_from_string("river ### river"),
        qcfg_rule.rule_from_string("river ### intersection ( river , NT_1 )"),
        qcfg_rule.rule_from_string("cities ### city"),
        qcfg_rule.rule_from_string("what are the NT_1 ### answer ( NT_1 )"),
        qcfg_rule.rule_from_string(
            "NT_1 in NT_2 ### intersection ( NT_1 , loc_2 ( NT_2 ) )"),
        qcfg_rule.rule_from_string(
            "what are the major NT_1 ### answer ( intersection ( major , NT_1 ) )"
        ),
        qcfg_rule.rule_from_string(
            "NT_1 in the NT_2 ### intersection ( NT_1 , loc_2 ( NT_2 ) )"),
        qcfg_rule.rule_from_string(
            "river in NT_1 ### intersection ( river , loc_2 ( NT_1 ) )"),
        qcfg_rule.rule_from_string(
            "the NT_1 NT_2 ### intersection ( NT_1 , NT_2 )")
    }
    derived_rules = set()
    can_derive = derivation_utils.can_derive(goal_rule, rules, derived_rules)
    print("derived_rules: %s" % derived_rules)
    self.assertFalse(can_derive)


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for rule_utils."""

from language.compgen.csl.induction import rule_utils
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


class RuleUtilsTest(tf.test.TestCase):

  def test_rhs_can_maybe_derive_true_1(self):
    rhs = tuple("NT_1 named NT_2".split())
    goal_rhs = tuple("NT_1 have a major city named NT_2".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_true_2(self):
    rhs = tuple("foo foo NT_1".split())
    goal_rhs = tuple("foo foo bar NT_1".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_false_1(self):
    rhs = tuple("NT_1 named NT_2".split())
    goal_rhs = tuple("NT_1 foo".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_true_4(self):
    rhs = tuple("NT_1 right".split())
    goal_rhs = tuple("run after run right thrice".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_true_5(self):
    rhs = tuple("I_TURN_RIGHT NT_1".split())
    goal_rhs = tuple(
        "I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_RUN".split(
        ))
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_false_2(self):
    rhs = tuple("NT_1 named NT_2".split())
    goal_rhs = tuple("NT_1 foo".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_1(self):
    rule_a = qcfg_rule.rule_from_string(
        "who is NT_1 ' s boss ? ### ( Yield :output ( FindManager :recipient ( NT_1 ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "who is NT_1 ? ### ( Yield :output ( NT_1 ) )")
    self.assertTrue(
        rule_utils.rhs_can_maybe_derive(rule_b.source, rule_a.source))
    self.assertTrue(
        rule_utils.rhs_can_maybe_derive(rule_b.target, rule_a.target))

  def test_rhs_can_maybe_derive_2(self):
    rhs = tuple("foo".split())
    goal_rhs = tuple("bar foo".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_3(self):
    rhs = tuple("foo".split())
    goal_rhs = tuple("foo bar".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_4(self):
    rhs = tuple("foo".split())
    goal_rhs = tuple("bar foo bar".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_5(self):
    rhs = tuple("NT_1 turn right NT_2".split())
    goal_rhs = tuple("turn left NT_1 NT_2 turn right".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_6(self):
    rhs = tuple("NT_1 turn left".split())
    goal_rhs = tuple("turn left turn right".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_7(self):
    rhs = tuple("turn right NT_1".split())
    goal_rhs = tuple("turn left turn right".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_8(self):
    rhs = tuple("foo NT_1 NT_2 bar".split())
    goal_rhs = tuple("foo foo bar buz buz".split())
    self.assertFalse(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_9(self):
    rhs = tuple("NT_1 ( capital )".split())
    goal_rhs = tuple("answer ( loc_1 ( smallest ( capital ) ) )".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))

  def test_rhs_can_maybe_derive_10(self):
    rhs = tuple("NT_1 foo bar".split())
    goal_rhs = tuple("NT_1 foo xyz foo bar".split())
    self.assertTrue(rule_utils.rhs_can_maybe_derive(rhs, goal_rhs))


if __name__ == "__main__":
  tf.test.main()

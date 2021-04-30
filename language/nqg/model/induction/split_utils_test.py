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
"""Tests for split_utils."""

from language.nqg.model.induction import split_utils

from language.nqg.model.qcfg import qcfg_rule

import tensorflow as tf


class SplitUtilsTest(tf.test.TestCase):

  def test_find_possible_splits_no_split(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 named NT_2 ### intersection ( NT_1 , NT_2 )")
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 have a major city named NT_2 ### intersection ( NT_1 , loc_1 ( intersection ( major , intersection ( city , NT_2 ) ) ) )"
    )
    ruleset = {rule_a}
    splits = split_utils.find_possible_splits(rule_b, ruleset)
    self.assertEmpty(splits)

  def test_find_possible_splits_1(self):
    rule_a = qcfg_rule.rule_from_string("foo NT_1 ### food NT_1")
    rule_b = qcfg_rule.rule_from_string("bar ### bar")
    rule_c = qcfg_rule.rule_from_string("foo bar ### foo bar")
    ruleset = {rule_a, rule_b}

    splits = split_utils.find_possible_splits(rule_c, ruleset)
    expected = qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_2(self):
    rule_a = qcfg_rule.rule_from_string("what is NT_1 ### answer ( NT_1 )")
    rule_b = qcfg_rule.rule_from_string(
        "what is NT_1 traversed by NT_2 ### answer ( intersection ( NT_1 , traverse_1 ( NT_2 ) ) )"
    )
    ruleset = {rule_a}

    splits = split_utils.find_possible_splits(rule_b, ruleset)
    expected = qcfg_rule.rule_from_string(
        "NT_1 traversed by NT_2 ### intersection ( NT_1 , traverse_1 ( NT_2 ) )"
    )
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_3(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 river ### intersection ( NT_1 , river )")
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 traversed by the NT_2 river ### intersection ( NT_1 , traverse_1 ( intersection ( NT_2 , river ) ) )"
    )
    ruleset = {rule_a}

    splits = split_utils.find_possible_splits(rule_b, ruleset)
    expected = qcfg_rule.rule_from_string(
        "NT_1 traversed by the NT_2 ### intersection ( NT_1 , traverse_1 ( NT_2 ) )"
    )
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_4(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 ' s star ### film.actor.film/ns:film.performance.film ( NT_1 )")
    rule_b = qcfg_rule.rule_from_string(
        "Which NT_1 was NT_2 ' s star ### answer ( film.actor.film/ns:film.performance.film ( NT_2 ) & NT_1 )"
    )
    ruleset = {rule_a}
    splits = split_utils.find_possible_splits(rule_b, ruleset)
    expected = qcfg_rule.rule_from_string(
        "Which NT_1 was NT_2 ### answer ( NT_2 & NT_1 )")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_5(self):
    rule_a = qcfg_rule.rule_from_string("Dutch ### nat ( m_059j2 )")
    rule_b = qcfg_rule.rule_from_string(
        "Who influenced M2 ' s Dutch child ### answer ( a(person) & influenced ( nat ( m_059j2 ) & parent ( M2 ) ) )"
    )
    ruleset = {rule_a}
    splits = split_utils.find_possible_splits(rule_b, ruleset)
    expected = qcfg_rule.rule_from_string(
        "Who influenced M2 ' s NT_1 child ### answer ( a(person) & influenced ( NT_1 & parent ( M2 ) ) )"
    )
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_6(self):
    rule_a = qcfg_rule.rule_from_string(
        "written and edited ### film.film.edited_by , film.film.written_by")
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 and edited ### film.film.edited_by , NT_1")
    ruleset = {rule_b}
    splits = split_utils.find_possible_splits(rule_a, ruleset)
    expected = qcfg_rule.rule_from_string("written ### film.film.written_by")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_7(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 ' s NT_2 director ### NT_2 & film.director.film ( NT_1 )")
    rule_b = qcfg_rule.rule_from_string("NT_1 ' s NT_2 ### NT_2 ( NT_1 )")
    ruleset = {rule_b}
    splits = split_utils.find_possible_splits(rule_a, ruleset)
    expected = qcfg_rule.rule_from_string(
        "NT_1 director ### NT_1 & film.director.film")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_8(self):
    rule_a = qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2")
    rule_b = qcfg_rule.rule_from_string("jump and NT_1 ### jump NT_1")
    ruleset = {rule_a}
    splits = split_utils.find_possible_splits(rule_b, ruleset)
    expected = qcfg_rule.rule_from_string("jump ### jump")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_9(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 right twice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1")
    ruleset = {rule_b}
    splits = split_utils.find_possible_splits(rule_a, ruleset)
    expected = qcfg_rule.rule_from_string("NT_1 right ### I_TURN_RIGHT NT_1")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_10(self):
    rule_a = qcfg_rule.rule_from_string("jump and NT_1 ### jump NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2")
    ruleset = {rule_b}
    splits = split_utils.find_possible_splits(rule_a, ruleset)
    expected = qcfg_rule.rule_from_string("jump ### jump")
    self.assertEqual({expected}, splits)

  def test_find_possible_splits_11(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 right twice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 right ### I_TURN_RIGHT NT_1")
    ruleset = {rule_b}
    splits = split_utils.find_possible_splits(rule_a, ruleset)
    expected = qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1")
    self.assertEqual({expected}, splits)


if __name__ == "__main__":
  tf.test.main()

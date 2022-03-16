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
"""Tests for unification_utils."""

from language.compgen.csl.induction import unification_utils
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


def _get_config():
  return {"allow_single_nt_target": True,
          "allow_repeated_target_nts": True,
          "max_num_nts": 3}


class UnificationUtilsTest(tf.test.TestCase):

  def test_get_rule_unifiers_no_split(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 have a major city named NT_2 ### intersection ( NT_1 , loc_1 ( intersection ( major , intersection ( city , NT_2 ) ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 named NT_2 ### intersection ( NT_1 , NT_2 )")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    self.assertEmpty(unifiers)

  def test_get_rule_unifiers_no_split_2(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 NT_2 NT_3 ### answer ( intersection ( NT_3 , loc_1 ( intersection ( NT_2 , NT_1 ) ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 NT_2 ### intersection ( NT_1 , NT_2 )"
    )

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    self.assertEmpty(unifiers)

  def test_get_rule_unifiers_1(self):
    rule_b = qcfg_rule.rule_from_string("bar ### bar")
    rule_a = qcfg_rule.rule_from_string("foo bar ### foo bar")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_2(self):
    rule_a = qcfg_rule.rule_from_string(
        "what is NT_1 traversed by NT_2 ### answer ( intersection ( NT_1 , traverse_1 ( NT_2 ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string("what is NT_1 ### answer ( NT_1 )")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "NT_1 traversed by NT_2 ### intersection ( NT_1 , traverse_1 ( NT_2 ) )"
    )
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_3(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 traversed by the NT_2 river ### intersection ( NT_1 , traverse_1 ( intersection ( NT_2 , river ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 river ### intersection ( NT_1 , river )")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "NT_1 traversed by the NT_2 ### intersection ( NT_1 , traverse_1 ( NT_2 ) )"
    )
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_4(self):
    rule_a = qcfg_rule.rule_from_string(
        "Which NT_1 was NT_2 ' s star ### answer ( film.actor.film/ns:film.performance.film ( NT_2 ) & NT_1 )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 ' s star ### film.actor.film/ns:film.performance.film ( NT_1 )")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "Which NT_1 was NT_2 ### answer ( NT_2 & NT_1 )")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_5(self):
    rule_a = qcfg_rule.rule_from_string(
        "Who influenced M2 ' s Dutch child ### answer ( a(person) & influenced ( nat ( m_059j2 ) & parent ( M2 ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string("Dutch ### nat ( m_059j2 )")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "Who influenced M2 ' s NT_1 child ### answer ( a(person) & influenced ( NT_1 & parent ( M2 ) ) )"
    )
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_6(self):
    rule_a = qcfg_rule.rule_from_string(
        "written and edited ### film.film.edited_by , film.film.written_by")
    rule_b = qcfg_rule.rule_from_string(
        "NT_1 and edited ### film.film.edited_by , NT_1")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("written ### film.film.written_by")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_7(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 ' s NT_2 director ### NT_2 & film.director.film ( NT_1 )")
    rule_b = qcfg_rule.rule_from_string("NT_1 ' s NT_2 ### NT_2 ( NT_1 )")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "NT_1 director ### NT_1 & film.director.film")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_8(self):
    rule_a = qcfg_rule.rule_from_string("jump and NT_1 ### jump NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("jump ### jump")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_9(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 right twice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("NT_1 right ### I_TURN_RIGHT NT_1")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_10(self):
    rule_a = qcfg_rule.rule_from_string("jump and NT_1 ### jump NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("jump ### jump")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_11(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 right twice ### I_TURN_RIGHT NT_1 I_TURN_RIGHT NT_1")
    rule_b = qcfg_rule.rule_from_string("NT_1 right ### I_TURN_RIGHT NT_1")
    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_12(self):
    rule_a = qcfg_rule.rule_from_string("foo NT_1 NT_2 ### NT_2 foo NT_1")
    rule_b = qcfg_rule.rule_from_string("foo ### foo")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("NT_1 NT_2 NT_3 ### NT_3 NT_1 NT_2")
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_13(self):
    rule_a = qcfg_rule.rule_from_string(
        "what NT_1 is the NT_2 in the m0 ### answer ( NT_2 ( intersection ( NT_1 , loc_2 ( m0 ) ) ) )"
    )
    rule_b = qcfg_rule.rule_from_string("in the m0 ### loc_2 ( m0 )")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "what NT_1 is the NT_2 NT_3 ### answer ( NT_2 ( intersection ( NT_1 , NT_3 ) ) )"
    )
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_14(self):
    rule_a = qcfg_rule.rule_from_string(
        "state has NT_1 major rivers NT_2 ### NT_1 ( state , NT_2 , intersection ( major , river ) )"
    )
    rule_b = qcfg_rule.rule_from_string(
        "state has NT_1 NT_2 NT_3 ### NT_1 ( state , NT_3 , NT_2 )"
    )

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string(
        "major rivers ### intersection ( major , river )"
    )
    self.assertEqual({expected}, unifiers)

  def test_get_rule_unifiers_single_nt(self):
    rule_a = qcfg_rule.rule_from_string("xyz NT_1 ### bar ( foo ( NT_1 ) )")
    rule_b = qcfg_rule.rule_from_string("NT_1 ### foo ( NT_1 )")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    expected = qcfg_rule.rule_from_string("xyz NT_1 ### bar ( NT_1 )")
    self.assertEqual({expected}, unifiers)

  def test_repeated_target_nts_1(self):
    rule_a = qcfg_rule.rule_from_string(
        "x NT_1 y z NT_2 ### X NT_2 Y NT_2 Z NT_2 W NT_1")
    rule_b = qcfg_rule.rule_from_string(
        "x NT_1 y NT_2 ### X NT_2 Y NT_2 W NT_1")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())

    # Should not include rule like `z NT_1 ### NT_1 Z NT_1` that has been formed
    # by replacing a sub-span of the target in rule_a which does not include
    # all of the NT_2s.
    self.assertEmpty(unifiers)

  def test_repeated_target_nts_2(self):
    rule_a = qcfg_rule.rule_from_string(
        "x NT_1 y z NT_2 ### X NT_2 Y NT_2 Z NT_2 W NT_1")
    rule_b = qcfg_rule.rule_from_string("z NT_1 ### NT_1 Z NT_1")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())

    # Should not include rule like `x NT_1 y NT_2 ### X NT_2 Y NT_2 W NT_1`.
    self.assertEmpty(unifiers)

  def test_repeated_target_nts_3(self):
    rule_a = qcfg_rule.rule_from_string(
        "NT_1 and jump twice ### NT_1 NT_1 JUMP")
    rule_b = qcfg_rule.rule_from_string("NT_1 and NT_2 ### NT_1 NT_2")
    rule_c = qcfg_rule.rule_from_string("jump twice ### NT_1 JUMP")

    # Should not include `rule_c`.
    self.assertEmpty(unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                         _get_config()))
    # Should not include `rule_b`.
    self.assertEmpty(unification_utils.get_rule_unifiers(rule_a, rule_c,
                                                         _get_config()))

  def test_get_rule_unifiers_repeated(self):
    rule_a = qcfg_rule.rule_from_string(
        "jump and jump twice ### JUMP JUMP JUMP")
    rule_b = qcfg_rule.rule_from_string("jump twice ### JUMP JUMP")

    unifiers = unification_utils.get_rule_unifiers(rule_a, rule_b,
                                                   _get_config())
    self.assertEmpty(unifiers)


if __name__ == "__main__":
  tf.test.main()

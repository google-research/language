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
"""Tests for qcfg_target_parser."""

from language.compgen.csl.qcfg import qcfg_target_parser
from language.compgen.csl.targets import target_grammar
import tensorflow as tf


def _format_parses(parses):
  new_parses = []
  for node in parses:
    new_parses.append([(begin, end, str(rule)) for begin, end, rule in node])
  return new_parses


# Track rules as tuples.
def _node_fn(span_begin, span_end, rule, children):
  rules = [(span_begin, span_end, rule)]
  for child in children:
    rules.extend(child)
  return [rules]


def _postprocess_fn(nodes):
  return nodes


class QcfgTargetParserTest(tf.test.TestCase):

  def test_parse(self):
    tokens = ["foo", "bar", "NT_1"]
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##C"),
        target_grammar.TargetCfgRule.from_string("C => bar ##A")
    ]
    parses = qcfg_target_parser.parse(tokens, rules, _node_fn, _postprocess_fn)
    self.assertLen(parses, 1)
    self.assertEqual(
        _format_parses(parses), [[(0, 3, "ROOT => foo ##C"),
                                  (1, 3, "C => bar ##A"),
                                  (2, 3, "A => ##NT_1")]])

  def test_parse_2(self):
    tokens = ["foo", "bar", "NT_1"]
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##C bar ##C"),
        target_grammar.TargetCfgRule.from_string("C => foo"),
        target_grammar.TargetCfgRule.from_string("ROOT => foo bar ##C")
    ]
    parses = qcfg_target_parser.parse(tokens, rules, _node_fn, _postprocess_fn)
    self.assertLen(parses, 2)
    self.assertEqual(
        _format_parses(parses), [[(0, 3, "ROOT => foo bar ##C"),
                                  (2, 3, "C => ##NT_1")],
                                 [(0, 3, "ROOT => ##C bar ##C"),
                                  (0, 1, "C => foo"), (2, 3, "C => ##NT_1")]])

  def test_parse_3(self):
    tokens = ["NT_2", "foo", "NT_1"]
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##C foo ##C"),
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##C"),
        target_grammar.TargetCfgRule.from_string("A => ##A foo"),
    ]
    parses = qcfg_target_parser.parse(tokens, rules, _node_fn, _postprocess_fn)
    self.assertLen(parses, 2)
    self.assertEqual(
        _format_parses(parses), [[(0, 3, "ROOT => ##C foo ##C"),
                                  (0, 1, "C => ##NT_2"), (2, 3, "C => ##NT_1")],
                                 [(0, 3, "ROOT => ##A ##C"),
                                  (0, 2, "A => ##A foo"), (0, 1, "A => ##NT_2"),
                                  (2, 3, "C => ##NT_1")]])

  def test_parse_4(self):
    tokens = ["NT_1", "foo", "NT_1"]
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##C foo ##C"),
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##C"),
        target_grammar.TargetCfgRule.from_string("A => ##A foo"),
    ]
    parses = qcfg_target_parser.parse(tokens, rules, _node_fn, _postprocess_fn)
    self.assertLen(parses, 2)
    self.assertEqual(
        _format_parses(parses), [[(0, 3, "ROOT => ##C foo ##C"),
                                  (0, 1, "C => ##NT_1"), (2, 3, "C => ##NT_1")],
                                 [(0, 3, "ROOT => ##A ##C"),
                                  (0, 2, "A => ##A foo"), (0, 1, "A => ##NT_1"),
                                  (2, 3, "C => ##NT_1")]])

  def test_parse_5(self):
    tokens = ["smallest", "(", "NT_1", ")"]
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => answer ( ##FUNC )"),
        target_grammar.TargetCfgRule.from_string(
            "FUNC => ##ARITY_1 ( ##FUNC )"),
        target_grammar.TargetCfgRule.from_string("ARITY_1 => smallest"),
    ]
    parses = qcfg_target_parser.parse(tokens, rules, _node_fn, _postprocess_fn)
    self.assertLen(parses, 1)
    self.assertEqual(
        _format_parses(parses), [[(0, 4, str(rules[1])), (0, 1, str(rules[2])),
                                  (2, 3, "FUNC => ##NT_1")]])

  def test_can_parse(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##C"),
        target_grammar.TargetCfgRule.from_string("C => bar ##A")
    ]
    self.assertTrue(qcfg_target_parser.can_parse("foo bar NT_1", rules))
    self.assertFalse(qcfg_target_parser.can_parse("bar foo NT_1", rules))

  def test_can_parse_2(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##C"),
        target_grammar.TargetCfgRule.from_string("A => ##A foo"),
    ]
    self.assertTrue(qcfg_target_parser.can_parse("NT_2 foo NT_1", rules))
    self.assertFalse(qcfg_target_parser.can_parse("NT_1 foo NT_1", rules))

  def test_can_parse_3(self):
    rules = [
        target_grammar.TargetCfgRule.from_string(
            "FUNC => ##ARITY_1 ( ##FUNC )"),
        target_grammar.TargetCfgRule.from_string("ARITY_1 => smallest"),
    ]
    self.assertTrue(qcfg_target_parser.can_parse("smallest ( NT_1 )", rules))
    self.assertTrue(qcfg_target_parser.can_parse("NT_2 ( NT_1 )", rules))
    self.assertFalse(qcfg_target_parser.can_parse("smallest ( city )", rules))
    self.assertFalse(qcfg_target_parser.can_parse("NT_1 ( city )", rules))

  def test_can_parse_4(self):
    rules = [
        target_grammar.TargetCfgRule.from_string(
            "ROOT => foo ##C ' ##QUOTED '"),
        target_grammar.TargetCfgRule.from_string("C => bar"),
        target_grammar.TargetCfgRule.from_string("QUOTED => buzz"),
    ]
    self.assertTrue(qcfg_target_parser.can_parse("foo bar ' buzz '", rules))
    self.assertTrue(qcfg_target_parser.can_parse("foo bar ' NT_1 '", rules))
    self.assertFalse(qcfg_target_parser.can_parse("foo bar", rules))
    self.assertFalse(qcfg_target_parser.can_parse("foo bar bar", rules))

  def test_can_parse_5(self):
    rules = [
        target_grammar.TargetCfgRule.from_string(
            "TIME => $am ( $hour ( ##NUMBER ) )"),
        target_grammar.TargetCfgRule.from_string(
            "DATETIME => $timeafterdatetime ( ##DATETIME ) ( ##TIME )"),
        target_grammar.TargetCfgRule.from_string(
            "DATETIME => now"),
        target_grammar.TargetCfgRule.from_string(
            "NUMBER => 1")
    ]
    self.assertFalse(
        qcfg_target_parser.can_parse(
            "$timeafterdatetime ( NT_1 ) ( $am ( $hour ( 9 ) ) )", rules))
    self.assertTrue(
        qcfg_target_parser.can_parse(
            "$timeafterdatetime ( now ) ( $am ( $hour ( 1 ) ) )", rules))
    self.assertTrue(
        qcfg_target_parser.can_parse(
            "$am ( $hour ( 1 ) )", rules))
    self.assertFalse(qcfg_target_parser.can_parse("foo bar bar", rules))

  def test_target_checker(self):
    rules = [
        target_grammar.TargetCfgRule.from_string(
            "TIME => $am ( $hour ( 1 ) )"),
        target_grammar.TargetCfgRule.from_string(
            "DATETIME => $timeafterdatetime ( ##DATETIME ) ( ##TIME )"),
        target_grammar.TargetCfgRule.from_string(
            "DATETIME => now")
    ]
    target_checker = qcfg_target_parser.TargetChecker(rules)

    self.assertTrue(
        target_checker.can_parse(
            "$timeafterdatetime ( NT_1 ) ( $am ( $hour ( 1 ) ) )".split()))
    self.assertTrue(
        target_checker.can_parse(
            "$timeafterdatetime ( now ) ( $am ( $hour ( 1 ) ) )".split()))
    self.assertTrue(
        target_checker.can_parse(
            "$am ( $hour ( 1 ) )".split()))
    self.assertFalse(target_checker.can_parse("foo bar bar".split()))


if __name__ == "__main__":
  tf.test.main()

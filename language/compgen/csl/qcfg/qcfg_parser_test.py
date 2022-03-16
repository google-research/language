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
"""Tests for qcfg_parser."""

from language.compgen.csl.qcfg import qcfg_parser
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


def _node_fn(unused_span_begin, unused_span_end, rule, children):
  """Nodes will represent target strings."""
  return qcfg_rule.apply_target(rule, children)


def _postprocess_cell_fn(nodes):
  return nodes


class QcfgParserTest(tf.test.TestCase):

  def test_parse(self):
    tokens = ["dax", "twice"]
    rules = [
        qcfg_rule.rule_from_string("dax ### DAX"),
        qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1"),
    ]
    parses = qcfg_parser.parse(tokens, rules, _node_fn, _postprocess_cell_fn)
    self.assertEqual(parses, ["DAX DAX"])

  def test_parse_flat(self):
    tokens = ["dax", "twice"]
    rules = [
        qcfg_rule.rule_from_string("dax twice ### DAX TWICE"),
    ]
    parses = qcfg_parser.parse(tokens, rules, _node_fn, _postprocess_cell_fn)
    self.assertEqual(parses, ["DAX TWICE"])

  def test_parse_3_nts(self):
    tokens = ["jump", "and", "walk", "after", "dax"]
    rules = [
        qcfg_rule.rule_from_string("jump ### JUMP"),
        qcfg_rule.rule_from_string("walk ### WALK"),
        qcfg_rule.rule_from_string("dax ### DAX"),
        qcfg_rule.rule_from_string(
            "NT_1 and NT_2 after NT_3 ### NT_1 NT_3 NT_2"),
    ]
    parses = qcfg_parser.parse(tokens, rules, _node_fn, _postprocess_cell_fn)
    self.assertEqual(parses, ["JUMP DAX WALK"])

  def test_parse_single_nt(self):
    tokens = ["list", "the", "states"]
    rules = [
        qcfg_rule.rule_from_string("list the states ### state"),
        qcfg_rule.rule_from_string("NT_1 ### answer ( NT_1 )"),
    ]
    parses = qcfg_parser.parse(tokens, rules, _node_fn, _postprocess_cell_fn)
    self.assertCountEqual(parses, ["state", "answer ( state )"])

  def test_parse_single_nt_2(self):
    tokens = ["where", "is", "m0"]
    rules = [
        qcfg_rule.rule_from_string("NT_1 ### answer ( NT_1 )"),
        qcfg_rule.rule_from_string("where is NT_1 ### NT_1"),
        qcfg_rule.rule_from_string("NT_1 ### loc_1 ( NT_1 )"),
        qcfg_rule.rule_from_string("m0 ### m0")
    ]
    parses = qcfg_parser.parse(tokens, rules, _node_fn, _postprocess_cell_fn)
    self.assertCountEqual(parses, [
        "m0", "answer ( m0 )", "loc_1 ( m0 )", "answer ( m0 )",
        "answer ( answer ( m0 ) )", "loc_1 ( answer ( m0 ) )", "loc_1 ( m0 )",
        "answer ( loc_1 ( m0 ) )", "loc_1 ( loc_1 ( m0 ) )"
    ])

  def test_can_parse(self):
    rules = [
        qcfg_rule.rule_from_string("dax ### DAX"),
        qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1"),
    ]
    can_parse = qcfg_parser.can_parse(
        source="dax twice", target="DAX DAX", rules=rules)
    self.assertTrue(can_parse)

  def test_can_parse_3_nts(self):
    rules = [
        qcfg_rule.rule_from_string("jump ### JUMP"),
        qcfg_rule.rule_from_string("walk ### WALK"),
        qcfg_rule.rule_from_string("dax ### DAX"),
        qcfg_rule.rule_from_string(
            "NT_1 and NT_2 after NT_3 ### NT_1 NT_3 NT_2"),
    ]
    can_parse = qcfg_parser.can_parse(
        source="jump and walk after dax", target="JUMP DAX WALK", rules=rules)
    self.assertTrue(can_parse)

  def test_can_parse_single_nt(self):
    rules = [
        qcfg_rule.rule_from_string("list the states ### state"),
        qcfg_rule.rule_from_string("NT_1 ### answer ( NT_1 )"),
    ]
    can_parse = qcfg_parser.can_parse(
        source="list the states", target="state", rules=rules)
    self.assertTrue(can_parse)
    can_parse = qcfg_parser.can_parse(
        source="list the states", target="answer ( state )", rules=rules)
    self.assertTrue(can_parse)


if __name__ == "__main__":
  tf.test.main()

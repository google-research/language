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

from language.nqg.model.qcfg import qcfg_parser
from language.nqg.model.qcfg import qcfg_rule

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

  def test_can_parse(self):
    rules = [
        qcfg_rule.rule_from_string("dax ### DAX"),
        qcfg_rule.rule_from_string("NT_1 twice ### NT_1 NT_1"),
    ]
    can_parse = qcfg_parser.can_parse(
        source="dax twice", target="DAX DAX", rules=rules)
    self.assertTrue(can_parse)


if __name__ == "__main__":
  tf.test.main()

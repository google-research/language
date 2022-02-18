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
"""Tests for cfg_parser."""

from absl.testing import parameterized

from language.compgen.nqg.common.cky import cfg_parser
from language.compgen.nqg.common.cky import cfg_rule

import tensorflow as tf

# Terminal IDs.
FOO = 1
BAR = 2

# Non-terminal IDs.
NT = 1


# Track anchored rule applications in derivation as tuples.
def _populate_fn(span_begin, span_end, rule, substitutions):
  anchored_rules = []
  anchored_rules.append((span_begin, span_end, rule.idx))
  for sub in substitutions:
    anchored_rules.extend(sub)
  return anchored_rules


# Use identity post-processing function.
def _postprocess_fn(nodes):
  return nodes


class CfgParserTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("use_trie_True", True),
                                  ("use_trie_False", False))
  def test_parse_1(self, use_trie):
    # NT -> BAR
    rule_1 = cfg_rule.CFGRule(
        idx=0, lhs=NT, rhs=(cfg_rule.CFGSymbol(BAR, cfg_rule.TERMINAL),))

    # NT -> FOO NT
    rule_2 = cfg_rule.CFGRule(
        idx=1,
        lhs=NT,
        rhs=(
            cfg_rule.CFGSymbol(FOO, cfg_rule.TERMINAL),
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
        ))

    input_ids = [FOO, FOO, BAR]

    parses = cfg_parser.parse(
        input_ids, [rule_1, rule_2], {NT},
        NT,
        _populate_fn,
        _postprocess_fn,
        use_trie=use_trie,
        verbose=True)
    self.assertLen(parses, 1)
    parse_node = parses[0]
    self.assertEqual(parse_node, [(0, 3, 1), (1, 3, 1), (2, 3, 0)])

  @parameterized.named_parameters(("use_trie_True", True),
                                  ("use_trie_False", False))
  def test_parse_2(self, use_trie):
    # NT -> BAR
    rule_1 = cfg_rule.CFGRule(
        idx=0, lhs=NT, rhs=(cfg_rule.CFGSymbol(BAR, cfg_rule.TERMINAL),))

    # NT -> NT FOO NT
    rule_2 = cfg_rule.CFGRule(
        idx=1,
        lhs=NT,
        rhs=(
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
            cfg_rule.CFGSymbol(FOO, cfg_rule.TERMINAL),
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
        ))

    # NT -> NT FOO BAR
    rule_3 = cfg_rule.CFGRule(
        idx=2,
        lhs=NT,
        rhs=(
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
            cfg_rule.CFGSymbol(FOO, cfg_rule.TERMINAL),
            cfg_rule.CFGSymbol(BAR, cfg_rule.TERMINAL),
        ))

    input_ids = [BAR, FOO, BAR]

    parses = cfg_parser.parse(
        input_ids, [rule_1, rule_2, rule_3], {NT},
        NT,
        _populate_fn,
        _postprocess_fn,
        use_trie=use_trie,
        verbose=True)
    self.assertLen(parses, 2)
    self.assertEqual(parses, [[(0, 3, 2),
                               (0, 1, 0)], [(0, 3, 1), (0, 1, 0), (2, 3, 0)]])


if __name__ == "__main__":
  tf.test.main()

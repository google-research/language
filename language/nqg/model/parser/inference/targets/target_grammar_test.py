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
"""Tests for target_grammar."""

from language.nqg.model.parser.inference.targets import target_grammar

import tensorflow as tf


class TargetGrammarTest(tf.test.TestCase):

  def test_can_parse_true(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##C"),
        target_grammar.TargetCfgRule.from_string("C => bar")
    ]
    self.assertTrue(target_grammar.can_parse("foo bar", rules, verbose=True))

  def test_can_parse_false(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##C"),
        target_grammar.TargetCfgRule.from_string("C => bar")
    ]
    self.assertFalse(target_grammar.can_parse("foo buzz", rules, verbose=True))

  def test_can_parse_wildcard_true(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##ANYTHING"),
    ]
    self.assertTrue(target_grammar.can_parse("foo buzz", rules, verbose=True))

  def test_can_parse_wildcard_false(self):
    rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => food ##ANYTHING"),
    ]
    self.assertFalse(target_grammar.can_parse("foo buzz", rules, verbose=True))

if __name__ == "__main__":
  tf.test.main()

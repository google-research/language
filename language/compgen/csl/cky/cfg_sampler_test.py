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
"""Tests for cfg_sampler."""

import random

from language.compgen.csl.cky import cfg_rule
from language.compgen.csl.cky import cfg_sampler
import mock
import tensorflow as tf

# Terminal IDs.
FOO = 1
BAR = 2

# Non-terminal IDs.
NT = 1


class CfgSamplerTest(tf.test.TestCase):

  @mock.patch.object(random, "choices")
  def test_sample(self, mock_random):
    mock_random.return_value = [0]
    # NT -> FOO NT
    rule_1 = cfg_rule.CFGRule(
        idx=0,
        lhs=NT,
        rhs=(
            cfg_rule.CFGSymbol(FOO, cfg_rule.TERMINAL),
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
        ))
    # NT -> BAR
    rule_2 = cfg_rule.CFGRule(
        idx=1, lhs=NT, rhs=(cfg_rule.CFGSymbol(BAR, cfg_rule.TERMINAL),))

    output = cfg_sampler.sample([rule_1, rule_2],
                                NT,
                                rule_values=[rule_1, rule_2],
                                verbose=True)
    self.assertEqual(output, [rule_1, [rule_2]])

  @mock.patch.object(random, "choices")
  def test_sample_2(self, mock_random):
    mock_random.return_value = [0]
    # NT -> FOO NT
    rule_1 = cfg_rule.CFGRule(
        idx=0,
        lhs=NT,
        rhs=(
            cfg_rule.CFGSymbol(FOO, cfg_rule.TERMINAL),
            cfg_rule.CFGSymbol(NT, cfg_rule.NON_TERMINAL),
        ))
    # NT -> BAR
    rule_2 = cfg_rule.CFGRule(
        idx=1, lhs=NT, rhs=(cfg_rule.CFGSymbol(BAR, cfg_rule.TERMINAL),))

    output = cfg_sampler.sample([rule_1, rule_2],
                                NT,
                                rule_values=[rule_1, rule_2],
                                max_recursion=2,
                                verbose=True)
    self.assertEqual(output, [rule_1, [rule_1, [rule_2]]])


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for sampler_utils."""

from language.compgen.csl.augment import sampler_utils
from language.compgen.csl.augment import test_utils as augment_test_utils
from language.compgen.csl.model import test_utils as model_test_utils
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.targets import target_grammar
import tensorflow as tf


def _sample_example(augment_config, model_dir, model_config, rules,
                    target_grammar_rules):
  sampler = sampler_utils.SamplerWrapper(
      augment_config=augment_config,
      model_dir=model_dir,
      model_config=model_config,
      rules=rules,
      target_grammar_rules=target_grammar_rules)
  example = sampler.sample_example(0)
  print("Example: (%s, %s)" % (example[0], example[1]))
  return example


class SamplerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(SamplerUtilsTest, self).setUp()
    self.rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
    ]
    self.target_grammar_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => foo ##NT"),
        target_grammar.TargetCfgRule.from_string("NT => bar")
    ]
    self.augment_config = augment_test_utils.get_test_config()
    self.model_config = model_test_utils.get_test_config()
    self.model_dir = self.get_temp_dir()

  def test_qcfg_sampler(self):
    example = _sample_example(
        augment_config=self.augment_config,
        model_dir=None,
        model_config=self.model_config,
        rules=self.rules,
        target_grammar_rules=None)
    self.assertIsNotNone(example)

  def test_qcfg_sampler_with_model(self):
    example = _sample_example(
        augment_config=self.augment_config,
        model_dir=self.model_dir,
        model_config=self.model_config,
        rules=self.rules,
        target_grammar_rules=None)
    self.assertIsNotNone(example)

  def test_joint_sampler(self):
    example = _sample_example(
        augment_config=self.augment_config,
        model_dir=None,
        model_config=self.model_config,
        rules=self.rules,
        target_grammar_rules=self.target_grammar_rules)
    self.assertIsNotNone(example)

  def test_joint_sampler_with_model(self):
    example = _sample_example(
        augment_config=self.augment_config,
        model_dir=self.model_dir,
        model_config=self.model_config,
        rules=self.rules,
        target_grammar_rules=self.target_grammar_rules)
    self.assertIsNotNone(example)


if __name__ == "__main__":
  tf.test.main()

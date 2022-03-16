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
"""Tests for joint_sampler."""

import os

from language.compgen.csl.augment import joint_sampler
from language.compgen.csl.model import test_utils
from language.compgen.csl.model.inference import inference_wrapper
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.targets import target_grammar
import numpy as np
import tensorflow as tf


class JointSamplerTest(tf.test.TestCase):

  def test_joint_rule_converter(self):
    target_cfg_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##B"),
        target_grammar.TargetCfgRule.from_string("A => a"),
        target_grammar.TargetCfgRule.from_string("A => and ( ##A , ##A )"),
        target_grammar.TargetCfgRule.from_string("B => b"),
        target_grammar.TargetCfgRule.from_string("B => and ( ##B , ##B )"),
    ]
    induced_rule = qcfg_rule.QCFGRule(
        tuple("NT_1 and NT_2".split()), tuple("and ( NT_1 , NT_2 )".split()), 2)
    converter = joint_sampler.JointRuleConverter(target_cfg_rules)
    joint_rule = converter.convert(induced_rule)
    self.assertEqual(joint_rule.qcfg_rule, induced_rule)
    self.assertEqual(joint_rule.cfg_nts_set, {
        ("A", "A", "A"),
        ("B", "B", "B"),
    })

  def test_joint_rule_converter_2(self):
    target_cfg_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##B"),
        target_grammar.TargetCfgRule.from_string("A => a"),
        target_grammar.TargetCfgRule.from_string("A => and ( ##A , ##B )"),
        target_grammar.TargetCfgRule.from_string("B => b"),
        target_grammar.TargetCfgRule.from_string("B => and ( ##B , ##A )"),
    ]
    induced_rule = qcfg_rule.QCFGRule(
        tuple("NT_1 and NT_2".split()), tuple("and ( NT_2 , NT_1 )".split()), 2)
    converter = joint_sampler.JointRuleConverter(target_cfg_rules)
    joint_rule = converter.convert(induced_rule)
    self.assertEqual(joint_rule.qcfg_rule, induced_rule)
    self.assertEqual(joint_rule.cfg_nts_set, {
        ("A", "B", "A"),
        ("B", "A", "B"),
    })

  def test_joint_sampler(self):
    target_cfg_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##B"),
        target_grammar.TargetCfgRule.from_string("A => a"),
        target_grammar.TargetCfgRule.from_string("A => and ( ##A , ##A )"),
        target_grammar.TargetCfgRule.from_string("B => b"),
        target_grammar.TargetCfgRule.from_string("B => and ( ##B , ##B )"),
    ]
    qcfg_rule_1 = qcfg_rule.QCFGRule(
        tuple("NT_1 NT_2".split()), tuple("NT_1 NT_2".split()), 2)
    qcfg_rule_2 = qcfg_rule.QCFGRule(
        tuple("NT_1 and NT_2".split()), tuple("and ( NT_1 , NT_2 )".split()), 2)
    qcfg_rule_3 = qcfg_rule.QCFGRule(("A",), ("a",), 0)
    qcfg_rule_4 = qcfg_rule.QCFGRule(("B",), ("b",), 0)

    sampler = joint_sampler.JointSampler.from_rules(
        target_cfg_rules, [qcfg_rule_1, qcfg_rule_2, qcfg_rule_3, qcfg_rule_4])
    sampled_source, sampled_target = sampler.sample()
    print("sampled: %s ### %s" %
          (" ".join(sampled_source), " ".join(sampled_target)))
    # The generated string should always contain these two characters based
    # on the target CFG.
    self.assertIn("A", sampled_source)
    self.assertIn("B", sampled_source)
    self.assertIn("a", sampled_target)
    self.assertIn("b", sampled_target)

  def test_joint_sampler_with_score_fn(self):
    target_cfg_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##B"),
        target_grammar.TargetCfgRule.from_string("A => a"),
        target_grammar.TargetCfgRule.from_string("A => and ( ##A , ##A )"),
        target_grammar.TargetCfgRule.from_string("B => b"),
        target_grammar.TargetCfgRule.from_string("B => and ( ##B , ##B )"),
    ]
    qcfg_rule_1 = qcfg_rule.QCFGRule(
        tuple("NT_1 NT_2".split()), tuple("NT_1 NT_2".split()), 2)
    qcfg_rule_2 = qcfg_rule.QCFGRule(
        tuple("NT_1 and NT_2".split()), tuple("and ( NT_1 , NT_2 )".split()), 2)
    qcfg_rule_3 = qcfg_rule.QCFGRule(("A",), ("a",), 0)
    qcfg_rule_4 = qcfg_rule.QCFGRule(("B",), ("b",), 0)
    qcfg_rules = [qcfg_rule_1, qcfg_rule_2, qcfg_rule_3, qcfg_rule_4]

    config = test_utils.get_test_config()
    wrapper = inference_wrapper.InferenceWrapper(qcfg_rules, config,
                                                 target_cfg_rules)
    wrapper.compute_application_scores(temperature=1, nonterminal_bias=1)

    def score_fn(parent_rule, nt_idx, child_rule):
      scores = np.exp(wrapper.application_scores)
      rhs_idx = wrapper.rhs_emb_idx_map[(parent_rule, nt_idx)]
      lhs_idx = wrapper.lhs_emb_idx_map[child_rule]
      return scores[lhs_idx, rhs_idx]

    sampler = joint_sampler.JointSampler.from_rules(
        target_cfg_rules, qcfg_rules)
    sampled_source, sampled_target = sampler.sample(score_fn=score_fn)
    print("sampled: %s ### %s" %
          (" ".join(sampled_source), " ".join(sampled_target)))
    # The generated string should always contain these two characters based
    # on the target CFG.
    self.assertIn("A", sampled_source)
    self.assertIn("B", sampled_source)
    self.assertIn("a", sampled_target)
    self.assertIn("b", sampled_target)

  def test_joint_sampler_save_and_load(self):
    target_cfg_rules = [
        target_grammar.TargetCfgRule.from_string("ROOT => ##A ##B"),
        target_grammar.TargetCfgRule.from_string("A => a"),
        target_grammar.TargetCfgRule.from_string("A => and ( ##A , ##A )"),
        target_grammar.TargetCfgRule.from_string("B => b"),
        target_grammar.TargetCfgRule.from_string("B => and ( ##B , ##B )"),
    ]
    qcfg_rule_1 = qcfg_rule.QCFGRule(
        tuple("NT_1 NT_2".split()), tuple("NT_1 NT_2".split()), 2)
    qcfg_rule_2 = qcfg_rule.QCFGRule(
        tuple("NT_1 and NT_2".split()), tuple("and ( NT_1 , NT_2 )".split()), 2)
    qcfg_rule_3 = qcfg_rule.QCFGRule(("A",), ("a",), 0)
    qcfg_rule_4 = qcfg_rule.QCFGRule(("B",), ("b",), 0)

    sampler = joint_sampler.JointSampler.from_rules(
        target_cfg_rules, [qcfg_rule_1, qcfg_rule_2, qcfg_rule_3, qcfg_rule_4])
    sampler_filepath = os.path.join(self.get_temp_dir(), "sampler.json")
    sampler.save(sampler_filepath)
    sampler2 = joint_sampler.JointSampler.from_file(sampler_filepath)
    self.assertEqual(str(sampler), str(sampler2))


if __name__ == "__main__":
  tf.test.main()

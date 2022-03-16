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
"""Tests for qcfg_sampler."""

from language.compgen.csl.augment import qcfg_sampler
from language.compgen.csl.model import test_utils
from language.compgen.csl.model.inference import inference_wrapper
from language.compgen.csl.qcfg import qcfg_rule
import numpy as np
import tensorflow as tf


class QCFGSamplerTest(tf.test.TestCase):

  def test_qcfg_sampler(self):
    qcfg_rules = [
        qcfg_rule.rule_from_string("NT_1 NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### and ( NT_1 , NT_2 )"),
        qcfg_rule.rule_from_string("A ### a"),
    ]

    sampler = qcfg_sampler.QCFGSampler(qcfg_rules)
    sampled_source, sampled_target = sampler.sample()
    print("sampled: %s ### %s" %
          (" ".join(sampled_source), " ".join(sampled_target)))
    self.assertIn("A", sampled_source)
    self.assertIn("a", sampled_target)

  def test_qcfg_sampler_with_score_fn(self):
    qcfg_rules = [
        qcfg_rule.rule_from_string("NT_1 NT_2 ### NT_1 NT_2"),
        qcfg_rule.rule_from_string("NT_1 and NT_2 ### and ( NT_1 , NT_2 )"),
        qcfg_rule.rule_from_string("A ### a"),
    ]

    config = test_utils.get_test_config()
    wrapper = inference_wrapper.InferenceWrapper(qcfg_rules, config)
    wrapper.compute_application_scores(temperature=1, nonterminal_bias=1)

    def score_fn(parent_rule, nt_idx, child_rule):
      scores = np.exp(wrapper.application_scores)
      rhs_idx = wrapper.rhs_emb_idx_map[(parent_rule, nt_idx)]
      lhs_idx = wrapper.lhs_emb_idx_map[child_rule]
      return scores[lhs_idx, rhs_idx]

    sampler = qcfg_sampler.QCFGSampler(qcfg_rules)
    sampled_source, sampled_target = sampler.sample(score_fn=score_fn)
    print("sampled: %s ### %s" %
          (" ".join(sampled_source), " ".join(sampled_target)))
    self.assertIn("A", sampled_source)
    self.assertIn("a", sampled_target)


if __name__ == "__main__":
  tf.test.main()

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
"""Class for generating predictions with weighted model."""

from language.compgen.csl.model import weighted_model
from language.compgen.csl.model.data import example_converter
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


class InferenceWrapper(object):
  """Provides interface for inference."""

  def __init__(self, rules, config, target_grammar_rules=None, verbose=False):
    self.config = config
    self.batch_size = 1
    self.model = weighted_model.Model(self.batch_size, config, training=False)
    current_step = tf.Variable(
        0, trainable=False, name="current_step", dtype=tf.int64)
    self.checkpoint = tf.train.Checkpoint(
        model=self.model, current_step=current_step)
    self.rules = rules
    self.lhs_emb_idx_map = example_converter.get_lhs_emb_idx_map(rules)
    self.rhs_emb_idx_map = example_converter.get_rhs_emb_idx_map(rules)
    self.verbose = verbose
    self.application_scores = None
    self.target_grammar_rules = target_grammar_rules

  def get_lhs_nonterminal_bias(self,
                               nonterminal_bias=0,
                               min_nonterminal_rule_arity=1):
    """Returns nonterminal bias for each lhs rule embedding."""
    lhs_nonterminal_bias = [0] * self.config["num_lhs_emb"]
    for rule in self.rules:
      rule_idx = self.lhs_emb_idx_map[rule]
      if (isinstance(rule, qcfg_rule.QCFGRule) and
          rule.arity >= min_nonterminal_rule_arity):
        lhs_nonterminal_bias[rule_idx] = nonterminal_bias
    # <float>[1, num_lhs_emb]
    return tf.expand_dims(
        tf.convert_to_tensor(lhs_nonterminal_bias, dtype=tf.float32), 0)

  def compute_application_scores(self,
                                 temperature=1,
                                 nonterminal_bias=0,
                                 min_nonterminal_rule_arity=1):
    """Compute application scores."""
    lhs_nonterminal_bias = self.get_lhs_nonterminal_bias(
        nonterminal_bias, min_nonterminal_rule_arity)
    if self.config.get("max_num_batch_embs", None):
      get_scores_fn = self.model.scoring_layer.get_scores_unstable
    else:
      get_scores_fn = self.model.scoring_layer.get_scores
    self.application_scores = get_scores_fn(
        temperature=temperature,
        lhs_nonterminal_bias=lhs_nonterminal_bias).numpy()

  def restore_checkpoint(self, latest_checkpoint):
    """Restore model parameters from checkpoint."""
    status = self.checkpoint.restore(latest_checkpoint)
    status.assert_existing_objects_matched()
    print("Restored checkpoint: %s" % latest_checkpoint)
    self.compute_application_scores()

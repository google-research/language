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
"""Class for generating predictions with NQG model."""

from language.nqg.model.parser import nqg_model
from language.nqg.model.parser.data import example_converter
from language.nqg.model.parser.data import tokenization_utils
from language.nqg.model.parser.inference import inference_parser
from language.nqg.model.parser.inference.targets import target_grammar

import tensorflow as tf


def _convert_to_int_tensor(values, padded_length):
  if len(values) > padded_length:
    raise ValueError("length %s is > %s" % (len(values), padded_length))
  for _ in range(len(values), padded_length):
    values.append(0)
  # Add outer dimension for batch size of 1.
  feature = tf.convert_to_tensor([values])
  return feature


def _get_score_fn(wordpiece_encodings, rules, model, token_start_wp_idx,
                  token_end_wp_idx):
  """Return score_fn."""
  # Assigns same rule to idx mapping as used for training.
  rule_key_to_idx_map = example_converter.get_rule_to_idx_map(rules)

  def score_fn(rule, span_begin, span_end):
    """Returns scalar score for anchored rule application."""
    application_span_begin = token_start_wp_idx[span_begin]
    # Need to convert between token index used by QCFG rules,
    # and wordpiece indexes used by neural model.
    # token_end_wp_idx is an *inclusive* idx.
    # span_end is an *exclusive* idx.
    # application_span_end is an *inclusive* idx.
    application_span_end = token_end_wp_idx[span_end - 1]
    application_rule_idx = rule_key_to_idx_map[rule]
    application_score = model.application_score_layer.score_application(
        wordpiece_encodings, application_span_begin, application_span_end,
        application_rule_idx)
    return application_score.numpy()

  return score_fn


class InferenceWrapper(object):
  """Provides interface for inference."""

  def __init__(self,
               tokenizer,
               rules,
               config,
               bert_config,
               target_grammar_rules=None,
               verbose=False):
    self.tokenizer = tokenizer
    self.config = config
    self.batch_size = 1
    self.model = nqg_model.Model(
        self.batch_size, config, bert_config, training=False)
    self.checkpoint = tf.train.Checkpoint(model=self.model)
    self.rules = rules
    self.target_grammar_rules = target_grammar_rules
    self.verbose = verbose

  def restore_checkpoint(self, latest_checkpoint):
    """Restore model parameters from checkpoint."""
    status = self.checkpoint.restore(latest_checkpoint)
    status.assert_existing_objects_matched()
    print("Restored checkpoint: %s" % latest_checkpoint)

  def get_output(self, source):
    """Returns (one-best target string, score) or (None, None)."""
    # Tokenize.
    tokens = source.split(" ")
    (wordpiece_ids, num_wordpieces, token_start_wp_idx,
     token_end_wp_idx) = tokenization_utils.get_wordpiece_inputs(
         tokens, self.tokenizer)
    wordpieces_batch = _convert_to_int_tensor(wordpiece_ids,
                                              self.config["max_num_wordpieces"])

    # Run encoder.
    wordpiece_encodings_batch = self.model.get_wordpiece_encodings(
        wordpieces_batch, [[num_wordpieces]])
    wordpiece_encodings = wordpiece_encodings_batch[0]

    # Create score_fn.
    score_fn = _get_score_fn(wordpiece_encodings, self.rules, self.model,
                             token_start_wp_idx, token_end_wp_idx)

    # Run parser.
    target_string, score = inference_parser.run_inference(
        source, self.rules, score_fn)

    # Validate target if target CFG provided.
    if (target_string and self.target_grammar_rules and
        not target_grammar.can_parse(target_string, self.target_grammar_rules)):
      if self.verbose:
        print("Invalid target: %s" % target_string)
      return None, None

    return target_string, score

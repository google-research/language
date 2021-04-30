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
"""Tests for nqg_model."""

from language.nqg.model.parser import nqg_model
from language.nqg.model.parser import test_utils

import numpy as np
import tensorflow as tf


class NqgModelUtilsTest(tf.test.TestCase):

  def test_application_scores(self):
    config = test_utils.get_test_config()
    application_score_layer = nqg_model.ApplicationScoreLayer(config)
    batch_size = config["batch_size"]
    bert_dims = 16
    wordpiece_encodings = tf.constant(
        np.random.randn(batch_size, config["max_num_wordpieces"], bert_dims),
        dtype=tf.float32)
    application_span_begin = tf.constant(
        np.random.randint(0, config["max_num_wordpieces"],
                          [batch_size, config["max_num_applications"]]),
        dtype=tf.int32)
    application_span_end = tf.constant(
        np.random.randint(0, config["max_num_wordpieces"],
                          [batch_size, config["max_num_applications"]]),
        dtype=tf.int32)
    application_rule_idx = tf.constant(
        np.random.randint(0, config["max_num_rules"],
                          [batch_size, config["max_num_applications"]]),
        dtype=tf.int32)

    application_scores = application_score_layer(wordpiece_encodings,
                                                 application_span_begin,
                                                 application_span_end,
                                                 application_rule_idx)

    print("application_scores: %s" % application_scores)
    self.assertEqual(application_scores.shape,
                     (batch_size, config["max_num_applications"]))

  def test_get_wordpiece_encodings(self):
    config = test_utils.get_test_config()
    batch_size = config["batch_size"]
    bert_config = test_utils.get_test_bert_config()
    model = nqg_model.Model(
        batch_size, config, bert_config, training=True, verbose=False)

    wordpiece_ids_batch = tf.constant(
        np.random.randint(0, bert_config.vocab_size,
                          [batch_size, config["max_num_wordpieces"]]),
        dtype=tf.int32)

    num_wordpieces = tf.constant([[3]] * batch_size, dtype=tf.int32)

    wordpiece_encodings = model.get_wordpiece_encodings(wordpiece_ids_batch,
                                                        num_wordpieces)
    print("wordpiece_encodings: %s" % wordpiece_encodings)
    self.assertEqual(
        wordpiece_encodings.shape,
        (batch_size, config["max_num_wordpieces"], bert_config.hidden_size))


if __name__ == "__main__":
  tf.test.main()

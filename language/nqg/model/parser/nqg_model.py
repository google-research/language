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
"""Defines the NQG neural parsing model.

The parsing model consists of a BERT encoder, feed forward layers to
compute vector representations for spans, and an embedding table for rules.

The model produces scores for anchored rule applications, which are based on
the span representations of the anchored span and a learned embedding for each
rule.

Note that the model is implemented in TensorFlow 2.x, based on the TF 2.x BERT
implementation here:
https://github.com/tensorflow/models/tree/master/official/nlp/bert

You can find documentation for downloading or converting BERT checkpoints to
be compatible with this implementation here:
https://github.com/tensorflow/models/tree/master/official/nlp/bert#pre-trained-models
"""

import tensorflow as tf

from official.nlp.bert import bert_models


def _feed_forward(output_dims, hidden_dims, name):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(hidden_dims, activation="relu", name="%s_1" % name),
      tf.keras.layers.Dense(output_dims, name="%s_2" % name)
  ],
                             name=name)


class ApplicationScoreLayer(tf.keras.layers.Layer):
  """Layer for computing scores for anchored rule applications.

  Span begin and end indexes should both be *inclusive*, i.e.
  for a span consisting of a single token the span begin and end indexes
  will be the same.

  It is up to the caller to establish consistent indexing of rules,
  as this layer simply allocates an embedding table of size equal to the
  max_num_rules in the config.
  """

  def __init__(self, config):
    super(ApplicationScoreLayer, self).__init__()
    self.feed_forward = _feed_forward(
        config["model_dims"], config["model_dims"], name="application_ffn")
    self.span_feed_forward = _feed_forward(
        1, config["model_dims"], name="span_ffn")
    self.rule_embeddings = tf.keras.layers.Embedding(config["max_num_rules"],
                                                     config["model_dims"])
    self.config = config

  def score_application(self, wordpiece_encodings, application_span_begin,
                        application_span_end, application_rule_idx):
    """Computes scores for a single anchored rule applications.

    Args:
      wordpiece_encodings: <float>[max_num_wordpieces, bert_dims]
      application_span_begin: <int>[1]
      application_span_end: <int>[1]
      application_rule_idx: <int>[1]

    Returns:
      application_score: <float>[1]
    """
    # <float>[bert_dims]
    span_begin_encoding = tf.gather(wordpiece_encodings, application_span_begin)
    span_end_encoding = tf.gather(wordpiece_encodings, application_span_end)
    # <float>[bert_dims * 2]
    span_encoding = tf.concat([span_begin_encoding, span_end_encoding], axis=0)
    # <float>[1, bert_dims * 2]
    span_encoding = tf.expand_dims(span_encoding, 0)
    # <float>[1, model_dims]
    span_ffn_encoding = self.feed_forward(span_encoding)

    # <float>[model_dims]
    application_rule_embedddings = self.rule_embeddings(application_rule_idx)
    # <float>[model_dims, 1]
    application_rule_embedddings = tf.expand_dims(application_rule_embedddings,
                                                  1)

    # <float>[1, 1]
    application_score = tf.matmul(span_ffn_encoding,
                                  application_rule_embedddings)
    # <float>[]
    application_score = tf.squeeze(application_score, [0, 1])

    # <float>[1, 1]
    span_score = self.span_feed_forward(span_encoding)
    # <float>[]
    span_score = tf.squeeze(span_score, [0, 1])

    return application_score + span_score

  def call(self, wordpiece_encodings, application_span_begin,
           application_span_end, application_rule_idx):
    """Computes scores for a batch of anchored rule applications.

    Args:
      wordpiece_encodings: <float>[batch_size, max_num_wordpieces, bert_dims]
      application_span_begin: <int>[batch_size, max_num_applications]
      application_span_end: <int>[batch_size, max_num_applications]
      application_rule_idx: <int>[batch_size, max_num_applications]

    Returns:
      application_scores: <float>[batch_size, max_num_applications]
    """
    # <float>[batch_size, max_num_applications, bert_dims]
    span_begin_encoding = tf.gather(
        wordpiece_encodings, application_span_begin, batch_dims=1)
    span_end_encoding = tf.gather(
        wordpiece_encodings, application_span_end, batch_dims=1)
    # <float>[batch_size, max_num_applications, bert_dims * 2]
    span_encodings = tf.concat([span_begin_encoding, span_end_encoding], axis=2)
    # <float>[batch_size, max_num_applications, model_dims]
    span_encodings_ffn = self.feed_forward(span_encodings)
    # <float>[batch_size, max_num_applications, 1, model_dims]
    span_encodings_ffn = tf.expand_dims(span_encodings_ffn, 2)

    # <float>[batch_size, max_num_applications, model_dims]
    application_rule_embedddings = self.rule_embeddings(application_rule_idx)
    # <float>[batch_size, max_num_applications, model_dims, 1]
    application_rule_embedddings = tf.expand_dims(application_rule_embedddings,
                                                  3)

    # <float>[batch_size, max_num_applications, 1, 1]
    application_scores = tf.matmul(span_encodings_ffn,
                                   application_rule_embedddings)
    # <float>[batch_size, max_num_applications]
    application_scores = tf.squeeze(application_scores, [2, 3])

    # <float>[batch_size, max_num_applications, 1]
    span_scores = self.span_feed_forward(span_encodings)
    # <float>[batch_size, max_num_applications]
    span_scores = tf.squeeze(span_scores, [2])

    return application_scores + span_scores


class Model(tf.keras.layers.Layer):
  """Defines NQG neural parsing model."""

  def __init__(self, batch_size, config, bert_config, training, verbose=False):
    super(Model, self).__init__()
    self.config = config
    self.bert_encoder = bert_models.get_transformer_encoder(
        bert_config, sequence_length=self.config["max_num_wordpieces"])
    self.application_score_layer = ApplicationScoreLayer(config)
    self.training = training
    self.batch_size = batch_size

  def call(self, wordpiece_ids_batch, num_wordpieces, application_span_begin,
           application_span_end, application_rule_idx):
    """Returns scores for a batch of anchored rule applications.

    Args:
      wordpiece_ids_batch: <int>[batch_size, max_num_wordpieces]
      num_wordpieces: <int>[batch_size, 1]
      application_span_begin: <int>[batch_size, max_num_applications]
      application_span_end: <int>[batch_size, max_num_applications]
      application_rule_idx: <int>[batch_size, max_num_applications]

    Returns:
      application_scores: <float>[batch_size, max_num_applications]
    """
    wordpiece_encodings_batch = self.get_wordpiece_encodings(
        wordpiece_ids_batch, num_wordpieces)
    application_scores_batch = self.application_score_layer(
        wordpiece_encodings_batch, application_span_begin, application_span_end,
        application_rule_idx)
    return application_scores_batch

  def get_wordpiece_encodings(self, wordpiece_ids_batch, num_wordpieces):
    """Returns contextualized encodings for a batch of wordpieces.

    Args:
      wordpiece_ids_batch: <int>[batch_size, max_num_wordpieces]
      num_wordpieces: <int>[batch_size, 1]

    Returns:
      wordpiece_encodings: <float>[batch_size, max_num_wordpieces, bert_dims]
    """
    num_wordpieces = tf.squeeze(num_wordpieces, 1)
    bert_input_mask = tf.sequence_mask(
        num_wordpieces, self.config["max_num_wordpieces"], dtype=tf.int32)
    bert_type_ids = tf.zeros(
        shape=[self.batch_size, self.config["max_num_wordpieces"]],
        dtype=tf.int32)
    wordpiece_encodings_batch, unused_cls_output = self.bert_encoder(
        [wordpiece_ids_batch, bert_input_mask, bert_type_ids],
        training=self.training)
    return wordpiece_encodings_batch

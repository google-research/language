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
"""Defines a weighted parsing model with latent states (i.e. types)."""

import tensorflow as tf


def _log_matmul_exp(tensor_a, tensor_b):
  """Numerically stable equivalent of log-matmul-exp computation.

  Is equivalent to:
    tf.math.log(
      tf.matmul(tf.math.exp(tensor_a),
                tf.math.exp(tensor_b)))

  See:
  https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

  Args:
    tensor_a: <float>[L, M].
    tensor_b: <float>[M, N].

  Returns:
    <float>[L, N].
  """
  max_a = tf.math.reduce_max(tensor_a, axis=1, keepdims=True)
  max_b = tf.math.reduce_max(tensor_b, axis=0, keepdims=True)
  tensor_a -= max_a
  tensor_b -= max_b
  tensor_c = tf.matmul(tf.math.exp(tensor_a), tf.math.exp(tensor_b))
  tensor_c = tf.math.log(tensor_c)
  tensor_c += max_a
  tensor_c += max_b
  return tensor_c


class ApplicationScoringLayer(tf.keras.layers.Layer):
  """Layer for interfacing with rule application scores."""

  def __init__(self, config):
    super(ApplicationScoringLayer, self).__init__()
    weights_init = tf.random_normal_initializer()

    self.rhs_type_scores = tf.Variable(
        initial_value=weights_init(
            shape=(config["num_rhs_emb"], config["num_types"]),
            dtype="float32"),
        trainable=True,
    )

    self.type_lhs_scores = tf.Variable(
        initial_value=weights_init(
            shape=(config["num_types"], config["num_lhs_emb"]),
            dtype="float32"),
        trainable=True,
    )

  def get_scores(self,
                 temperature=1,
                 lhs_nonterminal_bias=None,
                 rhs_emb_idxs=None,
                 lhs_emb_idxs=None,
                 approximate_denominator=False):
    """Computes scores.

    P(lhs|rhs) = sum over types { P(type|rhs) * P(lhs|type) }
    logP(lhs|rhs) = log sum over types { P(type|rhs) * P(lhs|type) }

    Can be computed as below for greater numerical stability:

    logP(lhs|rhs) = log sum exp over types { logP(type|rhs) + logP(lhs|type) }

    P(type|rhs) is "softmax" of score(rhs, type) normalized across type.
    P(lhs|type) is "softmax" of score(type, lhs) normalized across lhs.

    Args:
      temperature: Float indicating temperature for softmax (1 = standard
        softmax).
      lhs_nonterminal_bias: <float>[1, num_lhs_emb].
      rhs_emb_idxs: <int32>[num_rhs_emb_idxs].
      lhs_emb_idxs: <int32>[num_lhs_emb_idxs].
      approximate_denominator: Bool indicating whether to approximate the
        denominator of the softmax over rules by only considering rules in the
        current batch.

    Returns:
      <float>[num_lhs_emb, num_rhs_emb]
    """
    rhs_type_scores = self.rhs_type_scores
    type_lhs_scores = self.type_lhs_scores
    if rhs_emb_idxs is not None:
      rhs_type_scores = tf.gather(rhs_type_scores, rhs_emb_idxs, axis=0)
    # Gather *before* the softmax if approximate_denominator.
    if lhs_emb_idxs is not None and approximate_denominator:
      type_lhs_scores = tf.gather(type_lhs_scores, lhs_emb_idxs, axis=1)
    if lhs_nonterminal_bias is None:
      lhs_nonterminal_bias = tf.zeros_like(type_lhs_scores)

    # <float>[num_rhs_emb, num_types]
    logp_type_given_rhs = tf.nn.log_softmax(
        rhs_type_scores / temperature, axis=1)
    # <float>[num_types, num_lhs_emb]
    logp_lhs_given_type = tf.nn.log_softmax(
        type_lhs_scores / temperature + lhs_nonterminal_bias, axis=1)

    # Gather *after* the softmax to not affect denominator.
    if lhs_emb_idxs is not None and not approximate_denominator:
      logp_lhs_given_type = tf.gather(logp_lhs_given_type, lhs_emb_idxs, axis=1)

    # Avoid broadcast to large matrix.
    logp_lhs_given_rhs = _log_matmul_exp(logp_type_given_rhs,
                                         logp_lhs_given_type)

    # Transpose to provide expected return type.
    logp_lhs_given_rhs_t = tf.transpose(logp_lhs_given_rhs)
    return logp_lhs_given_rhs_t

  def get_scores_unstable(self, temperature=1, lhs_nonterminal_bias=None):
    """Same as get_scores but is less numerically stable."""
    # TODO(petershaw): Consider removing this and just using the above method?
    if lhs_nonterminal_bias is None:
      lhs_nonterminal_bias = tf.zeros_like(self.type_lhs_scores)
    # <float>[num_rhs_emb, num_types]
    type_given_rhs = tf.nn.softmax(self.rhs_type_scores / temperature, axis=1)
    # <float>[num_types, num_lhs_emb]
    lhs_given_type = tf.nn.softmax(
        self.type_lhs_scores / temperature + lhs_nonterminal_bias, axis=1)
    # Avoid broadcast to large matrix.
    logp_lhs_given_rhs = tf.math.log(tf.matmul(type_given_rhs, lhs_given_type))
    logp_lhs_given_rhs_t = tf.transpose(logp_lhs_given_rhs)
    return logp_lhs_given_rhs_t


class Model(tf.keras.layers.Layer):
  """Wraps rule application scores."""

  def __init__(self, batch_size, config, training, verbose=False):
    super(Model, self).__init__()
    self.config = config
    self.scoring_layer = ApplicationScoringLayer(config)
    self.training = training
    self.batch_size = batch_size

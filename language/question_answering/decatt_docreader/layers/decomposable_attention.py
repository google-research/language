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
"""Implementation of decomposable attention model.

 https://arxiv.org/abs/1606.01933.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.layers import common_layers
from language.common.utils import tensor_utils

import tensorflow as tf


def decomposable_attention(emb1, len1, emb2, len2, hidden_size, hidden_layers,
                           dropout_ratio, mode, epsilon=1e-8):
  """See https://arxiv.org/abs/1606.01933.

  Args:
    emb1: A Tensor with shape [batch_size, max_len1, emb_size] representing the
        first input sequence.
    len1: A Tensor with shape [batch_size], indicating the true sequence length
        of `emb1`. This is required due to padding.
    emb2: A Tensor with shape [batch_size, max_len2, emb_size] representing the
        second input sequence.
    len2: A Tensor with shape [batch_size], indicating the true sequence length
        of `emb1`. This is required due to padding.
    hidden_size: An integer indicating the size of each hidden layer in the
        feed-forward neural networks.
    hidden_layers: An integer indicating the number of hidden layers in the
        feed-forward neural networks.
    dropout_ratio: The probability of dropping out each unit in the activation.
        This can be None, and is only applied during training.
    mode: One of the keys from tf.estimator.ModeKeys.
    epsilon: A small positive constant to add to masks for numerical stability.

  Returns:
    final_emb: A Tensor with shape [batch_size, hidden_size].
  """
  # [batch_size, maxlen1]
  mask1 = tf.sequence_mask(len1, tensor_utils.shape(emb1, 1), dtype=tf.float32)

  # [batch_size, maxlen2]
  mask2 = tf.sequence_mask(len2, tensor_utils.shape(emb2, 1), dtype=tf.float32)

  with tf.variable_scope("attend"):
    projected_emb1 = common_layers.ffnn(
        emb1, [hidden_size] * hidden_layers, dropout_ratio, mode)
  with tf.variable_scope("attend", reuse=True):
    projected_emb2 = common_layers.ffnn(
        emb2, [hidden_size] * hidden_layers, dropout_ratio, mode)

  # [batch_size, maxlen1, maxlen2]
  attention_scores = tf.matmul(projected_emb1, projected_emb2, transpose_b=True)
  attention_weights1 = tf.nn.softmax(
      attention_scores + tf.log(tf.expand_dims(mask2, 1) + epsilon), 2)
  attention_weights2 = tf.nn.softmax(
      attention_scores + tf.log(tf.expand_dims(mask1, 2) + epsilon), 1)

  # [batch_size, maxlen1, emb_size]
  attended_emb1 = tf.matmul(attention_weights1, emb2)

  # [batch_size, maxlen2, emb_size]
  attended_emb2 = tf.matmul(attention_weights2, emb1, transpose_a=True)

  with tf.variable_scope("compare"):
    compared_emb1 = common_layers.ffnn(
        tf.concat([emb1, attended_emb1], -1),
        [hidden_size] * hidden_layers,
        dropout_ratio, mode)
  with tf.variable_scope("compare", reuse=True):
    compared_emb2 = common_layers.ffnn(
        tf.concat([emb2, attended_emb2], -1),
        [hidden_size] * hidden_layers,
        dropout_ratio, mode)

  compared_emb1 *= tf.expand_dims(mask1, -1)
  compared_emb2 *= tf.expand_dims(mask2, -1)

  # [batch_size, hidden_size]
  aggregated_emb1 = tf.reduce_sum(compared_emb1, 1)
  aggregated_emb2 = tf.reduce_sum(compared_emb2, 1)
  with tf.variable_scope("aggregate"):
    final_emb = common_layers.ffnn(
        tf.concat([aggregated_emb1, aggregated_emb2], -1),
        [hidden_size] * hidden_layers,
        dropout_ratio,
        mode)
  return final_emb

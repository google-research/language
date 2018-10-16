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
"""Commonly used layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.inputs import char_utils
from language.common.utils import tensor_utils
import tensorflow as tf


def ffnn(input_emb, hidden_sizes, dropout_ratio, mode):
  """A simple feed-forward neural network.

  Dropout is applied before each linear layer. Non-linearities are not applied
  to the final hidden layer.

  Args:
    input_emb: tensor<float> [..., embedding_size]
    hidden_sizes: list<int> [hidden_size_1, hidden_size_2, ...]
    dropout_ratio: The probability of dropping out each unit in the activation.
        This can be None, and is only applied during training.
    mode: One of the keys from tf.estimator.ModeKeys.

  Returns:
    output_emb: A Tensor with the same shape as `input_emb`, except for the last
        dimension which will have size `hidden_sizes[-1]` instead.
  """
  for i, h in enumerate(hidden_sizes):
    with tf.variable_scope("ffnn_{}".format(i)):
      if mode == tf.estimator.ModeKeys.TRAIN and dropout_ratio is not None:
        input_emb = tf.nn.dropout(input_emb, 1.0 - dropout_ratio)
      input_emb = tf.layers.dense(input_emb, h)
      if i < len(hidden_sizes) - 1:
        input_emb = tf.nn.relu(input_emb)
  return input_emb


def character_cnn(char_ids, num_chars=char_utils.NUM_CHARS, emb_size=32,
                  kernel_width=5, num_filters=100):
  """A character-level convolutional neural network with max-pooling.

  Args:
    char_ids: tensor<int32> [batch_size, ..., max_word_length]
    num_chars: The maximum number of character ids.
    emb_size: An integer indicating the size of each character embedding.
    kernel_width: An integer indicating the size of the kernel for the
        convolution filters.
    num_filters: An integer indicating the number of filters to use.

  Returns:
    pooled_emb: A tf.float32 Tensor of shape
        [batch_size, ..., num_filters] representing the filters
        after max-pooling over the positions in each word.
  """
  char_ids, flatten = tensor_utils.flatten(char_ids)
  embeddings = tf.get_variable(
      "char_emb", [num_chars, emb_size],
      initializer=tf.truncated_normal_initializer(stddev=0.1))
  char_emb = tf.nn.embedding_lookup(embeddings, char_ids)
  conv_emb = tf.layers.conv1d(char_emb, num_filters, kernel_width)
  pooled_emb = tf.reduce_max(conv_emb, -2)
  pooled_emb = flatten(pooled_emb)
  return pooled_emb


def stacked_highway(input_emb, hidden_sizes, dropout_ratio, mode,
                    layer_norm=True):
  """Construct multiple `highway` layers stacked on top of one another.

  Args:
    input_emb: tensor<float> [..., embedding_size]
    hidden_sizes: list<int> [hidden_size_1, hidden_size_2, ...]
    dropout_ratio: The probability of dropping out each unit in the activation.
        This can be None, and is only applied during training.
    mode: One of the keys from tf.estimator.ModeKeys.
    layer_norm: Boolean indicating whether we should apply layer normalization.

  Returns:
    output_emb: A Tensor with the same shape as `input_emb`, except for the last
        dimension which will have size `hidden_sizes[-1]` instead.
  """
  for i, h in enumerate(hidden_sizes):
    with tf.variable_scope("highway_{}".format(i)):
      input_emb = highway(input_emb, h, dropout_ratio, mode, layer_norm)
  return input_emb


def highway(input_emb, output_size, dropout_ratio=None, mode=None,
            layer_norm=True, carry_bias=1.0):
  """A single highway layer as described here: https://arxiv.org/abs/1505.00387.

  This is used as a generalization of a single hidden layer in a standard
  feedforward network. The difference is that the hidden layer with the
  non-linear activation is interpolated with a linear projection, enabling a
  "highway" for gradients.

  Args:
    input_emb: tensor<float> [..., embedding_size]
    output_size: An integer indicating the size of the hidden layer.
    dropout_ratio: The probability of dropping out each unit in the activation.
        This can be None, and is only applied during training.
    mode: One of the keys from tf.estimator.ModeKeys.
    layer_norm: Boolean indicating whether we should apply layer normalization.
    carry_bias: Float indicating the initial bias for the carry gate. This is
        typically set to a positive number to encourage information flow through
        the "highway" initially. This is analogous to the forget bias typically
        found in LSTMs.

  Returns:
    output_emb: A Tensor with the same shape as `input_emb`, except for the last
        dimension which will have size `output_size instead.
  """
  if mode == tf.estimator.ModeKeys.TRAIN and dropout_ratio is not None:
    input_emb = tf.nn.dropout(input_emb, 1.0 - dropout_ratio)

  with tf.variable_scope("joint_linear_layer"):
    # We need to compute three projections from the input embeddings for a
    # highway layer: (1) the hidden layer with a non-linear activation, (2) the
    # transform gate with a sigmoid activation, and (3) the projected input with
    # which the hidden layer is interpolated. These projections are combined for
    # better parallelism.
    joint_linear = tf.layers.dense(input_emb, output_size * 3)
  hidden_emb, carry_gate, project_input_emb = tf.split(joint_linear, 3, -1)
  hidden_emb = tf.nn.relu(hidden_emb)
  carry_gate = tf.sigmoid(carry_gate + carry_bias)
  if layer_norm:
    hidden_emb = tf.contrib.layers.layer_norm(hidden_emb)

  output_emb = carry_gate * project_input_emb + (1 - carry_gate) * hidden_emb
  return output_emb

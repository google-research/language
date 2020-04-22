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
"""Define the paragraph reconstruction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert import modeling

import tensorflow.compat.v1 as tf


def create_model(model, labels, label_types, num_choices, k_size=4):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    labels: ground truth paragraph order
    label_types: which k distances are being predicted
    num_choices: number of negatives samples + 1
    k_size: window size of CPC k distance

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cpc_loss"):
    output = tf.reshape(output_layer, (-1, num_choices + 1, hidden_size))
    contexts = output[:, 0, :]
    targets = output[:, 1:, :]

    softmax_weights = tf.get_variable(
        "cpc_weights", [k_size * 2, hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    context_encoded = tf.matmul(softmax_weights, contexts, transpose_b=True)
    context_encoded = tf.transpose(context_encoded, perm=[2, 0, 1])

    logits = tf.matmul(targets, context_encoded, transpose_b=True)
    logits = tf.transpose(logits, perm=[0, 2, 1])

    example_weights = tf.reduce_sum(tf.one_hot(label_types, k_size * 2), axis=1)

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    probabilities = tf.nn.softmax(logits, axis=-1)
    loss = tf.reduce_mean(
        tf.reduce_sum(example_weights * per_example_loss, axis=-1))

  return (loss, per_example_loss, logits, probabilities)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)

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


def create_model(
    model,
    labels,
    label_types,
    batch_size,
    num_choices,
    use_tpu,
    lv2loss=False,
    use_margin_loss=True,
    margin=1.):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    labels: ground truth paragraph order
    label_types: which k distances are being predicted
    batch_size: the batch size
    num_choices: number of negatives samples + 1
    use_tpu: if use tpu
    lv2loss: (bool) add a second level loss
    use_margin_loss: (bool) use margin loss instead of CE
    margin: (float) eta used in max margin loss

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cpc_loss"):

    softmax_weights = tf.get_variable(
        "softmax_weights", [hidden_size, 8],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    with tf.variable_scope("loss"):
      # if is_training:
      # I.e., 0.1 dropout
      #  output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      matmul_out = tf.matmul(output_layer, softmax_weights)

      logits = tf.reshape(matmul_out, (-1, num_choices, 8))
      logits = tf.transpose(logits, perm=[0, 2, 1])

      example_weights = tf.reduce_sum(tf.one_hot(label_types, 8), axis=1)

      if use_margin_loss:
        one_hot_labels = tf.one_hot(labels, num_choices)
        pos_logits = tf.reduce_sum(one_hot_labels * logits, axis=2)
        one_cold_labels = tf.ones_like(logits) - one_hot_labels
        downweighting = one_hot_labels * -9999
        neg_logits = tf.reduce_max(
            (one_cold_labels * logits) + downweighting, axis=2)
        per_example_loss = tf.maximum(0.,
                                      float(margin) - pos_logits + neg_logits)
      else:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
      probabilities = tf.nn.softmax(logits, axis=-1)
      loss_weights = tf.constant([0.1, 0.25, 0.5, 1, 1, 0.5, 0.25, 0.1])
      # loss_weights = tf.constant([0, 0, 0, 1., 1., 0, 0, 0])
      if use_tpu:
        loss_weights = tf.broadcast_to(loss_weights,
                                       [example_weights.shape[0], 8])
      else:
        loss_weights = tf.broadcast_to(loss_weights, [batch_size, 8])
      loss = tf.reduce_mean(
          tf.reduce_sum(
              loss_weights * example_weights * per_example_loss, axis=-1))

      if lv2loss:
        seq_output = tf.reshape(output_layer, [-1, num_choices, hidden_size])
        attn = modeling.attention_layer(
            seq_output, seq_output, size_per_head=hidden_size)

        attn = tf.reshape(attn, [-1, hidden_size])
        attn = tf.concat([output_layer, attn], axis=-1)

        attn_softmax_weights = tf.get_variable(
            "attn_softmax_weights", [hidden_size * 2, 8],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        attn_matmul_out = tf.matmul(attn, attn_softmax_weights)

        attn_logits = tf.reshape(attn_matmul_out, (-1, num_choices, 8))
        attn_logits = tf.transpose(attn_logits, perm=[0, 2, 1])

        attn_per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=attn_logits, labels=labels)
        attn_probabilities = tf.nn.softmax(logits, axis=-1)
        attn_loss = tf.reduce_mean(
            example_weights * per_example_loss) / tf.to_float(batch_size)
        loss += attn_loss
        return (loss, attn_per_example_loss, attn_logits, attn_probabilities)

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

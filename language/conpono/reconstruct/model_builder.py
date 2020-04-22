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
from tensorflow.contrib import seq2seq as contrib_seq2seq


class FixedSizeInferenceHelper(contrib_seq2seq.InferenceHelper):
  """Feeds in the output of the decoder at each step for fixed size."""

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for TrainingHelper."""
    return (finished, sample_ids, state)


def create_model(model,
                 labels,
                 decoder_inputs,
                 batch_size,
                 model_type="decode",
                 sep_positions=None):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    labels: ground truth paragraph order
    decoder_inputs: the input to the decoder if used
    batch_size: the batch size
    model_type: one of decode, pooled, attn
    sep_positions: (optional) for "pooled" indecies of SEP tokens

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value
  tpu_batch_size = tf.shape(output_layer)[0]

  num_labels = 5  # GOOGLE-INTERNAL TODO(daniter) this shouldn't be hardcoded

  with tf.variable_scope("paragraph_reconstruct"):
    if model_type == "decode":
      lstm_cell = tf.nn.rnn_cell.LSTMCell(
          num_units=hidden_size, use_peepholes=True, state_is_tuple=True)

      def sample_fn(x):
        return tf.to_float(tf.reshape(tf.argmax(x, axis=-1), (-1, 1)))

      helper = FixedSizeInferenceHelper(
          sample_fn=sample_fn,
          sample_shape=[1],
          sample_dtype=tf.float32,
          start_inputs=decoder_inputs[:, 0],
          end_fn=None)

      # Decoder
      project_layer = tf.layers.Dense(
          num_labels, use_bias=False, name="output_projection")

      my_decoder = contrib_seq2seq.BasicDecoder(
          lstm_cell,
          helper,
          tf.nn.rnn_cell.LSTMStateTuple(output_layer, output_layer),
          output_layer=project_layer)

      # Dynamic decoding
      outputs, _, _ = contrib_seq2seq.dynamic_decode(
          my_decoder,
          swap_memory=True,
          scope="paragraph_reconstruct",
          maximum_iterations=5)

      logits = outputs.rnn_output

      cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

      per_example_loss = cross_ent
      loss = tf.reduce_sum(cross_ent) / tf.to_float(batch_size)
      probabilities = tf.nn.softmax(logits, axis=-1)

    # GOOGLE-INTERAL: TODO(daniter)  currently neither of these actually train
    elif model_type == "pooled":
      token_embeddings = model.get_sequence_output()
      # sep positions come out batch by batch so we need to add the batch index
      # we do that explicitly here since we don't know the batch size in the
      # record decoder
      batch_idx = tf.range(tpu_batch_size)
      batch_idx = tf.reshape(batch_idx, [tpu_batch_size, 1])
      batch_idx = tf.tile(batch_idx, [1, 5])  # double check
      batch_idx = tf.reshape(batch_idx, [tpu_batch_size, 5, 1])
      # batch_idx = tf.Print(batch_idx, [batch_idx],
      #                      message="batch_idx", summarize=999999)
      sep_positions = tf.concat([batch_idx, sep_positions], axis=2)
      # sep_positions = tf.Print(sep_positions, [sep_positions],
      #                          message="sep_positions", summarize=999999)

      sep_vecs = tf.gather_nd(token_embeddings, sep_positions)
      sep_vecs = tf.reshape(sep_vecs, [tpu_batch_size, 5, hidden_size])
      # sep_vecs = tf.Print(sep_vecs, [sep_vecs], message="sep_vecs",
      #                     summarize=999999)

      logits = tf.layers.dense(
          inputs=sep_vecs, units=num_labels, name="output_projection")
      # logits = tf.Print(logits, [logits], message="logits", summarize=999999)
      cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

      per_example_loss = cross_ent
      loss = tf.reduce_sum(cross_ent) / tf.to_float(batch_size)
      probabilities = tf.nn.softmax(logits, axis=-1)

    elif model_type == "attn":
      # change size to match sequence embedding size
      input_consts = tf.constant([0, 1, 2, 3, 4])
      position_encoding = tf.broadcast_to(input_consts, [tpu_batch_size, 5])
      # position_encoding = tf.to_float(
      # tf.reshape(position_encoding, (-1, 5, 1)))
      token_type_table = tf.get_variable(
          name="attention_embedding",
          shape=[5, 512],  # don't hardcode
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      # This vocab will be small so we always do one-hot here, since it is
      # always faster for a small vocabulary.
      flat_token_type_ids = tf.reshape(position_encoding, [-1])
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=5)
      token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [tpu_batch_size, 5, 512])

      token_embeddings = model.get_sequence_output()
      attn = modeling.attention_layer(token_type_embeddings, token_embeddings)
      attn = tf.reshape(attn, (-1, 5, 512))  # head size
      logits = tf.layers.dense(
          inputs=attn, units=num_labels, name="output_projection")
      cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

      per_example_loss = cross_ent
      loss = tf.reduce_sum(cross_ent) / tf.to_float(batch_size)
      probabilities = tf.nn.softmax(logits, axis=-1)

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

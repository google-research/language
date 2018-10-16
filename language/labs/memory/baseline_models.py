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
"""Implements baselines for synthetic task memory experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.memory.model_utils import layer_norm

import tensorflow as tf


def vanilla_rnn(seqs, hidden_size=50):
  """A basic RNN.

  Args:
    seqs: <tf.float32>[batch_size, total_len, pattern_size] Sequences of
      patterns, where `total_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    hidden_size (int): Number of hidden units.

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.
  """
  pattern_size = seqs.shape[2]

  cell = tf.contrib.rnn.BasicRNNCell(hidden_size)

  # outputs: <tf.float32>[batch_size, total_len, hidden_size]
  # state (unused): <tf.float32>[batch_size, hidden_size]
  #
  # Note that output_size=cell_state_size=hidden_size in BasicRNNCell.
  outputs, _ = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=seqs)

  # Return only the last output
  # preds: <tf.float32>[batch_size, pattern_size]
  # outputs[:, -1]: <tf.float32>[batch_size, hidden_size] is the last hidden
  #   representation for each element in the batch.
  # W: <tf.float32>[hidden_size, pattern_size]
  preds = tf.layers.dense(outputs[:, -1], pattern_size)

  return preds


def vanilla_lstm(seqs, hidden_size=50):
  """A basic LSTM.

  Args:
    seqs: <tf.float32>[batch_size, total_len, pattern_size] Sequences of
      patterns, where `total_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    hidden_size (int): Number of hidden units.

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.
  """
  pattern_size = seqs.shape[2]

  cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

  # outputs: <tf.float32>[batch_size, total_len, hidden_size]
  # state (unused): <tf.float32>[
  #     batch_size, LSTMStateTuple(cell_state_size, hidden_size)]
  #
  # Note that output_size=cell_state_size=hidden_size.
  # Furthermore the returned state of BasicLSTMCell is a tuple containing the
  # c_state and m_state (both of size cell_state_size=hidden_size), i.e.
  # total cell size is 2xhidden_size.
  outputs, _ = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=seqs)

  # Return only the last output
  # preds: <tf.float32>[batch_size, pattern_size]
  # outputs[:, -1]: <tf.float32>[batch_size, hidden_size] is the last hidden
  #   representation for each element in the batch.
  # W: <tf.float32>[hidden_size, pattern_size]
  preds = tf.layers.dense(outputs[:, -1], pattern_size)

  return preds


def rnn_attention(seqs, batch_size=32, hidden_size=50):
  """A basic RNN with attention.

  Args:
    seqs: <tf.float32>[batch_size, total_len, pattern_size] Sequences of
      patterns, where `total_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    batch_size (int): Batch size.
    hidden_size (int): Number of hidden units.

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.
  """
  _, total_len, pattern_size = seqs.shape

  # Initialize parameters
  w1 = tf.get_variable("w1", shape=[pattern_size + hidden_size, hidden_size])
  b1 = tf.get_variable(
      "b1", shape=[hidden_size], initializer=tf.zeros_initializer())

  # state: <tf.float32>[batch_size, hidden_size]
  # Store all intermediate hidden states h_i in all_states, where we will later
  # use for computing the attention-based hidden representation c, where
  # c = \sum_i \alpha_i h_i
  state = tf.zeros([batch_size, hidden_size], name="state")
  all_states = []

  # Construct the unrolled RNN manually for the sequence
  for timestep in range(total_len):
    # concat_state: <tf.float32>[batch_size, pattern_size + hidden_size]
    # For each batch element, take the input for a given timestep and append the
    # hidden state vector. The weight matrix w1 and bias b1 encodes both the W
    # and U matrices in the h_t = g(Wx_i, Uh_{t-1}).
    concat_state = tf.concat([seqs[:, timestep, :], state], 1)
    state = tf.matmul(concat_state, w1) + b1
    state = tf.tanh(state)
    all_states.append(state)

  # Recall that the sequence contains all the input patterns plus a query
  # pattern and that the query patterns is a corrupted form of one of the input
  # patterns we want to recover. Remove last TIMESTEPS_PER_PATTERN=1 states for
  # the query pattern as that's irrelevant for computing the attention-weighted
  # hidden representation.
  #
  # all_states: <tf.float32>[total_len - 1, batch_size, hidden_size].
  all_states = tf.convert_to_tensor(all_states[:-1], dtype=tf.float32)

  # Transpose so batch_size is first dim
  # new shape <tf.float32>[batch_size, total_len - 1, hidden_size]
  all_states = tf.transpose(all_states, [1, 0, 2])

  # Compute similarity between final state and all previous states.
  # attention: <tf.float32>[batch_size, total_len - 1, 1]
  attention = tf.reduce_sum(
      tf.multiply(tf.expand_dims(state, 1), all_states), 2, keep_dims=True)

  # Turn into probability distribution
  # attention_dist: <tf.float32>[batch_size, total_len - 1, 1]
  attention_dist = tf.nn.softmax(attention, dim=1)
  # context: <tf.float32>[batch_size, hidden_size]
  context = tf.reduce_sum(tf.multiply(attention_dist, all_states), 1)

  # concat_state: <tf.float32>[batch_size, 2 * hidden_size]
  concat_state = tf.concat([context, state], 1)
  # preds: <tf.float32>[batch_size, pattern_size]
  preds = tf.layers.dense(concat_state, pattern_size)

  return preds


def rnn_fast_weights(seqs,
                     batch_size=32,
                     hidden_size=50,
                     fast_steps=1,
                     fast_decay_rate=.95,
                     fast_lr=.5):
  """Implements a RNN with fast weights.

  The RNN updates a fast-weights "memory" matrix A via Hebbian-style updates
  as described by [Ba et al. 2018] Using Fast Weights To Attend To the Recent
  Past (https://arxiv.org/abs/1610.062580).

  Args:
    seqs: <tf.float32>[batch_size, total_len, pattern_size] Sequences of
      patterns, where `total_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    batch_size (int): Batch size.
    hidden_size (int): Number of hidden units in the RNN.
    fast_steps (int): Number of inner loop iterations we apply fast weights.
    fast_decay_rate (float): Decay rate (lambda) for fast weights update.
    fast_lr (float): Learning rate (eta) for fast weights update.

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.
  """

  _, total_len, pattern_size = seqs.shape

  # Initialize parameters
  w1 = tf.get_variable("w1", shape=[pattern_size + hidden_size, hidden_size])
  b1 = tf.get_variable("b1", shape=[hidden_size],
                       initializer=tf.zeros_initializer())

  # state: <tf.float32>[batch_size, hidden_size]
  state = tf.zeros([batch_size, hidden_size], name="state")
  # a_memory: <tf.float32>[batch_size, hidden_size, hidden_size]
  a_memory = tf.zeros(
      [batch_size, hidden_size, hidden_size],
      dtype=tf.float32,
      name="A")

  # Unroll graph manually
  for timestep in range(total_len):
    # concat_state: <tf.float32>[batch_size, pattern_size + hidden_size]
    # For each batch element, take the input for a given timestep and append the
    # hidden state vector. The weight matrix w1 and bias b1 encodes both the W
    # and U matrices in the h_t = g(Wx_i, Uh_{t-1}).
    inp = tf.concat([seqs[:, timestep, :], state], 1)
    state = tf.matmul(inp, w1) + b1
    boundary_state = state  # "sustained boundary condition" pre-nonlinearity
    state = tf.tanh(state)

    # Apply fast weights
    for _ in range(fast_steps):
      # fw_state: <tf.float32>[batch_size, hidden_size]
      fw_state = tf.squeeze(tf.matmul(a_memory, tf.expand_dims(state, 2)))
      state = boundary_state + fw_state
      state = layer_norm(state, hidden_size)
      state = tf.tanh(state)

    # Update fast weights matrix
    a_memory = fast_decay_rate * a_memory + fast_lr * tf.matmul(
        tf.expand_dims(state, 2),  # <tf.float32>[batch_size, hidden_size, 1]
        tf.expand_dims(state, 1))  # <tf.float32>[batch_size, 1, hidden_size]

  # preds: <tf.float32>[batch_size, pattern_size]
  preds = tf.layers.dense(state, pattern_size)

  return preds

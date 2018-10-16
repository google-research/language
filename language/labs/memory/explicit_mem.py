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
"""Implements a fixed size memory baseline with a simple r/w rule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_explicit_mem(seqs,
                     batch_size=32,
                     hidden_size=50):
  """Implements a fixed size memory baseline with a simple r/w rule.

  Args:
    seqs: <tf.float32>[batch_size, seq_len, pattern_size] Sequences of
      patterns, where `seq_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    batch_size (int): Batch size.
    hidden_size (int): Number of hidden units

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.
  """

  _, seq_len, pattern_size = seqs.shape

  # Initialize parameters
  w1 = tf.get_variable("w1", shape=[pattern_size + hidden_size, hidden_size])
  b1 = tf.get_variable(
      "b1", shape=[hidden_size], initializer=tf.zeros_initializer())
  write_w = tf.get_variable("write_w", shape=[hidden_size, hidden_size])

  # state: <tf.float32>[batch_size, hidden_size]
  state = tf.zeros([batch_size, hidden_size], name="state")
  # a_memory: <tf.float32>[batch_size, hidden_size, hidden_size]
  a_memory = tf.zeros(
      [batch_size, hidden_size, hidden_size], dtype=tf.float32, name="A")

  # Unroll graph manually
  for timestep in range(seq_len):
    # inp: <tf.float32>[batch_size, pattern_size + hidden_size]
    inp = tf.concat([seqs[:, timestep, :], state], 1)

    # state: <tf.float32>[batch_size, hidden_size]
    state = tf.matmul(inp, w1) + b1
    boundary_state = state  # "sustained boundary condition," pre-nonlinearity
    state = tf.tanh(state)

    # Determine how much to read from each row of memory
    # read_coeffs: <tf.float32>[batch_size, hidden_size, 1]
    read_coeffs = tf.nn.softmax(tf.matmul(a_memory, tf.expand_dims(state, 2)))
    # Retrieve from memory
    # mem_values: <tf.float32>[batch_size, hidden_size]
    mem_values = tf.squeeze(tf.matmul(a_memory, read_coeffs))

    state = boundary_state + mem_values
    state = tf.tanh(state)

    # Determine how much to write to each row of memory
    # write_state: <tf.float32>[batch_size, hidden_size, 1]
    write_state = tf.expand_dims(tf.matmul(state, write_w), 2)
    # write_coeff: <tf.float32>[batch_size, hidden_size, 1]
    write_coeffs = tf.nn.softmax(tf.matmul(a_memory, write_state))
    # Keep (1 - write_coeffs) of old memory and update (write_coeffs)
    # write_state: <tf.float32>[batch_size, 1, hidden_size]
    write_state = tf.transpose(write_state, [0, 2, 1])
    a_memory = (tf.multiply(a_memory, 1 - write_coeffs) +
                tf.multiply(tf.tile(write_state, [1, hidden_size, 1]),
                            write_coeffs))

  # preds: <tf.float32>[batch_size, pattern_size]
  preds = tf.layers.dense(state, pattern_size)
  tf.summary.histogram("preds", preds)

  return preds

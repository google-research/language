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
"""Implements differentiable plasticity from Miconi et al 2018.

See [Miconi et al. 2018] Differentiable plasticity: training plastic neural
networks with backpropagation (https://arxiv.org/pdf/1804.02464.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.memory.model_utils import layer_norm

import tensorflow as tf


def rnn_differentiable_plasticity(seqs,
                                  batch_size=32,
                                  hidden_size=50,
                                  fast_steps=1,
                                  fast_lr_fixed=None,
                                  use_oja_rule=None,
                                  update_mem_with_prev_timestep=None,
                                  learn_fast_lr=None,
                                  learn_plasticity_coeffs=None):
  """Implements differentiable plasticity from Miconi et al. 2018.

  Differences with FW: (1) learnable plasticity_coeffs (2) learnable fast
  learning rate, eta (3a) Oja's rule to update memory matrix, instead of Hebb's
  rule OR (3b) use previous timestep to update memory matrix with Hebb's rule

  Args:
    seqs: <tf.float32>[batch_size, seq_len, pattern_size] Sequences of
      patterns, where `seq_len` is the length of a sequence (including the
      query pattern) and `pattern_size` is the dimensionality of each pattern.
    batch_size (int): Batch size.
    hidden_size (int): Number of hidden units
    fast_steps (int): Number of inner loop iterations we apply fast weights.
    fast_lr_fixed (float): Learning rate (eta) for fast weights update if fast
      lr is not learned.
    use_oja_rule (bool): True if we update the memory matrix with Oja's rule.
    update_mem_with_prev_timestep (bool): True if we update the memory matrix
      by a dot product with the previous hidden state. (only applies if we use
      Hebb's rule to update)
    learn_fast_lr (bool): True if the fast learning rate is learnable.
    learn_plasticity_coeffs (bool): True if the plasticity coefficients are
      learnable.

  Returns:
    preds: <tf.float32>[batch_size, pattern_size] The retrieved pattern for
      the degraded query at the end of the sequence.

  Raises:
    TypeError: If kwargs are not specified.
  """

  # Validate boolean args that would otherwise fail silently
  if (use_oja_rule is None or update_mem_with_prev_timestep is None or
      learn_fast_lr is None or learn_plasticity_coeffs is None):
    raise TypeError("Settings must be specified for differentiable plasticity.")

  _, seq_len, pattern_size = seqs.shape

  # Initialize parameters
  w1 = tf.get_variable("w1", shape=[pattern_size + hidden_size, hidden_size])
  b1 = tf.get_variable(
      "b1", shape=[hidden_size], initializer=tf.zeros_initializer())

  # state: <float32>[batch_size, hidden_size]
  state = tf.zeros([batch_size, hidden_size], name="state")
  # a_memory: <float32>[batch_size, hidden_size, hidden_size]
  a_memory = tf.zeros(
      [batch_size, hidden_size, hidden_size], dtype=tf.float32, name="A")

  if learn_plasticity_coeffs:
    # plasticity_coeffs: <float32>[hidden_size, hidden_size]
    plasticity_coeffs = tf.get_variable(
        "plasticity_coeffs", shape=[hidden_size, hidden_size])

  if learn_fast_lr:
    # fast_lr_learned: <float32>[]
    fast_lr_learned = tf.get_variable("fast_lr_learned", shape=[])
    fast_lr = tf.nn.sigmoid(fast_lr_learned)
  else:
    fast_lr = fast_lr_fixed
  tf.summary.scalar("fast_lr", fast_lr)

  # Unroll graph manually
  for timestep in range(seq_len):
    # inp: <float32>[batch_size, pattern_size + hidden_size]
    inp = tf.concat([seqs[:, timestep, :], state], 1)
    last_state = state

    # state: <float32>[batch_size, hidden_size]
    state = tf.matmul(inp, w1) + b1
    boundary_state = state  # "sustained boundary condition," pre-nonlinearity
    state = tf.tanh(state)

    # Apply fast weights
    for _ in range(fast_steps):
      # fw_state: <float32>[batch_size, hidden_size]
      fw_state = tf.squeeze(tf.matmul(a_memory, tf.expand_dims(state, 2)))

      # Apply plasticity coefficient
      if learn_plasticity_coeffs:
        # fw_state: <float32>[batch_size, hidden_size, 1]
        fw_state = tf.expand_dims(fw_state, 2)
        # pc_tiled: <float32>[batch_size, hidden_size, hidden_size]
        pc_tiled = (tf.ones([batch_size, hidden_size, hidden_size])
                    * plasticity_coeffs)
        # fw_state: <float32>[batch_size, hidden_size]
        fw_state = tf.squeeze(tf.matmul(pc_tiled, fw_state))

      state = boundary_state + fw_state
      state = layer_norm(state, hidden_size)
      state = tf.tanh(state)

    # Update fast weights matrix
    if use_oja_rule:
      a_memory = a_memory + fast_lr * (
          tf.multiply(
              tf.expand_dims(state, 1),
              tf.expand_dims(last_state, 2) - tf.multiply(
                  tf.expand_dims(state, 1), a_memory)))
    elif update_mem_with_prev_timestep:
      a_memory = (
          fast_lr * tf.matmul(tf.expand_dims(last_state, 2),
                              tf.expand_dims(state, 1))
          + (1 - fast_lr) * a_memory)
    else:
      # Fast weights update, except only fast_lr is parameterized
      a_memory = (1 - fast_lr) * a_memory + fast_lr * tf.matmul(
          tf.expand_dims(state, 2),  # <float32>[batch_size, hidden_size, 1]
          tf.expand_dims(state, 1))  # <float32>[batch_size, 1, hidden_size]

  # preds: <float32>[batch_size, pattern_size]
  preds = tf.layers.dense(state, pattern_size)
  tf.summary.histogram("preds", preds)

  return preds

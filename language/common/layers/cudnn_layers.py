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
"""Layers using cuDNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops


def _single_lstm(input_emb, input_len, hidden_size, is_fwd, use_cudnn):
  """Compute the outputs of a single LSTM (subroutine of stacked_bilstm).

  Be careful if used anywhere outside of stacked_bilstm, which converts the
  sequences to the time-major format expected by this function.

  Args:
    input_emb: <float32> [sequence_length, batch_size, emb]
    input_len: <int32> [batch_size]
    hidden_size: Number of units in the LSTM cell.
    is_fwd: Boolean indicator the directionality of the LSTM.
    use_cudnn: Boolean indicating the use of cudnn.

  Returns:
    output_emb: <float32> [sequence_length, batch_size, emb]
  """
  if not is_fwd:
    input_emb = tf.reverse_sequence(
        input_emb,
        input_len,
        seq_axis=0,
        batch_axis=1)
  if use_cudnn:
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=1,
        num_units=hidden_size,
        input_mode=cudnn_rnn_ops.CUDNN_INPUT_LINEAR_MODE,
        direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION)
    lstm.build(input_emb.shape)
    output_emb, _ = lstm(input_emb)
  else:
    cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
    cell = tf.contrib.rnn.MultiRNNCell([cell])
    output_emb, _ = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=input_emb,
        sequence_length=input_len,
        dtype=tf.float32,
        time_major=True)
  if not is_fwd:
    output_emb = tf.reverse_sequence(
        output_emb,
        input_len,
        seq_axis=0,
        batch_axis=1)
  return output_emb


def stacked_bilstm(input_emb, input_len, hidden_size, num_layers, dropout_ratio,
                   mode, use_cudnn=None):
  """Encode inputs via stacked bidirectional LSTMs with residual connections.

  Args:
    input_emb: <float32> [batch_size, sequence_length, emb]
    input_len: <int32> [batch_size]
    hidden_size: Size of each LSTM layer.
    num_layers: Number of LSTM layers.
    dropout_ratio: Probability of dropout out dimensions of each hidden layer.
    mode: One of the keys from tf.estimator.ModeKeys.
    use_cudnn: Specify the use of cudnn. `None` denotes automatic selection.

  Returns:
    output_emb: <float32> [batch_size, sequence_length, emb]
  """
  # cuDNN expects time-major inputs, so we transpose before and after.
  input_emb = tf.transpose(input_emb, [1, 0, 2])
  if use_cudnn is None:
    use_cudnn = tf.test.is_gpu_available(cuda_only=True)

  for i in range(num_layers):
    with tf.variable_scope("lstm_{}".format(i)):
      if mode == tf.estimator.ModeKeys.TRAIN:
        input_emb = tf.nn.dropout(input_emb, 1.0 - dropout_ratio)

      output_emb = []
      for is_fwd in (True, False):
        with tf.variable_scope("fw" if is_fwd else "bw"):
          output_emb.append(_single_lstm(
              input_emb=input_emb,
              input_len=input_len,
              hidden_size=hidden_size,
              is_fwd=is_fwd,
              use_cudnn=use_cudnn))
      output_emb = tf.concat(output_emb, -1)
      if i == 0:
        input_emb = output_emb
      else:
        # Add residual connection after the first layer.
        input_emb += output_emb

  # cuDNN expects time-major inputs, so we transpose before and after.
  output_emb = tf.transpose(input_emb, [1, 0, 2])
  return output_emb

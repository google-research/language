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
"""This is based on code in tf.contrib.seq2seq."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from language.labs.exemplar_decoding.models.common import sparse_batched_matmul
from language.labs.exemplar_decoding.models.linear import Linear
from language.labs.exemplar_decoding.utils import tensor_utils
import tensorflow as tf
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class OutputWrapper(tf.nn.rnn_cell.RNNCell):
  """Based on tensorflow.contrib.rnn.OutputProjectionWrapper."""

  def __init__(self, cell, num_layers, hidden_dim, output_size,
               weights=None,
               activation=tf.tanh,
               dropout=0.,
               use_copy=False,
               encoder_emb=None,
               sparse_inputs=None,
               mask=None,
               hps=None,
               mode=tf.estimator.ModeKeys.EVAL,
               reuse=None):
    """Create a cell with output projection.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      num_layers: number of MLP layers.
      hidden_dim: hidden size of the MLP.
      output_size: integer, the size of the output after projection.
      weights: (optional) a specified tensor.
      activation: (optional) an optional activation function.
      dropout: dropout rate for dropout at the output layer.
      use_copy: Use copy mechanism or not.
      encoder_emb: Outputs of the encoder.
      sparse_inputs: Sparse inputs.
      mask: mask.
      hps: Hyperparameters.
      mode: train/eval.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if output_size is not positive.
    """
    super(OutputWrapper, self).__init__(_reuse=reuse)
    if output_size < 1:
      raise ValueError("Parameter output_size must be > 0: %d." % output_size)
    self._cell = cell
    self._num_layers = num_layers
    self._activation = activation
    self._weights = weights
    if self._weights is None:
      self._output_size = output_size
    else:
      self._output_size = tensor_utils.shape(self._weights, 1)

    self._hidden_dim = hidden_dim
    self._dropout = dropout
    self._reuse = reuse
    self._mode = None
    self._sigmoid = tf.sigmoid
    self._linear1, self._linear2, self._linear_copy = None, None, None
    assert self._num_layers <= 2
    self._use_copy = use_copy
    self._encoder_emb = encoder_emb
    self._sparse_inputs, self._mask = sparse_inputs, mask
    self._mode = mode
    self._reuse_attention = hps.reuse_attention
    if self._use_copy:
      assert self._sparse_inputs is not None
      assert self._mask is not None
      if not self._reuse_attention:
        assert self._encoder_emb is not None
        encoder_dim = tf.shape(self._encoder_emb)[-1]
        if encoder_dim != self._hidden_dim:
          assert False
    self._eps = 1e-8
    self._vocab_offset = hps.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def _compute_copy_prob(self, rnn_out, prob_gen, alignment):

    with tf.variable_scope("linear_copy", reuse=self._reuse):
      if self._linear_copy is None:
        self._linear_copy = Linear(
            rnn_out, 1, True,
            weights=None,
            weight_initializer=tf.contrib.layers.xavier_initializer())

    alignment = alignment * self._mask
    alignment /= tf.maximum(tf.reduce_sum(alignment, axis=1, keepdims=True),
                            self._eps)
    p_copy = self._sigmoid(self._linear_copy(rnn_out))

    # [batch_size, vocab_size]
    prob_copy = tf.squeeze(
        sparse_batched_matmul(
            self._sparse_inputs, tf.expand_dims(alignment, 2)),
        axis=2)
    prob = p_copy * prob_copy + (1. - p_copy) * prob_gen
    return prob

  def _compute_logits(self, rnn_out):
    if self._num_layers == 1 and self._weights is not None:
      assert tensor_utils.shape(rnn_out, -1) == self._hidden_dim

    if self._num_layers == 1:
      with tf.variable_scope("mlp1", reuse=self._reuse):
        if self._weights is None:
          scale = (3.0 / self._hidden_dim) ** 0.5
          weight_initializer = tf.random_uniform_initializer(
              minval=-scale, maxval=scale)
          self._linear1 = Linear(
              rnn_out,
              self._output_size,
              True, weights=None,
              weight_initializer=weight_initializer)
        else:
          self._linear1 = Linear(
              rnn_out, self._output_size, True, weights=self._weights)
        logits = self._linear1(rnn_out)
    else:
      assert False
      assert self._num_layers == 2
      with tf.variable_scope("mlp1", reuse=self._reuse):
        if self._linear1 is None:
          self._linear1 = Linear(
              rnn_out, self._hidden_dim, True,
              weights=None,
              weight_initializer=tf.contrib.layers.xavier_initializer())
        hidden = self._linear1(rnn_out)
        if self._activation:
          hidden = self._activation(hidden)

        if self._mode == tf.estimator.ModeKeys.TRAIN and self._dropout > 0.:
          hidden = tf.nn.dropout(hidden, keep_prob=1.-self._dropout)

      with tf.variable_scope("mlp2", reuse=self._reuse):
        if self._linear2 is None:
          if self._weights is None:
            scale = (3.0 / self._hidden_dim) ** 0.5
            weight_initializer = tf.random_uniform_initializer(
                minval=-scale, maxval=scale)
            self._linear2 = Linear(
                hidden,
                self._output_size,
                True, weights=None,
                weight_initializer=weight_initializer)
          else:
            self._linear2 = Linear(
                hidden, self._output_size, True, weights=self._weights)

        logits = self._linear2(hidden)
    return logits

  def call(self, inputs, state):
    """Run the cell and output projection on inputs, starting from state."""

    rnn_out, res_state = self._cell(inputs, state)

    # before dropout
    if self._use_copy:
      if self._reuse_attention:
        alignments = state.alignments
      else:
        scores = tf.squeeze(
            tf.matmul(self._encoder_emb, tf.expand_dims(rnn_out, axis=2)),
            axis=2)
        alignments = tf.nn.softmax(scores)

    if self._mode == tf.estimator.ModeKeys.TRAIN and self._dropout > 0.:
      rnn_out = tf.nn.dropout(rnn_out, keep_prob=1.-self._dropout)
    gen_logits = self._compute_logits(rnn_out)
    if self._use_copy:
      gen_prob = tf.nn.softmax(gen_logits, axis=-1)
      prob = self._compute_copy_prob(rnn_out, gen_prob, alignments)
      logits = tf.log(prob + self._eps)
    else:
      logits = gen_logits
    indices = tf.range(tf.shape(logits)[1])
    indices_mask = tf.zeros_like(indices, dtype=tf.float32) + self._vocab_offset
    negative_mask = tf.zeros_like(indices, dtype=tf.float32) - tf.constant(1e+8)
    zero_mask = tf.zeros_like(indices, dtype=tf.float32)
    bias = tf.where(tf.greater_equal(tf.cast(indices, dtype=tf.float32),
                                     indices_mask),
                    negative_mask, zero_mask)
    logits = logits + bias
    return logits, res_state

  def initialize(self):
    """Used to construct LSTM parameters."""
    self._cell.initialize()

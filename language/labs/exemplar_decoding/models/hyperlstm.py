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
"""Infrastructure for hypernet LSTM.

This is based on code in
tf.contrib.seq2seq.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
# pylint: disable=g-long-ternary


class HyperLSTMCell(tf.contrib.rnn.LayerRNNCell):
  """Based on the LSTM cell implementation from tensorflow.contrib.rnn.
  """

  def __init__(self, num_units, mem_input,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, dtype=None,
               use_beam=False,
               hps=None):
    """Initialize the HyperLSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      mem_input: mem_input.
      use_peepholes: bool, use peephole connections or not.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      use_beam: Use beam search or not.
      hps: hyperparameters.
    """

    super(HyperLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
    if not state_is_tuple:
      tf.logging.warn("%s: Using a concatenated state is slower and will soon "
                      "be deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      tf.logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    assert not use_peepholes, "currently not supporting peephole connections"
    assert hps is not None
    # Inputs must be 2-dimensional.
    self.input_spec = tf.layers.InputSpec(ndim=2)

    self._num_units = num_units
    self._rank = hps.rank
    assert self._rank == self._num_units or self._rank == 2 * self._num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or tf.tanh
    self._sigma_norm = hps.sigma_norm
    self._beam_width = hps.beam_width
    self._mem_input = mem_input
    self._use_beam = use_beam

    if num_proj:
      self._state_size = (
          tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

    input_depth = hps.emb_dim + hps.decoder_dim
    # if hps.encode_neighbor:
    #   input_depth += hps.decoder_dim
    h_depth = self._num_units if self._num_proj is None else self._num_proj

    maybe_partitioner = (
        tf.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None else None)

    # `u`s are matrices of [input_shape, rank], `v`s being [rank, hidden_size]
    # they are the collection of rank-1 parameter matrices.
    # The full parameter matrix is constructed by taking `U\sigma V`,
    # with diagonal matrix `\sigma` computed in the `self.initialize` function.

    redundant_rank = (self._rank > self._num_units)
    # `u`, `v` used to construct matrix from input `x` to input_gate `i`.
    u_xi, v_xi = self._orthogonal_init(
        shape=[input_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_xi = tf.get_variable(
        "u_xi/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_xi,
        partitioner=maybe_partitioner)
    self._v_xi = tf.get_variable(
        "v_xi/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_xi,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix that maps input `x` to cell_state `j`.
    u_xj, v_xj = self._orthogonal_init(
        shape=[input_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_xj = tf.get_variable(
        "u_xj/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_xj,
        partitioner=maybe_partitioner)
    self._v_xj = tf.get_variable(
        "v_xj/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_xj,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps input `x` to forget_gate `f`.
    u_xf, v_xf = self._orthogonal_init(
        shape=[input_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_xf = tf.get_variable(
        "u_xf/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_xf,
        partitioner=maybe_partitioner)
    self._v_xf = tf.get_variable(
        "v_xf/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_xf,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps input `x` to output_gate `o`.
    u_xo, v_xo = self._orthogonal_init(
        shape=[input_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_xo = tf.get_variable(
        "u_xo/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_xo,
        partitioner=maybe_partitioner)
    self._v_xo = tf.get_variable(
        "v_xo/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_xo,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps hid_state `h` to input_gate `i`.
    u_hi, v_hi = self._orthogonal_init(
        shape=[h_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_hi = tf.get_variable(
        "u_hi/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_hi,
        partitioner=maybe_partitioner)
    self._v_hi = tf.get_variable(
        "v_hi/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_hi,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps hid_state `h` to cell_state `j`.
    u_hj, v_hj = self._orthogonal_init(
        shape=[h_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_hj = tf.get_variable(
        "u_hj/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_hj,
        partitioner=maybe_partitioner)
    self._v_hj = tf.get_variable(
        "v_hj/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_hj,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps hid_state `h` to forget_gate `f`.
    u_hf, v_hf = self._orthogonal_init(
        shape=[h_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_hf = tf.get_variable(
        "u_hf/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_hf,
        partitioner=maybe_partitioner)
    self._v_hf = tf.get_variable(
        "v_hf/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_hf,
        partitioner=maybe_partitioner)

    # `u`, `v` used to construct matrix
    # that maps hid_state `h` to output_gate `o`.
    u_ho, v_ho = self._orthogonal_init(
        shape=[h_depth, self._num_units],
        initializer=initializer,
        redundant_rank=redundant_rank)
    self._u_ho = tf.get_variable(
        "u_ho/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=u_ho,
        partitioner=maybe_partitioner)
    self._v_ho = tf.get_variable(
        "v_ho/%s" % _WEIGHTS_VARIABLE_NAME,
        initializer=v_ho,
        partitioner=maybe_partitioner)

    self._c = tf.get_variable(
        "c/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._rank],
        initializer=tf.contrib.layers.xavier_initializer(),
        partitioner=maybe_partitioner)

    initializer = tf.zeros_initializer(dtype=tf.float32)
    self._b = tf.get_variable(
        "b/%s" % _BIAS_VARIABLE_NAME,
        shape=[4 * h_depth, self._rank],
        initializer=initializer)

    if self._num_proj is not None:
      if self._num_proj_shards is not None:
        maybe_proj_partitioner = (
            tf.fixed_size_partitioner(self._num_proj_shards))
      else:
        maybe_proj_partitioner = (None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=tf.uniform_unit_scaling_initializer(),
          partitioner=maybe_proj_partitioner)
    self.initialize()
    self.built = True

  def _orthogonal_init(self, shape, initializer,
                       dtype=tf.float32,
                       redundant_rank=False):
    if redundant_rank:
      matrix1 = initializer(shape=shape, dtype=dtype)
      _, u1, v1 = tf.svd(matrix1, full_matrices=False, compute_uv=True)
      matrix2 = initializer(shape=shape, dtype=dtype)
      _, u2, v2 = tf.svd(matrix2, full_matrices=False, compute_uv=True)

      u = tf.concat([u1, u2], axis=1)
      v = tf.concat([v1, v2], axis=0)
    else:
      matrix = initializer(shape=shape, dtype=dtype)
      _, u, v = tf.svd(matrix, full_matrices=False, compute_uv=True)
    return u, v

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def build(self, inputs_shape):
    # pylint: disable=g-doc-args
    """Parameter variables are already constructed in __init__.

    This function should never be called and aims only to keep consistency.
    """
    # pylint: enable=g-doc-args
    assert False, "the variables should have been constructed in __init__."

  def initialize(self):
    """Construct RNN params.
    """

    sigma = tf.matmul(self._mem_input, self._c)

    if self._sigma_norm > 0.:
      sigma = tf.nn.l2_normalize(sigma, axis=1) * self._sigma_norm
    elif self._sigma_norm == -1.:
      sigma = tf.nn.softmax(sigma, axis=1)
    sigma_diag = tf.matrix_diag(sigma)

    # The weight matrices.
    # {`x`: input, `h`: hidden_state}.
    # {`i`: input_gate, `j`: cell_state, `f`: forget_gate, `o`: output_gate}.

    # Weight matrix that maps input `x` to input_gate `i`.
    w_xi = tf.einsum("ij,ajk,kl->ail", self._u_xi, sigma_diag, self._v_xi)

    # Weight matrix that maps input `x` to cell_state `j`.
    w_xj = tf.einsum("ij,ajk,kl->ail", self._u_xj, sigma_diag, self._v_xj)

    # Weight matrix that maps input `x` to forget_gate `f`.
    w_xf = tf.einsum("ij,ajk,kl->ail", self._u_xf, sigma_diag, self._v_xf)

    # Weight matrix that maps input `x` to output_gate `o`.
    w_xo = tf.einsum("ij,ajk,kl->ail", self._u_xo, sigma_diag, self._v_xo)

    # Weight matrix that maps hidden_state `h` to input_gate `i`.
    w_hi = tf.einsum("ij,ajk,kl->ail", self._u_hi, sigma_diag, self._v_hi)

    # Weight matrix that maps hidden_state `h` to cell_state `j`.
    w_hj = tf.einsum("ij,ajk,kl->ail", self._u_hj, sigma_diag, self._v_hj)

    # Weight matrix that maps hidden_state `h` to forget_gate `f`.
    w_hf = tf.einsum("ij,ajk,kl->ail", self._u_hf, sigma_diag, self._v_hf)

    # Weight matrix that maps hidden_state `h` to output_gate `o`.
    w_ho = tf.einsum("ij,ajk,kl->ail", self._u_ho, sigma_diag, self._v_ho)

    w_x = tf.concat([w_xi, w_xj, w_xf, w_xo], axis=2)
    w_h = tf.concat([w_hi, w_hj, w_hf, w_ho], axis=2)

    self._weight = tf.concat([w_x, w_h], axis=1)
    self._bias = tf.einsum("ij,aj->ai", self._b, sigma)

    if self._use_beam and self._beam_width > 1:
      self._weight = tf.contrib.seq2seq.tile_batch(
          self._weight, multiplier=self._beam_width)
      self._bias = tf.contrib.seq2seq.tile_batch(
          self._bias, multiplier=self._beam_width)

  def call(self, inputs, state):
    """Run one step of HyperLSTM.

    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = tf.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      assert False

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    lstm_matrix = self._bias + tf.einsum(
        "ai,aij->aj", tf.concat([inputs, m_prev], 1), self._weight)

    i, j, f, o = tf.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    # Diagonal connections
    c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
         self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type

    m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = tf.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = tf.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type
    if self._state_is_tuple:
      new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m))
    else:
      new_state = (tf.concat([c, m], 1))
    return m, new_state

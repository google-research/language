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
"""A linear layer for output projection.

This is based on code in tf.contrib.seq2seq.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from language.labs.exemplar_decoding.models.common import dimension_value
import tensorflow as tf
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

__all__ = [
    "Linear",
    "HyperDense",
]


class Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch, n, Tensors.
    output_size: int, second dimension of weight variable.
    weights: (optional) a specified tensor.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               weights=None,
               weight_initializer=None,
               bias_initializer=None):
    self._build_bias = build_bias

    if args is None or (tf.contrib.framework.nest.is_sequence(args) and
                        not args):
      raise ValueError("`args` must be specified")
    if not tf.contrib.framework.nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      if weights is None:
        self._weights = tf.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=weight_initializer)
      else:
        self._weights = weights
      if build_bias:
        with tf.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
          self._biases = tf.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = tf.matmul(args[0], self._weights)
    else:
      # Explicitly creating a one for a minor performance improvement.
      one = tf.constant(1, dtype=tf.int32)
      res = tf.matmul(tf.concat(args, one), self._weights)
    if self._build_bias:
      res = tf.nn.bias_add(res, self._biases)
    return res


class HyperDense(tf.keras.layers.Layer):
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then
  it is flattened prior to the initial dot product with `kernel`.


  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
      nD tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               mem_input,
               hps,
               use_beam=False,
               activation=None,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if "input_shape" not in kwargs and "input_dim" in kwargs:
      kwargs["input_shape"] = (kwargs.pop("input_dim"),)

    super(HyperDense, self).__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.units = int(units)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._mem_input = mem_input

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
    self._can_use_graph_functions = True
    self._decoder_dim = hps.decoder_dim
    self._rank = hps.rank
    self._tau = hps.tau
    self._sigma_norm = hps.sigma_norm
    self._beam_width = hps.beam_width
    self._use_beam = use_beam

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense` "
                       "should be defined. Found `None`.")
    last_dim = dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

    self._c = tf.get_variable(
        "c", [self._decoder_dim, self._rank],
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    sigma = tf.matmul(self._mem_input, self._c)
    if self._sigma_norm > 0.:
      sigma = tf.nn.l2_normalize(sigma, axis=1) * self._sigma_norm
    elif self._sigma_norm == -1.:
      sigma = tf.nn.softmax(sigma / self._tau, axis=1)
    sigma_diag = tf.matrix_diag(sigma)

    self._u = tf.get_variable(
        "u", [last_dim, self._rank],
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self._v = tf.get_variable(
        "v", [self._rank, self.units],
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.kernel = tf.einsum("ij,ajk,kl->ail", self._u, sigma_diag, self._v)
    if self._use_beam and self._beam_width:
      self.kernel = tf.contrib.seq2seq.tile_batch(
          self.kernel, multiplier=self._beam_width)

    if self.use_bias:
      self._b = self.add_weight(
          "b",
          shape=[self.units, self._rank],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
      self.bias = tf.einsum("ij,aj->ai", self._b, sigma)
      if self._use_beam and self._beam_width:
        self.bias = tf.contrib.seq2seq.tile_batch(
            self.bias, multiplier=self._beam_width)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    rank = tf.rank(inputs)
    if rank > 2:
      outputs = tf.einsum("aki,aij->akj", inputs, self.kernel)

      # Reshape the output back to the original ndim of the input.
      if not tf.executing_eagerly():
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      assert False
      # outputs = tf.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if dimension_value(input_shape[-1]) is None:
      raise ValueError(
          "The innermost dimension of input_shape must be defined, but saw: %s"
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        "units":
            self.units,
        "activation":
            tf.keras.activations.serialize(self.activation),
        "use_bias":
            self.use_bias,
        "kernel_initializer":
            tf.keras.initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self.bias_constraint)
    }
    base_config = super(HyperDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

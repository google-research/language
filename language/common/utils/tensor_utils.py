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
"""Utils for basic tensor manipulation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def shape(tensor, dim=None):
  """Gets the most specific dimension size(s) of the given tensor.

  This is a wrapper around the two ways to get the shape of a tensor: (1)
  t.get_shape() to get the static shape, and (2) tf.shape(t) to get the dynamic
  shape. This function returns the most specific available size for each
  dimension. If the static size is available, that is returned. Otherwise, the
  tensor representing the dynamic size is returned.

  Args:
    tensor: Input tensor.
    dim: Desired dimension. Use None to retrieve the list of all sizes.

  Returns:
    output = Most specific static dimension size(s).
  """
  static_shape = tensor.get_shape()
  dynamic_shape = tf.shape(tensor)
  if dim is not None:
    return static_shape[dim].value or dynamic_shape[dim]
  else:
    return [d.value or dynamic_shape[i] for i, d in enumerate(static_shape)]


def where(condition, if_true, if_false):
  """Selects slices of `if_true` and `if_false` based on `condition`.

  This is a wrapper around tf.where() that allows it to select slices at any
  level of a tensor.  (Recall that tf.where() can only select slices on the
  first axis or select individual elements).

  Example usage:

  condition = tf.constant([[True, False],
                           [False, True]])
  if_true = tf.constant([[[1, 1], [2, 2]],
                         [[3, 3], [4, 4]]])
  if_false = tf.constant([[[5, 5], [6, 6]],
                          [[7, 7], [8, 8]]])

  result = where(condition, if_true, if_false)

  assert result.eval() == [[[1, 1], [6, 6]],
                           [[7, 7], [4, 4]]]

  Args:
    condition: <bool>[...] Condition for selecting slices.
    if_true: <T>[condition.shape, ...] Values to use if `condition` is true.
    if_false: <T>[condition.shape, ...] Values to use if `condition` is false.

  Returns:
    <T>[if_true.shape] Values after conditional selection.
  """
  # TODO(terrykoo): Rewrite when tf.where() supports broadcasting: b/78594182.
  condition_rank = condition.shape.ndims
  result_rank = if_true.shape.ndims

  # Use tf.where() directly if it can handle the args.
  if condition_rank == 1 or condition_rank == result_rank:
    return tf.where(condition, if_true, if_false)

  result_dims = shape(if_true)
  inner_dims = result_dims[condition_rank:]

  # Flatten outer dims so we can select slices on axis=0.
  condition = tf.reshape(condition, [-1])
  if_true = tf.reshape(if_true, [-1] + inner_dims)
  if_false = tf.reshape(if_false, [-1] + inner_dims)

  result = tf.where(condition, if_true, if_false)

  # Restore original shape.
  return tf.reshape(result, result_dims)


def shaped_py_func(func, inputs, types, shapes, stateful=True, name=None):
  """Wrapper around tf.py_func that adds static shape information to the output.

  Args:
    func: Python function to call.
    inputs: List of input tensors.
    types: List of output tensor types.
    shapes: List of output tensor shapes.
    stateful: Whether or not the python function is stateful.
    name: Name of the op.

  Returns:
    output_tensors: List of output tensors.
  """
  output_tensors = tf.py_func(
      func=func,
      inp=inputs,
      Tout=types,
      stateful=stateful,
      name=name)
  for t, s in zip(output_tensors, shapes):
    t.set_shape(s)
  return output_tensors


def flatten(t):
  """Collapse all dimensions of a tensor but the last.

  This function also returns a function to unflatten any tensors to recover
  the original shape (aside from the last dimension). This is useful for
  interfacing with functions that are agnostic to all dimensions but the last.
  For example, if we want to apply a linear projection to a batched sequence
  of embeddings:

    t = tf.random_uniform([batch_size, sequence_length, embedding_size])
    w = tf.get_variable("w", [embedding_size, projection_size])
    flat_t, unflatten = flatten(t)
    flat_projected = tf.matmul(flat_t, w)
    projected = unflatten(flat_projected)

  Args:
    t: [dim_1, ..., dim_(n-1), dim_n]

  Returns:
    output: [dim_1 * ... * dim_(n-1), dim_n]
    _unflatten: A function that when called with a flattened tensor returns the
        unflattened version, i.e. reshapes any tensor with shape
        [dim_1 * ... * dim_(n-1), dim_new] to [dim_1, ..., dim_(n-1), dim_new].
  """
  input_shape = shape(t)
  if len(input_shape) > 2:
    t = tf.reshape(t, [-1, input_shape[-1]])

  def _unflatten(flat_t):
    if len(input_shape) > 2:
      return tf.reshape(flat_t, input_shape[:-1] + [shape(flat_t, -1)])
    else:
      return flat_t
  return t, _unflatten


def transpose_batch_time(tensor):
  """Transposes the batch and time dimensions of the `tensor`.

  If the `tensor` has less than 2 dimensions, then no operation is performed.
  Otherwise, swaps the first 2 dimensions, which are assumed to be the batch and
  time dimensions.

  Args:
    tensor: Tensor to transpose.

  Returns:
    Possibly-transposed version of `tensor`.

  Raises:
    ValueError: If the `tensor` has unknown rank.
  """
  rank = tensor.shape.ndims
  if rank is None:
    raise ValueError("Tensor with unknown rank")
  if rank < 2:
    return tensor
  return tf.transpose(tensor, perm=[1, 0] + range(2, rank))


def sequence_mask(lengths, maxlen=None, dtype=tf.bool, transpose=False):
  """Returns a sequence mask, like tf.sequence_mask().

  Unlike tf.sequence_mask(), this can also generate a transposed mask, which is
  convenient for working with time-major sequences.

  Args:
    lengths: <int>[batch_size] Sequence lengths.
    maxlen: <int>[] Maximum length, or None to compute the max of `lengths`.
    dtype: DType of the generated mask.
    transpose: Whether to generate the transpose of the mask.

  Returns:
    <dtype>[maxlen, batch_size] Sequence mask.
  """
  with tf.name_scope("sequence_mask"):
    if maxlen is None:
      maxlen = tf.reduce_max(lengths)
    positions = tf.range(maxlen, dtype=lengths.dtype)

    positions_singleton_axis = 1 if transpose else 0
    lengths_singleton_axis = 0 if transpose else 1
    positions = tf.expand_dims(positions, axis=positions_singleton_axis)
    lengths = tf.expand_dims(lengths, axis=lengths_singleton_axis)
    mask = positions < lengths

    if dtype != tf.bool:
      mask = tf.cast(mask, dtype)
    return mask

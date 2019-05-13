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

import math

import tensorflow as tf


def linear_interpolation(t, minimum, maximum):
  t_min = tf.reduce_min(t)
  t_max = tf.reduce_max(t)
  return minimum + (t - t_min) * (maximum - minimum) / (t_max - t_min)


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int
  Returns:
    Tensor of shape [length, 2 * num_timescales].
  """
  positions = tf.to_float(tf.range(length))
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def batch_boolean_mask(mask):
  """Get indices of true values.

  Args:
    mask: [batch_size, num_values]

  Returns:
    true_indices: [batch_size, max_true]
    gathered_mask: [batch_size, max_true]
  """
  # [batch_size, num_values]
  mask = tf.to_int32(mask)

  # [batch_size]
  num_true = tf.reduce_sum(mask, 1)

  # []
  max_true = tf.reduce_max(num_true)

  # [batch_size, max_true]
  gathered_mask, true_indices = tf.nn.top_k(mask, max_true)
  gathered_mask = tf.cast(gathered_mask, tf.bool)

  return gathered_mask, true_indices


def trues_like(t):
  return tf.cast(tf.ones_like(t), tf.bool)


def falses_like(t):
  return tf.cast(tf.zeros_like(t), tf.bool)


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


def batch_gather(tensors, indices):
  """Gather indices into any dimension rather than dimension 0 of a tensor.

  Gathering is usually performed on dimension 0 of a tensor, such as a word
  embedding matrix. In the case where the target is batched, such as selecting
  a token or a set of tokens for each sequence in the batch, then this function
  will gather the desired elements correctly. For example, given inputs:

  tensor = [[11, 12, 13],
            [21, 22, 23]]
  indices = [[2, 1, 0, 1],
             [0, 1, 1, 2]])

  we expect `batch_gather` to return the following tensor:

  gathered = [[13, 12, 11, 12],
              [21, 22, 22, 23]]

  Args:
    tensors: Tensor or list of Tensors with shape
        [batch_size, sequence_length, d1, d2, ...].
    indices: [batch_size, i1, i2, ...]

  Returns:
    gathered: [batch_size, i1, i2, ..., d1, d2, ...]
  """
  is_singleton = not isinstance(tensors, (list, tuple))

  if is_singleton:
    tensors = [tensors]

  sequence_dim = len(indices.get_shape()) - 1
  if sequence_dim > 1:
    batch_shape = shape(tensors[0])[:sequence_dim]
    batch_size = 1
    for s in batch_shape:
      batch_size *= s
    tensors = [tf.reshape(t, [batch_size] + shape(t)[sequence_dim:])
               for t in tensors]
    indices = tf.reshape(indices, [batch_size] + shape(indices)[sequence_dim:])

  batch_size = shape(indices, 0)
  sequence_length = shape(tensors[0], 1)

  # [batch_size]
  offset = tf.range(batch_size) * sequence_length

  # [batch_size, 1, 1, ...]
  for _ in range(len(shape(indices)) - 1):
    offset = tf.expand_dims(offset, 1)

  offset_indices = indices + offset

  gathered_tensors = []
  for t in tensors:
    # [batch_size * sequence_length, d1, d2, ...]
    flat_tensor = tf.reshape(t, [batch_size * sequence_length] + shape(t)[2:])

    if t.dtype in (tf.int32, tf.int64, tf.bool):
      # Cast to float to enable gathering on GPU.
      flat_tensor = tf.to_float(flat_tensor)

    gathered = tf.gather(flat_tensor, offset_indices)

    if t.dtype in (tf.int32, tf.int64, tf.bool):
      # Undo previous cast.
      gathered = tf.cast(gathered, t.dtype)

    # [batch_size, i1, i2, ..., d1, d2, ...]
    gathered_tensors.append(gathered)

  if sequence_dim > 1:
    gathered_tensors = [tf.reshape(
        t, batch_shape + shape(t)[sequence_dim-1:]) for t in gathered_tensors]
  if is_singleton:
    return gathered_tensors[0]
  else:
    return gathered_tensors

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
"""Tensor manipulation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def boolean_mask(tensor, mask):
  """Select elements from tensor where mask is True, and pads the rest with 0.

  Example:
    tensor = [1 2 3 4 5 6 7]
    mask   = [T T F F T T T]
    output = [1 2 5 6 7 0 0]

  Args:
    tensor: <T> [batch_size, dim, ...]
    mask: <bool> [batch_size, dim]

  Returns:
    masked_tensor: <T> [batch_size, dim, ...]. The first k items of row i
      correspond to the k truthy values in row i of the mask tensor,
      indexed into the provided tensor.
  """
  batch_size = shape(mask, 0)
  dim = shape(mask, 1)

  # Bring the coordinates of un-masked elements to the front.
  mask = tf.cast(mask, tensor.dtype)
  indices = tf.argsort(mask, axis=1, direction="DESCENDING", stable=True)
  batch_indices = tf.expand_dims(tf.range(tf.shape(mask)[0]), 1)
  batch_indices = tf.tile(batch_indices, [1, tf.shape(indices)[1]])
  coords = tf.stack([batch_indices, indices], axis=2)

  # Gather and zero masked elements.
  selected_tensor = tf.gather_nd(tensor, coords)
  selected_mask = tf.gather_nd(mask, coords)
  reshape = [batch_size, dim] + [1] * (tensor.get_shape().ndims - 2)
  selected_mask = tf.reshape(selected_mask, reshape)
  masked_tensor = selected_tensor * selected_mask

  return masked_tensor


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
    Most specific static dimension size(s).
  """
  static_shape = tensor.get_shape()
  dynamic_shape = tf.shape(tensor)
  if dim is not None:
    return static_shape[dim].value or dynamic_shape[dim]
  else:
    return [d.value or dynamic_shape[i] for i, d in enumerate(static_shape)]


def unravel_index_2d(indices, dims):
  """Unravel index, for 2D inputs only.

  See Numpy's unravel.

  Args:
    indices: <int32> [num_elements], coordinates into 2D row-major tensor.
    dims: (N, M), dimensions of the 2D tensor.

  Returns:
    coordinates: <int32> [2, num_elements], row (1st) and column (2nd) indices.
  """
  row_inds = tf.floordiv(indices, dims[1])
  col_inds = tf.floormod(indices, dims[1])
  return tf.stack([row_inds, col_inds], axis=0)


def tile_batch(t, multiplier):
  """Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

  Args:
    t: Tensor shaped [batch_size, ...]
    multiplier: <int32>

  Returns:
    A possibly nested structure of Tensor shaped [batch_size * multipler, ...]
  """

  def _tile_batch(t, multiplier):
    """Core single-tensor implementation of tile_batch."""
    tensor = tf.convert_to_tensor(t)
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = multiplier
    tiled = tf.tile(tensor, tile_dims)
    return tf.reshape(tiled, [-1] + shape(tensor)[2:])

  return tf.nest.map_structure(lambda t_: _tile_batch(t_, multiplier), t)


def gather(src, indices):
  """A partial implementation of PyTorch gather, for dim=1.

  Given a src tensor of variable size, this function gathers multiple elements
  along the 2nd dimension (dim=1), for every batch in the 1st dimension (dim=0).

  Args:
    src: <T> [batch_size, d_1, ..., d_n]
    indices: <int32> [batch_size, num_indices]

  Returns:
    output: <T> [batch_size, num_indices, d_2, ..., d_n]
  """
  # Hereby begins the painful dance of tf.gather_nd.
  batch_size = shape(indices, 0)
  num_indices = shape(indices, 1)

  # [batch_size, 1]
  batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1])

  # [batch_size, num_indices]
  batch_idx = tf.tile(batch_idx, [1, num_indices])

  # [batch_size * num_indices]
  batch_idx = tf.reshape(batch_idx, [-1])

  # [batch_size * num_indices]
  flat_indices = tf.reshape(indices, [-1])

  # [batch_size * num_indices, 2]
  idx = tf.stack([batch_idx, flat_indices], 1)

  # [batch_size * num_indices, d_2, ..., d_n]
  output = tf.gather_nd(src, idx)

  # [batch_size, num_indices, d_2, ..., d_n]
  output = tf.reshape(output, [batch_size, num_indices] +
                      [shape(output, i) for i in range(1, len(output.shape))])

  return output

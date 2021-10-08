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
"""Contains Jax utils."""



import jax
import jax.numpy as jnp

from language.mentionmemory.utils.custom_types import Array

_SMALL = 1e-8


def matmul_slice(array, indices):
  """Convenience function for indexing that differs along first dimension.

  The function supports batched and not-batched input:

  (1) Input is NOT batched, meaning that `indices` has shape [n_index] and
  `array` has shape [array_len, ...]. The output is precisely array[indices].
  (2) Input is batched, so `indices` has shape [batch, n_index] and
  `array` has shape [batch, array_len, ...].

  Values of the `indices` should be from 0 to array_len - 1.

  Args:
    array: array to be indexed over with shape either [batch, array_len, ...] or
      [array_len, ...].
    indices: array of indices with shape either [batch, n_index] or [n_index].

  Returns:
    Sliced values.
  """
  if indices.ndim == 1:
    # Case (1): The input is not batched, so neither `array` not `indices` have
    # batch dimension
    array_len = array.shape[0]
    one_hot_indices = jax.nn.one_hot(indices, array_len, dtype=array.dtype)
    value = jnp.einsum('p...,ip->i...', array, one_hot_indices)
  else:
    # Case (2): The input is batched, so both `array` and `indices` have
    # extra batch dimension (the first one).
    array_len = array.shape[1]
    one_hot_indices = jax.nn.one_hot(indices, array_len, dtype=array.dtype)
    value = jnp.einsum('bp...,bip->bi...', array, one_hot_indices)
  return value


def matmul_2d_index_select(array, indices):
  """Function subselects an array with 2D index using matrix multiplication.

  Args:
    array: [batch, seq, ...] array to be indexed over.
    indices: Tuple[[n_index], [n_index]] arrays of indices.

  Returns:
    Subselected array: array[indices[0], indices[1]].
  """
  batch_size = array.shape[0]
  seq_len = array.shape[1]
  flat_seq_len = seq_len * batch_size
  flat_array = jnp.reshape(array, (flat_seq_len, *array.shape[2:]))
  one_hot_indices = jax.nn.one_hot(
      indices[0] * seq_len + indices[1], flat_seq_len, dtype=array.dtype)
  value = jnp.einsum('b...,ib->i...', flat_array, one_hot_indices)
  return value


def matmul_index_add(array, indices, values):
  """Convenience function for index add that differs along first dimension.

  Args:
    array: [batch, seq, dim] array to be added to.
    indices: [batch, n_index] array of indices to add to.
    values: [batch, n_index, dim] array of values to add.

  Returns:
    Added array.
  """
  seq_len = array.shape[1]
  one_hot_indices = jax.nn.one_hot(indices, seq_len, dtype=array.dtype)
  encoding_addition = jnp.einsum('bqp,bqd->bpd', one_hot_indices, values)
  added_array = array + encoding_addition
  return added_array


def matmul_2d_index_add(array, indices,
                        values):
  """Convenience function for 2D index add.

  Args:
    array: [batch, seq, dim] array to be added to.
    indices: Tuple[[n_index], [n_index]] arrays of indices.
    values: [n_index, dim] array of values to add.

  Returns:
    Added array: array[indices[0], indices[1]] + values.
  """
  batch_size = array.shape[0]
  seq_len = array.shape[1]
  flat_seq_len = seq_len * batch_size
  flat_array = jnp.reshape(array, (flat_seq_len, *array.shape[2:]))
  one_hot_indices = jax.nn.one_hot(
      indices[0] * seq_len + indices[1], flat_seq_len, dtype=array.dtype)
  encoding_addition = jnp.einsum('ib,i...->b...', one_hot_indices, values)
  added_array = flat_array + encoding_addition
  added_array = jnp.reshape(added_array, array.shape)
  return added_array


@jax.vmap
def vmap_slice(array, indices):
  """Convenience function for indexing that differs along first dimension ."""
  return array[indices]


@jax.vmap
def vmap_index_add(array, indices, values):
  """Convenience function for index add that differs along first dimension."""
  return jax.ops.index_add(array, indices, values)


def cosine_similarity(a, b):
  """Computes batched cosine similarity between two 2D arrays."""
  a_norm = jnp.linalg.norm(a, axis=-1)
  b_norm = jnp.linalg.norm(b, axis=-1)
  dot = jnp.einsum('bd,bd->b', a, b)
  return dot / (_SMALL + a_norm * b_norm)

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
"""Contains utils for mention-related Jax computations."""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp

from language.mentionmemory.utils.custom_types import Array


def all_compare(xs: Array, ys: Array) -> Array:
  return jnp.expand_dims(xs, 1) == jnp.expand_dims(ys, 0)


def all_compare_without_pad(xs: Array, ys: Array) -> Array:
  """Performs all-to-all comparison of two 1D arrays ignoring pad values."""
  xs_expanded = jnp.expand_dims(xs, 1)
  ys_expanded = jnp.expand_dims(ys, 0)
  same_ids = xs_expanded == ys_expanded
  same_ids = jnp.logical_and(same_ids, xs_expanded != 0)
  same_ids = jnp.logical_and(same_ids, ys_expanded != 0)
  return same_ids


def sum_by_batch_position(mention_batch_positions: Array, values: Array,
                          batch_size: int) -> Array:
  """Sum values per position in the batch."""
  position2item = (
      jnp.expand_dims(jnp.arange(batch_size), 1) == mention_batch_positions)
  position2item = position2item.astype(jnp.int32)

  # [batch_size, ...]
  values_sum_per_batch_position = jnp.einsum('bm,m...->b...', position2item,
                                             values)
  return values_sum_per_batch_position


def get_device_id(axis_name: str) -> Optional[int]:
  try:
    return jax.lax.axis_index(axis_name)
  except NameError:
    return None


def get_globally_consistent_batch_positions(
    mention_batch_positions: Array, batch_size: int) -> Tuple[Array, Array]:
  """Adjust batch positions to be unique in the global batch.

  If the function is executed only on a single device (outside of `pmap`) then
  the results is just `mention_batch_positions`. Otherwise, the function
  returns mention_batch_positions for the first device, mention_batch_positions
  + batch size for the second device, 2*batch_size + mention_batch_positions for
  third, etc. The method returns local and global mention_batch_positions with
  adjusted batch positions.

  For example, let batch size be 3. There are two devices with
  mention_batch_positions equal to [0, 1, 2] on the first device
  and [2, 0, 0] on the second device. Then the function
  returns [0, 1, 2], [0, 1, 2, 5, 3, 3] on the first device
  and [5, 3, 3], [0, 1, 2, 5, 3, 3] on the second device.

  Args:
    mention_batch_positions: position of a mention in its batch.
    batch_size: batch size per device.

  Returns:
    Local mention_batch_positions and global mention_batch_positions with
    globally consistent IDs.
  """
  device_id = get_device_id('batch')
  if device_id is not None:
    mention_batch_positions = mention_batch_positions + batch_size * device_id
    all_mention_batch_positions = jax.lax.all_gather(mention_batch_positions,
                                                     'batch')
    all_mention_batch_positions = all_mention_batch_positions.reshape([-1])
    return mention_batch_positions, all_mention_batch_positions
  else:
    return mention_batch_positions, mention_batch_positions


def mask_duplicate_ids(batch_positions: Array, ids: Array) -> Array:
  """Zero out duplicate items within the same batch position.

  The function is a variation of a `unique` function applied to ids array
  batch_position-wise. The difference with a `np.unique` is that instead of
  discarding repeated elements (ids), the function makes them 0. For example,
  if `batch_positions` is [1, 1, 1, 2, 2] and `ids` is [1, 1, 2, 2, 2] then
  the output is [1, 0, 2, 2, 0].

  Args:
    batch_positions: position of a mention in its batch.
    ids: IDs of mentions in the batch.

  Returns:
    A modified version of `ids` where all duplicate IDs in the same batch
    position are set to zero.
  """
  same_position = all_compare(batch_positions, batch_positions)
  same_ids = all_compare(ids, ids)
  same_ids = jnp.logical_and(same_ids, same_position)
  item_is_not_duplicate = jnp.tril(same_ids).sum(axis=-1) <= 1
  ids = ids * item_is_not_duplicate
  return ids


def get_num_common_unique_items(batch_positions: Array, ids: Array,
                                batch_size: int) -> Array:
  """Get the number of unique entity IDs shared between samples in the batch.

  The function produces two matrices `num_common_ids_between_samples` of shape
  [batch_size, batch_size * n_devices] and `num_common_ids_between_mentions` of
  shape [n_mentions, n_mentions * n_devices]. The first matrix is the number
  of unique entity IDs that exists in both sample A and sample B. The second
  matrix is similar, but is indexed with respect to individual mentions --
  how many unique entity IDs exists for a sample that contains mention A and
  a sample that contains mention B.

  More formally, let `all_ids` be IDs concatenated from all devices and
  `global_batch_position` be global batch positions produced by the
  `get_globally_consistent_batch_positions` method.

  `num_common_ids_between_samples[i, j] = k` <=> The number of unique IDs in the
  intersection of ids[batch_positions == i] and
  all_ids[global_batch_position == j] is k.

  `num_common_ids_between_mentions[i, j] = k` <=> If b_i is a batch position
  corresponding to a local mention i and b_j is a global batch position
  corresponding to a global mention j then
  num_common_ids_between_samples[b_i, b_j] = k

  Args:
    batch_positions: position of a mention in its batch.
    ids: IDs of mentions in the batch.
    batch_size: batch size per device.

  Returns:
    Matrices with the number of unique common elements between a row in local
    batch and a row in global batch. Shapes are
    [batch_size, n_devices * batch_size] between samples in a batch and
    [n_mentions, n_devices * n_mentions] between mentions in a batch.
  """
  ids = mask_duplicate_ids(batch_positions, ids)

  device_id = get_device_id('batch')
  if device_id is not None:
    all_ids = jax.lax.all_gather(ids, 'batch')
    n_devices = all_ids.shape[0]
    all_ids = all_ids.reshape([-1])
  else:
    n_devices = 1
    all_ids = ids

  # [n_mentions, n_global_mentions]
  same_ids = all_compare_without_pad(ids, all_ids)
  same_ids = same_ids.astype(jnp.int32)

  (_, global_batch_positions) = get_globally_consistent_batch_positions(
      batch_positions, batch_size)

  # [batch_size, n_mentions]
  position2item = (
      jnp.expand_dims(jnp.arange(batch_size), 1) == batch_positions)
  position2item = position2item.astype(jnp.int32)

  # [global_batch_size, n_global_mentions]
  item2global_position = (
      jnp.expand_dims(jnp.arange(n_devices * batch_size),
                      1) == global_batch_positions)
  item2global_position = item2global_position.astype(jnp.int32)

  # [batch_size, n_global_mentions]
  num_common_ids_between_samples = sum_by_batch_position(
      batch_positions, same_ids, batch_size)

  # [batch_size, global_batch_size]
  num_common_ids_between_samples = jnp.einsum('bn,gn->bg',
                                              num_common_ids_between_samples,
                                              item2global_position)
  num_common_ids_between_mentions = jnp.einsum('bm,bg->mg', position2item,
                                               num_common_ids_between_samples)
  num_common_ids_between_mentions = jnp.einsum('gn,mg->mn',
                                               item2global_position,
                                               num_common_ids_between_mentions)
  return num_common_ids_between_samples, num_common_ids_between_mentions  # pytype: disable=bad-return-type  # jax-ndarray

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
"""Contains memory extraction layer."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import


class MemoryExtractionLayer(nn.Module):
  """Extracts mention memory from global batch.

  Given a global batch of passages across devices, this layer generates a
  mention memory by extracting mention key and value representations from
  contextual representations of tokens in the mentions. The keys and values are
  then all-gathered such that each device has access to the same table of keys
  and values across all devices. This layer can only be run inside a pmap with
  'batch' axis.

  Attributes:
    memory_key_dim: dimensionality of memory keys.
    memory_value_dim: dimensionality of memory values.
    dtype: precision of computation.
  """

  memory_key_dim: int
  memory_value_dim: int
  dtype: Dtype

  def setup(self):
    self.key_projector = nn.Dense(
        features=self.memory_key_dim,
        dtype=self.dtype,
    )

    self.value_projector = nn.Dense(
        features=self.memory_value_dim,
        dtype=self.dtype,
    )

  def __call__(
      self,
      encoding: Array,
      mention_batch_positions: Array,
      mention_start_positions: Array,
      mention_end_positions: Array,
      mention_mask: Array,
      mention_entity_ids: Array,
  ) -> Array:
    """.

    Args:
      encoding: [batch_size, n_tokens, hidden_size].
      mention_batch_positions: [n_mentions].
      mention_start_positions: [n_mentions].
      mention_end_positions: [n_mentions].
      mention_mask: [n_mentions].
      mention_entity_ids: [n_mentions].

    Returns:
      Array of entity linking attention scores, shape [n_mentions, hidden_size].
    """
    mention_start_encodings = jut.matmul_2d_index_select(
        encoding, (mention_batch_positions, mention_start_positions))
    mention_end_encodings = jut.matmul_2d_index_select(
        encoding, (mention_batch_positions, mention_end_positions))
    projection_input = jnp.concatenate(
        (mention_start_encodings, mention_end_encodings), axis=-1)
    n_mentions = projection_input.shape[0]
    local_memory_keys = self.key_projector(projection_input)
    local_memory_values = self.value_projector(projection_input)

    memory_keys = jax.lax.all_gather(local_memory_keys, 'batch')
    memory_values = jax.lax.all_gather(local_memory_values, 'batch')
    memory_mask = jax.lax.all_gather(mention_mask, 'batch')
    memory_entity_ids = jax.lax.all_gather(mention_entity_ids, 'batch')
    n_devices = memory_keys.shape[0]

    memory_keys = memory_keys.reshape(n_devices * n_mentions,
                                      self.memory_key_dim)
    memory_values = memory_values.reshape(n_devices * n_mentions,
                                          self.memory_value_dim)
    memory_mask = memory_mask.reshape(n_devices * n_mentions)
    memory_entity_ids = memory_entity_ids.reshape(n_devices * n_mentions)

    return_dict = {
        'memory_keys': memory_keys,
        'memory_values': memory_values,
        'memory_mask': memory_mask,
        'memory_entity_ids': memory_entity_ids,
        'local_memory_keys': local_memory_keys,
        'local_memory_values': local_memory_values,
    }

    return return_dict  # pytype: disable=bad-return-type  # jax-ndarray

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
"""Contains entity attention layer."""

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import

_SMALL_NUMBER = 1e-8


class EntityAttentionLayer(nn.Module):
  """Performs attention update over entity embeddings for passage mentions.

    Attributes:
    entity_dim: dimensionality of entity representations.
    hidden_size: dimensionality of input token representations.
    dtype: precision of computation.
  """

  entity_dim: int
  hidden_size: int
  dtype: Dtype
  layer_norm_epsilon: float = 1e-12

  def setup(self):
    self.mention_query_projector = nn.Dense(
        features=self.entity_dim,
        dtype=self.dtype,
    )

    self.entity_projector = nn.Dense(
        features=self.hidden_size,
        dtype=self.dtype,
    )
    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

  def __call__(
      self,
      encoded_input: Array,
      mention_batch_positions: Array,
      mention_start_positions: Array,
      mention_end_positions: Array,
      mention_mask: Array,
      entity_embeddings: Array,
  ) -> Dict[str, Array]:
    """Perform attention update over entity embedding table.

    Args:
      encoded_input: [batch_size, n_tokens, hidden_size].
      mention_batch_positions: [n_mentions].
      mention_start_positions: [n_mentions].
      mention_end_positions: [n_mentions].
      mention_mask: [n_mentions] attention mask to prevent updates from padding
        mentions.
      entity_embeddings: entity embedding table.

    Returns:
      Updated input, mention encodings and entity attention scores.
    """

    mention_start_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_start_positions))
    mention_end_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_end_positions))
    mention_encodings = self.mention_query_projector(
        jnp.concatenate((mention_start_encodings, mention_end_encodings),
                        axis=-1))

    scores = jnp.einsum('qd,ed->qe', mention_encodings, entity_embeddings)
    attention_weights = nn.softmax(scores, axis=-1)

    retrieved_values = jnp.einsum('qe,ed->qd', attention_weights,
                                  entity_embeddings)
    retrieved_values = self.entity_projector(retrieved_values)
    retrieved_values = retrieved_values * jnp.expand_dims(mention_mask, -1)

    encoded_input = jut.matmul_2d_index_add(
        encoded_input, (mention_batch_positions, mention_start_positions),
        retrieved_values)
    encoded_input = self.layer_norm(encoded_input)

    # The cosine similarity is computed as dot product divided by norms of
    # both vectors.
    mention_encodings_norm = jnp.linalg.norm(mention_encodings, axis=-1)
    entity_embeddings_norm = jnp.linalg.norm(entity_embeddings, axis=-1)
    cos_scores = scores
    cos_scores = cos_scores / (
        _SMALL_NUMBER + jnp.expand_dims(mention_encodings_norm, 1))
    cos_scores = cos_scores / (
        _SMALL_NUMBER + jnp.expand_dims(entity_embeddings_norm, 0))

    return {
        'encoded_output': encoded_input,
        'mention_encodings': mention_encodings,
        'cosine_similarity': cos_scores,
        'attention_weights': attention_weights,
    }

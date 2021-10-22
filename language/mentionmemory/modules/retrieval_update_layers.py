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
"""Contains layers for incorporating retrievals into passage representation.

In particular, given an input representation sequence, retrieved values and
retrieval scores, these layers produce a new input representation sequence.
"""

import abc

import flax.linen as nn
import jax.numpy as jnp

from language.mentionmemory.modules import mlp
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import


class RetrievalUpdateLayer(abc.ABC):
  """Interface for retrieval update layers.

  These layers incorporate retrieval results from mention queries into the input
  representation.
  """

  def __call__(
      self,
      encoded_input,
      retrieval_values,
      retrieval_scores,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      deterministic,
  ):
    """Incorporates retrieval into input.

    Args:
      encoded_input: [batch_size, n_tokens, hidden_size] input representation.
      retrieval_values: [n_mentions, n_retrieval, retrieval_dim] retrievals.
      retrieval_scores: [n_mentions, n_retrieval] retrieval scores.
      mention_batch_positions: [n_mentions] mention sample position in batch.
      mention_start_positions: [n_mentions] mention start position in input.
      mention_end_positions: [n_mentions] mention end position in input.
      mention_mask: [n_mentions] attention mask to prevent updates from padding.
      deterministic: don't apply dropout if true.

    Returns:
      Updated input.
    """


class AdditiveUpdate(nn.Module, RetrievalUpdateLayer):
  """Adds projected retrieved values to input encoding.

  Attributes:
    input_dim: dimension of input representation.
    dtype: precision of computation.
  """

  input_dim: int
  dtype: Dtype
  layer_norm_epsilon: float

  def setup(self):
    self.retrieval_projector = nn.Dense(
        features=self.input_dim,
        dtype=self.dtype,
    )

    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

  def __call__(
      self,
      encoded_input,
      retrieval_values,
      retrieval_scores,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      deterministic,
  ):

    weighted_values = jnp.einsum('qk,qkd->qd', retrieval_scores,
                                 retrieval_values)
    projected_values = self.retrieval_projector(weighted_values)
    projected_values = projected_values * mention_mask.reshape(-1, 1)
    encoded_input = jut.matmul_2d_index_add(
        encoded_input, (mention_batch_positions, mention_start_positions),
        projected_values)

    encoded_input = self.layer_norm(encoded_input)

    return encoded_input


class DummyUpdate(nn.Module, RetrievalUpdateLayer):
  """Dummy update layer that returns input as-is."""

  input_dim: int
  dtype: Dtype
  layer_norm_epsilon: float

  def __call__(
      self,
      encoded_input,
      retrieval_values,
      retrieval_scores,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      deterministic,
  ):
    return encoded_input


class ConcatMLPUpdate(nn.Module, RetrievalUpdateLayer):
  """First applies MLP, then pools, then applies another MLP to compute update.

  This layer separately concatenates the passage mention value to each
  retrieved vector, and applies an MLP. This is followed by weighted pooling
  step, using the attention scores. Finally another MLP is applied to compute
  the update. The main idea is that the mention passage representation is able
  to interact with each retrieved vector separately, leading to a more
  expressive update.

  Attributes:
    input_dim: dimension of input representation.
    retrieval_dim: dimension of retrieval vectors.
    hidden_dim: hidden dimension in mlp layers.
    n_additional_concat_layers: number of additional mlp layers to apply to
      concatenated mention and retrieval value beyond the first.
    n_pooled_layers: number of mlp layers to apply to pooled representation.
    dropout_rate: dropout rate.
    dtype: precision of computation.
    layer_norm_epsilon: epsilon value for layer norm.
  """

  input_dim: int
  retrieval_dim: int
  hidden_dim: int
  n_additional_concat_layers: int
  n_pooled_layers: int
  dropout_rate: float
  dtype: Dtype
  layer_norm_epsilon: float

  def setup(self):
    self.value_projection = nn.Dense(
        features=self.retrieval_dim,
        dtype=self.dtype,
    )

    self.concat_mlp = nn.Dense(
        features=self.hidden_dim,
        dtype=self.dtype,
    )
    self.concat_dense = nn.Dense(
        features=self.input_dim,
        dtype=self.dtype,
    )
    self.concat_dropout = nn.Dropout(self.dropout_rate)

    self.additional_concat_mlp_layers = [
        mlp.MLPBlock(  # pylint: disable=g-complex-comprehension
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
        ) for _ in range(self.n_additional_concat_layers)
    ]

    self.pooled_mlp_layers = [
        mlp.MLPBlock(  # pylint: disable=g-complex-comprehension
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
        ) for _ in range(self.n_pooled_layers)
    ]

    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

  def __call__(
      self,
      encoded_input,
      retrieval_values,
      retrieval_scores,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      deterministic,
  ):

    # Generate mention values from input representation
    mention_start_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_start_positions))
    mention_end_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_end_positions))

    passage_mention_values = self.value_projection(
        jnp.concatenate((mention_start_encodings, mention_end_encodings),
                        axis=-1))
    k_retrieval = retrieval_scores.shape[-1]
    passage_mention_values = jnp.expand_dims(passage_mention_values, axis=1)
    passage_mention_values = jnp.tile(passage_mention_values,
                                      (1, k_retrieval, 1))

    # Generate concatenated values of shape [mentions, k, 2 * retrieval_dim]
    concat_values = jnp.concatenate((passage_mention_values, retrieval_values),
                                    axis=-1)

    # MLP over concatenation mention value and individual retrieved value
    concat_values = nn.gelu(self.concat_mlp(concat_values))
    concat_values = self.concat_dense(concat_values)
    concat_values = self.concat_dropout(concat_values, deterministic)

    # Additional MLP layers
    for concat_mlp_layer in self.additional_concat_mlp_layers:
      concat_values = concat_mlp_layer(concat_values, deterministic)

    pooled_values = jnp.einsum('qk,qkd->qd', retrieval_scores, concat_values)

    # MLP layers applied to pooled retrieval values
    for pooled_mlp_layer in self.pooled_mlp_layers:
      pooled_values = pooled_mlp_layer(pooled_values, deterministic)
    pooled_values = pooled_values * mention_mask.reshape(-1, 1)

    encoded_input = jut.matmul_2d_index_add(
        encoded_input, (mention_batch_positions, mention_start_positions),
        pooled_values)

    encoded_input = self.layer_norm(encoded_input)

    return encoded_input


RETRIEVAL_UPDATE_REGISTRY = {
    'additive': AdditiveUpdate,
    'concat_mlp': ConcatMLPUpdate,
    'dummy': DummyUpdate,
}

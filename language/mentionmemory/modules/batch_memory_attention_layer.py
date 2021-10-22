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
"""Contains batch memory attention layer."""



import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import retrieval_update_layers
from language.mentionmemory.modules import topk_similarity_layer
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import
import ml_collections

_LARGE_NUMBER = 1e10


class BatchMemoryAttentionLayer(nn.Module):
  """Performs attention update over batch memory table for passage mentions.

  Given an input sequence representation and passage mention positions, this
  layer applies single-head dot-product top-k attention over a memory table
  generated from batch mention representations, generating mention
  representations from input representation to use as queries. The attention is
  followed by a layer norm.

  Attributes:
    memory_key_dim: dimensionality of memory key.
    input_dim: dimensionality of input token representations.
    memory_update_type: means by which retrieved memory vectors are incorporated
      into input representation, such as simple addition or concatenation + MLP.
    memory_update_config: hyperparameters for the update layer, beyond input
      dimension and datatype.
    dtype: precision of computation.
    k_top: top-k retrieved memory vectors.
    rows: if applying approximate top-k, number of rows to reshape memory keys
      into. Governs tradeoff between recall and speed, with more rows leading to
      higher recall and lower speed.
    layer_norm_epsilon: epsilon of layer norm.
  """

  memory_key_dim: int
  input_dim: int
  memory_update_type: str
  memory_update_config: ml_collections.FrozenConfigDict
  dtype: Dtype
  k_top: Optional[int] = None
  rows: Optional[int] = None
  layer_norm_epsilon: float = 1e-12

  def setup(self):
    self.query_projector = nn.Dense(
        features=self.memory_key_dim,
        dtype=self.dtype,
    )

    self.topk_similarity = topk_similarity_layer.TopKSimilarityLayer(
        k_top=self.k_top,
        splits=1,
    )

    self.update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        self.memory_update_type](
            input_dim=self.input_dim,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
            **self.memory_update_config)

  def __call__(
      self,
      encoding,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      memory_keys,
      memory_values,
      memory_mask,
      memory_entity_ids,
      deterministic,
  ):
    """Perform attention update over memory table.

    Args:
      encoding: [batch_size, n_tokens, hidden_size] input representation.
      mention_batch_positions: [n_mentions] mention sample position in batch.
      mention_start_positions: [n_mentions] mention start position in input.
      mention_end_positions: [n_mentions] mention end position in input.
      mention_mask: [n_mentions] attention mask to prevent updates from padding.
      memory_keys: [memory_size, memory_key_dim] mention memory keys.
      memory_values: [memory_size, memory_value_dim] mention memory values.
      memory_mask: [memory_size] mask for valid mentions in memory.
      memory_entity_ids: [memory_size] mention memory entity ids.
      deterministic: don't apply dropout if true.

    Returns:
      Updated input, loss and logging helper dicts.
    """
    loss_helpers, logging_helpers = {}, {}

    # We generate mention representations to use as queries for similarity
    # search by concatenating start and end tokens for each mention and
    # projecting the concatenation with a dense layer.
    mention_start_encodings = jut.matmul_2d_index_select(
        encoding, (mention_batch_positions, mention_start_positions))
    mention_end_encodings = jut.matmul_2d_index_select(
        encoding, (mention_batch_positions, mention_end_positions))

    queries = self.query_projector(
        jnp.concatenate((mention_start_encodings, mention_end_encodings),
                        axis=-1))

    n_queries = queries.shape[0]

    # For attention over entire memory table, we do not want to duplicate the
    # entire memory table for each query. Instead, we perform an
    # attention-weighted sum to produce a single value. We then feed this value
    # to the update layer as a set of retrieved values of size 1, with score 1.
    if self.k_top is None:
      loss_helpers['top_entity_ids'] = jnp.tile(memory_entity_ids,
                                                (n_queries, 1))
      scores = jnp.einsum('qd,md->qm', queries, memory_keys)
      scores = scores - (1 - memory_mask) * _LARGE_NUMBER
      true_attention_weights = nn.softmax(scores, axis=-1)
      loss_helpers['memory_attention_weights'] = true_attention_weights
      top_values = jnp.einsum('qm,md->qd', true_attention_weights,
                              memory_values)
      # Expand value as though it were a set of retrieved values for each query.
      # Shape (n_queries, 1, memory_value_dim)
      top_values = jnp.expand_dims(top_values, axis=1)
      # Generate pseudo-score (n_queries, 1).
      attention_weights = jnp.ones_like(top_values, shape=(n_queries, 1))
    else:
      # Reshape memory keys for use in approximate top-k similarity layer.
      memory_keys = memory_keys.reshape(self.rows, -1, self.memory_key_dim)
      # We generate a version of the queries with stop gradient to use as input
      # to the topk similarity layer. We actually do want gradient to flow to
      # the queries, but backward differentiation over the topk layer yields
      # inefficient HLO ops. Instead we use queries with gradient to recompute
      # attention scores later.
      queries_sg = jax.lax.stop_gradient(queries)

      # Perform top-k similarity search over queries, yielding
      #   top_values: (queries, k_top, memory_dim)
      #   top_ids: (queries, k_top)
      top_keys, _, top_ids = self.topk_similarity(queries_sg, memory_keys)

      top_ids = top_ids.reshape(n_queries, self.k_top)
      top_values = memory_values[top_ids]
      loss_helpers['top_entity_ids'] = memory_entity_ids[top_ids]

      # We re-compute top scores using the queries with gradient (wg) to make
      # sure the query projector and the rest of the model receives gradient.
      top_scores_wg = jnp.einsum('qd,qkd->qk', queries, top_keys)
      top_mask = memory_mask[top_ids]
      top_scores_wg = top_scores_wg - (1 - top_mask) * _LARGE_NUMBER

      # We perform dot product attention using retrieved memory vectors as key,
      # dense projection of retrieved vectors as value and value and mention
      # representations as query.
      attention_weights = nn.softmax(top_scores_wg, axis=-1)
      loss_helpers['memory_attention_weights'] = attention_weights
    encoding = self.update_layer(
        encoded_input=encoding,
        retrieval_values=top_values,
        retrieval_scores=attention_weights,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        deterministic=deterministic,
    )

    return encoding, loss_helpers, logging_helpers

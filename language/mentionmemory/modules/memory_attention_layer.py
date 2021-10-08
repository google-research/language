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
"""Contains memory attention layer."""

from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.modules import memory_retrieval_layer
from language.mentionmemory.modules import retrieval_update_layers
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import
import ml_collections


def _assert_array_is_integer_or_none(array: Optional[Array], array_name: str):
  if array is not None and array.dtype.kind != 'i':
    raise ValueError('Array %s must be integer (currently, %s). '
                     'Otherwise, there might be data corruption.' %
                     (array_name, str(array.dtype)))


class MemoryAttentionLayer(nn.Module):
  """Performs attention update over memory table for passage mentions.

  Given an input sequence representation and passage mention positions, this
  layer applies single-head dot-product top-k attention over a memory table,
  generating mention representations from input representation to use as
  queries and either the values from the memory table or a separate table of
  keys as keys. The attention is followed by a layer norm.

  This layer is designed to be used for memory tables that are too large to fit
  on a single device, and are therefore sharded over multiple devices. The layer
  gathers queries from all devices, performs top-(k/devices) attention over the
  local memory shard, and then distributes results back to each device using
  collective operations. The layer can only be run in a parallel setting using
  pmap with 'batch' as the mapped dimension.

  Attributes:
    memory_key_dim: dimensionality of memory keys.
    input_dim: dimensionality of input token representations.
    memory_update_type: means by which retrieved memory vectors are incorporated
      into input representation, such as simple addition or concatenation + MLP.
    memory_update_config: hyperparameters for the update layer, beyond input
      dimension and datatype.
    k_top_device: top-k retrieved memory vectors per device.
    splits: Governs a tradeoff between speed and memory usage in topk similarity
      search layer and has no effect on actual search results. A higher number
      of splits is slower but uses less memory.
    dtype: precision of computation.
    layer_norm_epsilon: epsilon of layer norm.
    k_top_post_selection: select top-k memories after retrieving `k_top_device`
      top memories from every device.
  """

  memory_key_dim: int
  input_dim: int
  memory_update_type: str
  memory_update_config: ml_collections.FrozenConfigDict
  k_top_device: int
  splits: int
  dtype: Dtype
  layer_norm_epsilon: float = 1e-12
  k_top_post_selection: Optional[int] = None

  def setup(self):
    self.query_projector = nn.Dense(
        features=self.memory_key_dim,
        dtype=self.dtype,
    )

    self.memory_retrieval_layer = memory_retrieval_layer.MemoryRetrievalLayer(
        k_top_device=self.k_top_device,
        splits=self.splits,
        k_top_post_selection=self.k_top_post_selection,
    )

    self.update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        self.memory_update_type](
            input_dim=self.input_dim,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
            **self.memory_update_config)

  def __call__(
      self,
      encoded_input: Array,
      mention_batch_positions: Array,
      mention_start_positions: Array,
      mention_end_positions: Array,
      mention_mask: Array,
      memory_keys: Array,
      memory_identifiers: Array,
      memory_entity_ids: Array,
      deterministic: bool,
      memory_values: Optional[Array] = None,
      text_identifiers: Optional[Array] = None,
      memory_text_entities: Optional[Array] = None,
      same_passage_memory_policy: str = 'disallow',
  ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """Perform attention update over memory table.

    Args:
      encoded_input: [batch_size, n_tokens, hidden_size] input representation.
      mention_batch_positions: [n_mentions] mention sample position in batch.
      mention_start_positions: [n_mentions] mention start position in input.
      mention_end_positions: [n_mentions] mention end position in input.
      mention_mask: [n_mentions] attention mask to prevent updates from padding.
      memory_keys: [rows, values per row, key_dim] mention memory keys. The
        number of rows in the memory table governs the recall vs speed of the
        topk similarity search. Search is performed by taking max over each row,
        and then top-k between rows. Distributing the same values over more rows
        leads to higher recall but slower search.
      memory_identifiers: [memory_size] identifier for memory vectors.
      memory_entity_ids: [memory_size] entity ids for mentions in memory table
      deterministic: don't apply dropout if true.
      memory_values: [values, memory_dim] if separate keys and values.
      text_identifiers: [n_mentions] search will not retrieve memory vectors
        with the same identifier as passage mention.
      memory_text_entities: [n_mentions, n_memory_text_entities] entity ids for
        passages where memories are coming from.
      same_passage_memory_policy: how to treat mentions from the same passage.
        Possible options: `allow`, `disallow` and `only`.

    Returns:
      Updated input, loss and logging helper dicts.
    """
    _assert_array_is_integer_or_none(mention_batch_positions,
                                     'mention_batch_positions')
    _assert_array_is_integer_or_none(mention_start_positions,
                                     'mention_start_positions')
    _assert_array_is_integer_or_none(mention_end_positions,
                                     'mention_end_positions')
    _assert_array_is_integer_or_none(memory_entity_ids, 'memory_entity_ids')
    _assert_array_is_integer_or_none(memory_identifiers, 'memory_identifiers')
    _assert_array_is_integer_or_none(memory_text_entities,
                                     'memory_text_entities')
    _assert_array_is_integer_or_none(text_identifiers, 'text_identifiers')

    loss_helpers, logging_helpers = {}, {}

    # We generate mention representations to use as queries for similarity
    # search by concatenating start and end tokens for each mention and
    # projecting the concatenation with a dense layer.
    mention_start_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_start_positions))
    mention_end_encodings = jut.matmul_2d_index_select(
        encoded_input, (mention_batch_positions, mention_end_positions))

    queries = self.query_projector(
        jnp.concatenate((mention_start_encodings, mention_end_encodings),
                        axis=-1))

    loss_helpers['memory_attention_mention_encodings'] = queries

    retrieval_result = self.memory_retrieval_layer(
        queries=queries,
        memory_keys=memory_keys,
        memory_identifiers=memory_identifiers,
        memory_entity_ids=memory_entity_ids,
        memory_values=memory_values,
        text_identifiers=text_identifiers,
        memory_text_entities=memory_text_entities,
        same_passage_memory_policy=same_passage_memory_policy,
    )

    # Most of the information from retrieval_result goes to `loss_helpers`
    # except `n_disallowed`. In future, we might join these two into a single
    # dictionary.
    loss_helpers.update(retrieval_result)
    if 'n_disallowed' in retrieval_result:
      logging_helpers['n_disallowed'] = retrieval_result['n_disallowed']

    encoded_input = self.update_layer(
        encoded_input=encoded_input,
        retrieval_values=retrieval_result['top_values'],
        retrieval_scores=retrieval_result['memory_attention_weights'],
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        deterministic=deterministic,
    )

    return encoded_input, loss_helpers, logging_helpers

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



import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import retrieval_update_layers
from language.mentionmemory.modules import topk_similarity_layer
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import
import ml_collections

_LARGE_NUMBER = 1e10


def _assert_array_is_integer_or_none(array, array_name):
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

    self.topk_similarity = topk_similarity_layer.TopKSimilarityLayer(
        k_top=self.k_top_device,
        splits=self.splits,
    )

    self.update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        self.memory_update_type](
            input_dim=self.input_dim,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
            **self.memory_update_config)

  def __call__(
      self,
      encoded_input,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      mention_mask,
      memory_keys,
      memory_identifiers,
      memory_entity_ids,
      deterministic,
      memory_values = None,
      text_identifiers = None,
      memory_text_entities = None,
      same_passage_memory_policy = 'disallow',
  ):
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
    memory_size = memory_keys.shape[0] * memory_keys.shape[1]

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

    n_queries = queries.shape[0]

    # We generate a version of the queries with stop gradient to use as input to
    # the topk similarity layer. We actually do want gradient to flow to the
    # queries, but backward differentiation over the topk layer yields
    # inefficient HLO ops. Instead we use queries with gradient to recompute
    # attention scores later.
    queries_sg = jax.lax.stop_gradient(queries)

    # Gather queries from all devices. Each device contains a shard of the
    # mention memory. Ultimately we want to perform search over the entire
    # mention memory, so we gather mentions from all devices, apply similarity
    # search over the local shard, then distribute the results back.
    gathered_queries = jax.lax.all_gather(queries_sg, 'batch')
    if text_identifiers is not None:
      gathered_identifiers = jax.lax.all_gather(text_identifiers, 'batch')

    n_devices = gathered_queries.shape[0]
    gathered_queries = gathered_queries.reshape(n_devices * n_queries,
                                                self.memory_key_dim)

    # Perform top-k similarity search over queries, yielding
    # top_values: (n_devices * queries_per_device, k_top_device, memory_key_dim)
    # top_ids: (n_devices * queries_per_device, k_top_device)
    top_keys, top_scores, top_ids = self.topk_similarity(
        gathered_queries, memory_keys)

    if memory_values is not None:
      top_values = memory_values[top_ids]
    else:
      top_values = top_keys
    memory_dim = top_values.shape[-1]

    # Also return entity ids
    top_entity_ids = memory_entity_ids[top_ids]

    top_values = top_values.reshape(n_devices, n_queries, self.k_top_device,
                                    memory_dim)
    top_entity_ids = top_entity_ids.reshape(n_devices, n_queries,
                                            self.k_top_device)
    global_top_ids = top_ids.reshape(n_devices, n_queries, self.k_top_device)

    # Now that we have searched the local shard using queries from all devices,
    # we need to distribute the search results back to all devices. Applying
    # pswapaxes followed by swapaxes makes us go from
    # (devices, queries per device, local shard retrievals) to
    # (local queries, devices, memory retrievals per device).
    (top_values, top_entity_ids, global_top_ids) = jax.lax.pswapaxes(
        (top_values, top_entity_ids, global_top_ids), axis_name='batch', axis=0)

    top_values = jnp.swapaxes(top_values, 0, 1)
    top_entity_ids = jnp.swapaxes(top_entity_ids, 0, 1)

    # (local queries, devices, memory retrievals per device).
    global_top_ids = jnp.swapaxes(global_top_ids, 0, 1)
    # IDs are device specific. Therefore, we need to convert them to `global`
    # memory IDs. Note that every devices operates on a memory of the same size.
    # Therefore, IDs on the device 0 don't need to be changed, we need to add
    # `memory_size` to IDs from the device 1, 2 * `memory_size` to IDs from the
    # device 2, etc.
    global_top_ids = global_top_ids + jnp.arange(n_devices).reshape(
        1, -1, 1) * memory_size

    # Reshape results to (local_queries, global retrievals).
    k_top = n_devices * self.k_top_device
    top_values = top_values.reshape(n_queries, k_top, memory_dim)
    top_entity_ids = top_entity_ids.reshape(n_queries, k_top)
    global_top_ids = global_top_ids.reshape(n_queries, k_top)

    # At this point, we have selected `k_top = n_devices * self.k_top_device`
    # memories for every query. The selection process is approximate since
    # we retrieve `self.k_top_device` memories from every device and then
    # just concatenate the results.
    # Due to computational constraints we may wish to limit the number
    # of memories per query, so we subselect even further and keep only
    # `self.k_top_post_selection` retrieved memories for every query.
    if self.k_top_post_selection is not None:
      top_scores = top_scores.reshape(n_devices, n_queries, self.k_top_device)
      top_scores = jax.lax.pswapaxes(top_scores, axis_name='batch', axis=0)
      top_scores = jnp.swapaxes(top_scores, 0, 1)
      top_scores = top_scores.reshape(n_queries, k_top)
      # Take k highest scores among all rows.
      # pylint:disable=invalid-unary-operand-type
      top_post_selection_index = jnp.argsort(
          top_scores, axis=-1)[:, :-self.k_top_post_selection - 1:-1]
      # pylint:enable=invalid-unary-operand-type
      top_values = jut.matmul_slice(top_values, top_post_selection_index)
      top_entity_ids = jut.matmul_slice(top_entity_ids,
                                        top_post_selection_index)
      global_top_ids = jut.matmul_slice(global_top_ids,
                                        top_post_selection_index)

    # If we use separate memory values, distribute keys back also.
    if memory_values is not None:
      top_keys = top_keys.reshape(n_devices, n_queries, self.k_top_device,
                                  self.memory_key_dim)
      top_keys = jax.lax.pswapaxes(top_keys, axis_name='batch', axis=0)
      top_keys = jnp.swapaxes(top_keys, 0, 1)
      top_keys = top_keys.reshape(n_queries, k_top, self.memory_key_dim)
      if self.k_top_post_selection is not None:
        top_keys = jut.matmul_slice(top_keys, top_post_selection_index)
    else:
      top_keys = top_values

    loss_helpers['top_entity_ids'] = top_entity_ids
    loss_helpers['top_memory_ids'] = global_top_ids

    # We re-compute top scores using the queries with gradient (wg) to make sure
    # the mention encoder and the rest of the model receives gradient
    top_scores_wg = jnp.einsum('qd,qkd->qk', queries, top_keys)

    loss_helpers['memory_attention_scores_with_disallowed'] = top_scores_wg

    # We want to disallow some mentions from being retrieved (i.e. from same
    # passage during pre-training). Here we mask retrieved mentions which have
    # the same identifier as the query.
    if text_identifiers is not None:
      top_ids = top_ids.reshape(n_devices, n_queries, self.k_top_device)
      gathered_identifiers = gathered_identifiers.reshape(
          n_devices, n_queries, 1)
      identifier_mask = (memory_identifiers[top_ids] == gathered_identifiers)

      # We manually cast `identifier_mask` into int32. Otherwise, `pswapaxes`
      # which is known to have undefined behaviour on CPU, "corrupts" a vector
      # making it effectively int32, while keeping boolean dtype. This in turn
      # leads to a compilation error for the einsum operation in the
      # `matmul_slice` (types mismatch).
      identifier_mask = identifier_mask.astype(dtype=jnp.int32)
      identifier_mask = jax.lax.pswapaxes(
          identifier_mask, axis_name='batch', axis=0)
      identifier_mask = jnp.swapaxes(identifier_mask, 0, 1)
      identifier_mask = identifier_mask.reshape(n_queries, k_top)
      if self.k_top_post_selection is not None:
        identifier_mask = jut.matmul_slice(identifier_mask,
                                           top_post_selection_index)
      loss_helpers['memory_attention_disallowed_mask'] = identifier_mask.astype(
          jnp.bool_)
      identifier_mask = identifier_mask.astype(top_scores_wg.dtype)

      # Depending on `same_passage_memory_policy` we treat memories from the
      # same passage as query mentions differently.
      if same_passage_memory_policy == 'disallow':
        top_scores_wg = top_scores_wg - identifier_mask * _LARGE_NUMBER
      elif same_passage_memory_policy == 'only':
        top_scores_wg = top_scores_wg - (1.0 - identifier_mask) * _LARGE_NUMBER
      elif same_passage_memory_policy == 'allow':
        pass
      else:
        raise ValueError('Unknown value for `same_passage_memory_policy: %s' %
                         same_passage_memory_policy)
      n_disallowed = identifier_mask.sum()
      logging_helpers['n_disallowed'] = n_disallowed

    if memory_text_entities is not None:
      top_ids = top_ids.reshape(n_devices, n_queries, self.k_top_device)
      # shape [n_devices, n_queries, k_top_device, n_text_entities_per_passage]
      top_text_entities = memory_text_entities[top_ids]
      top_text_entities = jax.lax.pswapaxes(
          top_text_entities, axis_name='batch', axis=0)
      # shape [n_queries, n_devices, k_top_device, n_text_entities_per_passage]
      top_text_entities = jnp.swapaxes(top_text_entities, 0, 1)
      # shape [n_queries, n_devices * k_top_device, n_text_entities_per_passage]
      top_text_entities = top_text_entities.reshape(n_queries, k_top, -1)
      if self.k_top_post_selection is not None:
        top_text_entities = jut.matmul_slice(top_text_entities,
                                             top_post_selection_index)
      loss_helpers['memory_top_text_entities'] = top_text_entities

    # We perform dot product attention using retrieved memory vectors as key,
    # dense projection of retrieved vectors as value and value and mention
    # representations as query.
    attention_weights = nn.softmax(top_scores_wg, axis=-1)
    loss_helpers['memory_attention_weights'] = attention_weights
    encoded_input = self.update_layer(
        encoded_input=encoded_input,
        retrieval_values=top_values,
        retrieval_scores=attention_weights,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        deterministic=deterministic,
    )

    return encoded_input, loss_helpers, logging_helpers

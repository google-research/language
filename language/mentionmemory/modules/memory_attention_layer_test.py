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
"""Tests for memory attention layer."""

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import memory_attention_layer
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import scipy.special

_LARGE_NUMBER = 1e10


def _gen_compare_retrievals_with_numpy_list():
  top_k_max_text_identifiers_list = [
      # First, we experiment without max_text_identifiers
      (None, None),
      (3, None),
      (5, None),
      # Second, we experiment with very small max_text_identifiers
      # which should significantly increase the num_disallowed
      (None, 2),
      (3, 2),
      (5, 2),
      # Finally, we experiment with medium max_text_identifiers
      (None, 10),
      (3, 10),
      (5, 10),
  ]
  same_passage_memory_policy_list = ['disallow', 'allow', 'only']

  test_list = itertools.product(top_k_max_text_identifiers_list,
                                same_passage_memory_policy_list)
  test_list = [
      tuple([index] + list(x) + [y]) for index, (x, y) in enumerate(test_list)
  ]
  return test_list


class MemoryAttentionLayerTest(parameterized.TestCase):
  """Memory attention layer test."""

  dtype = jnp.float32
  memory_key_dim = 4
  input_dim = 8
  memory_update_type = 'additive'
  table_size = 128
  rows = 4
  splits = 2
  k_top_device = 2
  k_top_post_selection = 3
  seq_len = 20
  bsz = 2
  n_devices = 4
  n_mentions = 5
  entity_vocab_size = 10

  memory_update_config = ml_collections.FrozenConfigDict({})

  @parameterized.parameters(
      (False),
      (True),
  )
  def test_mention_memory_layer(self, separate_memory_values):
    """Testing memory attention layer."""

    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    model = memory_attention_layer.MemoryAttentionLayer(
        memory_key_dim=self.memory_key_dim,
        input_dim=self.input_dim,
        memory_update_type=self.memory_update_type,
        memory_update_config=self.memory_update_config,
        k_top_device=self.k_top_device,
        k_top_post_selection=self.k_top_post_selection,
        splits=self.splits,
        dtype=self.dtype)

    static_argnums = (9) if separate_memory_values else (9, 10)
    pinit_with_output = jax.pmap(
        model.init_with_output,
        axis_name='batch',
        static_broadcasted_argnums=static_argnums)

    rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(rng, self.n_devices)
    encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.input_dim), dtype=self.dtype)
    encoded_input = jax.device_put_replicated(encoded_input, devices)

    mention_batch_positions = jnp.tile(
        jnp.arange(self.bsz).reshape(-1, 1), (1, 3)).reshape(-1)
    mention_batch_positions = jax.device_put_replicated(mention_batch_positions,
                                                        devices)

    mention_start_positions = jnp.tile(jnp.asarray([0, 5, 10]), (self.bsz))
    mention_start_positions = jax.device_put_replicated(mention_start_positions,
                                                        devices)

    mention_end_positions = jnp.tile(jnp.asarray([2, 7, 12]), (self.bsz))
    mention_end_positions = jax.device_put_replicated(mention_end_positions,
                                                      devices)

    n_mentions = mention_start_positions.shape[-1]

    mention_mask = jnp.tile(jnp.asarray([1, 1, 1]), (self.bsz))
    mention_mask = jax.device_put_replicated(mention_mask, devices)

    memory_table = np.ones(
        (self.n_devices * self.table_size, self.memory_key_dim),
        dtype=self.dtype)
    # Make sure id 0 or 1 will be highest scoring
    memory_table[0] = memory_table[0] * 2.0
    memory_table[1] = memory_table[1] * -2.0
    memory_table = jnp.asarray(memory_table, dtype=self.dtype)

    memory_keys = memory_table.reshape(self.n_devices, self.rows,
                                       self.table_size // self.rows,
                                       self.memory_key_dim)

    memory_keys_sharded = jax.device_put_sharded(list(memory_keys), devices)
    if separate_memory_values:
      memory_values = memory_table.reshape(self.n_devices, self.table_size,
                                           self.memory_key_dim)
      memory_values = jax.device_put_sharded(list(memory_values), devices)
    else:
      memory_values = None

    memory_entity_ids = np.arange(self.n_devices * self.table_size).reshape(
        self.n_devices, self.table_size)
    memory_entity_ids = jax.device_put_sharded(list(memory_entity_ids), devices)

    # Use entity id as identifier here
    memory_identifiers = memory_entity_ids

    (encoded_output, loss_helpers, _), _ = pinit_with_output(
        split_rng,
        encoded_input,
        mention_batch_positions,
        mention_start_positions,
        mention_end_positions,
        mention_mask,
        memory_keys_sharded,
        memory_identifiers,
        memory_entity_ids,
        True,  # deterministic
        memory_values,
        text_identifiers=None,
    )

    attention_weights = loss_helpers['memory_attention_weights']
    entity_ids = loss_helpers['top_entity_ids']

    normed_input = encoded_input - 1.0

    # Check input was changed
    self.assertFalse(jnp.allclose(encoded_output, normed_input))

    # Check input was not changed where it should not be
    all_indices = set(
        itertools.product(np.arange(self.bsz), np.arange(self.seq_len)))
    # Note that mention positions is the same across all of the devices
    start_indices = set(
        zip(mention_batch_positions[0].tolist(),
            mention_start_positions[0].tolist()))
    non_start_indices = all_indices.difference(start_indices)
    non_start_indices_1, non_start_indices_2 = zip(*non_start_indices)
    non_start_indices_1 = jnp.asarray(non_start_indices_1)
    non_start_indices_2 = jnp.asarray(non_start_indices_2)

    non_start_outputs = encoded_output[:, non_start_indices_1,
                                       non_start_indices_2]
    non_start_inputs = normed_input[:, non_start_indices_1, non_start_indices_2]
    self.assertTrue(jnp.allclose(non_start_outputs, non_start_inputs))

    # Check shapes as expected
    self.assertSequenceEqual(
        encoded_output.shape,
        (self.n_devices, self.bsz, self.seq_len, self.input_dim))

    self.assertSequenceEqual(
        attention_weights.shape,
        (self.n_devices, n_mentions, self.k_top_post_selection))

    self.assertSequenceEqual(
        entity_ids.shape,
        (self.n_devices, n_mentions, self.k_top_post_selection))

    # Check id 0 or 1 retrieved
    self.assertTrue(
        jnp.all((entity_ids[..., 0] == 0) + (entity_ids[..., 0] == 1)))

    # Set some text identifiers to 0 and others to 1 so that some are binding
    text_identifiers = np.zeros((n_mentions), dtype=np.int32)
    text_identifiers[:n_mentions // 2] = 1
    text_identifiers = jax.device_put_replicated(text_identifiers, devices)

    # Initialize and run one forward pass of model
    (_, loss_helpers, logging_helpers), _ = pinit_with_output(
        split_rng,
        encoded_input,
        mention_batch_positions,
        mention_start_positions,
        mention_end_positions,
        mention_mask,
        memory_keys_sharded,
        memory_identifiers,
        memory_entity_ids,
        True,  # deterministic
        memory_values,  # memory_values
        text_identifiers=text_identifiers,
    )
    attention_weights_wid = loss_helpers['memory_attention_weights']
    entity_ids_wid = loss_helpers['top_entity_ids']
    n_disallowed = logging_helpers['n_disallowed'][0]

    # Check no effect on ids
    self.assertTrue(jnp.all(entity_ids == entity_ids_wid))

    # Check id 0 or 1 have 0 scores
    text_identifiers = jnp.expand_dims(text_identifiers, -1)
    score_masked = (text_identifiers == entity_ids_wid) * attention_weights_wid
    self.assertAlmostEqual(score_masked.sum(), 0.0)

    # Check number disallowed as expected
    self.assertEqual(n_disallowed, n_mentions // 2)

  def test_memory_attention_backward(self):
    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    model = memory_attention_layer.MemoryAttentionLayer(
        memory_key_dim=self.memory_key_dim,
        input_dim=self.input_dim,
        memory_update_type=self.memory_update_type,
        memory_update_config=self.memory_update_config,
        k_top_device=self.k_top_device,
        k_top_post_selection=self.k_top_post_selection,
        splits=self.splits,
        dtype=self.dtype)

    pinit = jax.pmap(
        model.init, axis_name='batch', static_broadcasted_argnums=(9, 10))

    rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(rng, self.n_devices)
    encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.input_dim), dtype=self.dtype)
    encoded_input = jax.device_put_replicated(encoded_input, devices)

    mention_batch_positions = jnp.tile(
        jnp.asarray([[0], [1], [2]]), (1, self.bsz)).reshape(-1)
    mention_batch_positions = jax.device_put_replicated(mention_batch_positions,
                                                        devices)

    mention_start_positions = jnp.tile(jnp.asarray([0, 5, 10]), (self.bsz))
    mention_start_positions = jax.device_put_replicated(mention_start_positions,
                                                        devices)

    mention_end_positions = jnp.tile(jnp.asarray([2, 7, 12]), (self.bsz))
    mention_end_positions = jax.device_put_replicated(mention_end_positions,
                                                      devices)

    mention_mask = jnp.tile(jnp.asarray([1, 1, 1]), (self.bsz))
    mention_mask = jax.device_put_replicated(mention_mask, devices)

    memory_table = np.ones(
        (self.n_devices * self.table_size, self.memory_key_dim),
        dtype=self.dtype)

    memory_table = jnp.asarray(memory_table, dtype=self.dtype)
    memory_table = memory_table.reshape(self.n_devices, self.rows,
                                        self.table_size // self.rows,
                                        self.memory_key_dim)
    memory_table_sharded = jax.device_put_sharded(list(memory_table), devices)

    memory_entity_ids = np.arange(self.n_devices * self.table_size).reshape(
        self.n_devices, self.table_size)
    memory_entity_ids = jax.device_put_sharded(list(memory_entity_ids), devices)

    # Use entity id as identifier here
    memory_identifiers = memory_entity_ids

    initial_parameters = pinit(
        split_rng,
        encoded_input,
        mention_batch_positions,
        mention_start_positions,
        mention_end_positions,
        mention_mask,
        memory_table_sharded,
        memory_identifiers,
        memory_entity_ids,
        True,  # deterministic
        None,  # memory_values
        text_identifiers=None,
    )

    def step_fn(
        params,
        encoded_input,
        mention_batch_positions,
        mention_start_positions,
        mention_end_positions,
        mention_mask,
        memory_keys,
        memory_identifiers,
        memory_entity_ids,
    ):

      def loss_fn(params):
        encoded_output, _, _ = model.apply(
            {'params': params},
            rngs=None,
            encoded_input=encoded_input,
            mention_batch_positions=mention_batch_positions,
            mention_start_positions=mention_start_positions,
            mention_end_positions=mention_end_positions,
            mention_mask=mention_mask,
            memory_keys=memory_keys,
            memory_identifiers=memory_identifiers,
            memory_entity_ids=memory_entity_ids,
            deterministic=True,
            text_identifiers=None,
        )
        return encoded_output.sum()

      loss, grad = jax.value_and_grad(loss_fn)(params)
      return loss, grad

    pstep = jax.pmap(step_fn, axis_name='batch')

    _ = pstep(
        initial_parameters['params'],
        encoded_input=encoded_input,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        memory_keys=memory_table_sharded,
        memory_identifiers=memory_identifiers,
        memory_entity_ids=memory_entity_ids,
    )

  @parameterized.parameters(_gen_compare_retrievals_with_numpy_list())
  def test_compare_retrievals_with_numpy(self, seed, k_top_post_selection,
                                         max_text_identifiers,
                                         same_passage_memory_policy):
    """Test whether retrieval results are correct."""
    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()
    n_text_entities_per_memory = 3

    model = memory_attention_layer.MemoryAttentionLayer(
        memory_key_dim=self.memory_key_dim,
        input_dim=self.input_dim,
        memory_update_type=self.memory_update_type,
        memory_update_config=self.memory_update_config,
        k_top_device=self.k_top_device,
        k_top_post_selection=k_top_post_selection,
        splits=self.splits,
        dtype=self.dtype)
    pinit_with_output = jax.pmap(
        model.init_with_output,
        axis_name='batch',
        static_broadcasted_argnums=(9, 10, 13))

    rng = jax.random.PRNGKey(seed)
    split_rng = jax.random.split(rng, self.n_devices)
    encoded_input = jax.random.uniform(
        rng,
        shape=(self.n_devices, self.bsz, self.seq_len, self.input_dim),
        dtype=self.dtype)
    mention_batch_positions = jax.random.randint(
        rng, minval=0, maxval=self.bsz, shape=(self.n_devices, self.n_mentions))
    mention_start_positions = jax.random.randint(
        rng,
        minval=0,
        maxval=self.seq_len,
        shape=(self.n_devices, self.n_mentions))
    mention_end_positions = mention_start_positions
    mention_mask = jnp.ones(shape=(self.n_devices, self.n_mentions))

    memory_table = jax.random.uniform(
        rng,
        shape=(self.n_devices, self.rows, self.table_size // self.rows,
               self.memory_key_dim))
    memory_entity_ids = jax.random.randint(
        rng,
        minval=0,
        maxval=self.entity_vocab_size,
        shape=(self.n_devices, self.table_size))
    if max_text_identifiers is not None:
      memory_identifiers = jax.random.randint(
          rng,
          minval=0,
          maxval=max_text_identifiers,
          shape=(self.n_devices, self.table_size))
      text_identifiers = jax.random.randint(
          rng,
          minval=0,
          maxval=max_text_identifiers,
          shape=(self.n_devices, self.n_mentions))
    else:
      text_identifiers = None

    if n_text_entities_per_memory is not None:
      memory_text_entities = jax.random.randint(
          rng,
          minval=0,
          maxval=self.entity_vocab_size,
          shape=(self.n_devices, self.table_size, n_text_entities_per_memory))
    else:
      memory_text_entities = None

    encoded_input_sharded = jax.device_put_sharded(list(encoded_input), devices)
    mention_batch_positions_sharded = jax.device_put_sharded(
        list(mention_batch_positions), devices)
    mention_start_positions_sharded = jax.device_put_sharded(
        list(mention_start_positions), devices)
    mention_end_positions_sharded = jax.device_put_sharded(
        list(mention_end_positions), devices)
    mention_mask_sharded = jax.device_put_sharded(list(mention_mask), devices)
    memory_table_sharded = jax.device_put_sharded(list(memory_table), devices)
    memory_entity_ids_sharded = jax.device_put_sharded(
        list(memory_entity_ids), devices)
    if max_text_identifiers is not None:
      memory_identifiers_sharded = jax.device_put_sharded(
          list(memory_identifiers), devices)
      text_identifiers_sharded = jax.device_put_sharded(
          list(text_identifiers), devices)
    else:
      memory_identifiers_sharded = None
      text_identifiers_sharded = None

    if memory_text_entities is not None:
      memory_text_entities_sharded = jax.device_put_sharded(
          list(memory_text_entities), devices)
    else:
      memory_text_entities_sharded = None

    memory_ids = jnp.arange(self.n_devices * self.table_size)
    memory_ids = memory_ids.reshape(self.n_devices, self.table_size)

    (_, loss_helpers, logging_helpers), params = pinit_with_output(
        split_rng,
        encoded_input_sharded,
        mention_batch_positions_sharded,
        mention_start_positions_sharded,
        mention_end_positions_sharded,
        mention_mask_sharded,
        memory_table_sharded,
        memory_identifiers_sharded,
        memory_entity_ids_sharded,
        True,
        None,  # memory_values
        text_identifiers_sharded,
        memory_text_entities_sharded,
        same_passage_memory_policy,
    )

    params = params.unfreeze()['params']

    mention_encodings = []
    for device_id in range(self.n_devices):
      mention_start_encodings = encoded_input[device_id][
          mention_batch_positions[device_id],
          mention_start_positions[device_id]]
      mention_end_encodings = encoded_input[device_id][
          mention_batch_positions[device_id], mention_end_positions[device_id]]
      mention_encodings_on_device = jnp.concatenate(
          [mention_start_encodings, mention_end_encodings], axis=-1)
      mention_encodings_on_device = np.matmul(
          mention_encodings_on_device,
          params['query_projector']['kernel'][device_id])
      mention_encodings_on_device += params['query_projector']['bias'][
          device_id]
      mention_encodings.append(mention_encodings_on_device)

    # [n_devices, n_mentions, memory_key_dim]
    mention_encodings_stacked = jnp.stack(mention_encodings)
    mention_encodings_stacked = mention_encodings_stacked.reshape(
        [self.n_devices * self.n_mentions, self.memory_key_dim])

    # Object which represents a single retrieval result with additional info.
    RetrievedMemory = collections.namedtuple('RetrievedMemory', [
        'device', 'row', 'rowwise_index', 'devicewise_index', 'global_index',
        'score', 'memory', 'entity_id', 'memory_hash',
        'memory_passage_text_entities'
    ])

    num_disallowed_per_device = [0 for _ in range(self.n_devices)]
    # Manually simulate retrieval per every query
    for query_id in range(self.n_devices * self.n_mentions):
      query = mention_encodings_stacked[query_id]
      top_memories_query = []
      # Collect retirevals for a single query on each devices separately
      for device_id in range(self.n_devices):
        top_memories_per_device = []
        for row_id in range(self.rows):
          scores = np.einsum('mh,h->m', memory_table[device_id, row_id], query)
          top_index = np.argmax(scores)
          devicewise_index = row_id * (self.table_size // self.rows) + top_index
          global_index = memory_ids[device_id, devicewise_index]
          self.assertEqual(global_index,
                           devicewise_index + device_id * self.table_size)
          if max_text_identifiers is not None:
            memory_hash = memory_identifiers[device_id, devicewise_index].item()
          else:
            memory_hash = None
          if memory_text_entities is not None:
            memory_passage_text_entities = memory_text_entities[
                device_id, devicewise_index]
          else:
            memory_passage_text_entities = None
          top_memories_per_device.append(
              RetrievedMemory(
                  device=device_id,
                  row=row_id,
                  rowwise_index=top_index,
                  devicewise_index=devicewise_index,
                  global_index=global_index,
                  score=scores[top_index].item(),
                  memory=memory_table[device_id, row_id, top_index],
                  entity_id=memory_entity_ids[device_id,
                                              devicewise_index].item(),
                  memory_hash=memory_hash,
                  memory_passage_text_entities=memory_passage_text_entities,
              ))
        # Sort by score. In case two scores are equal (likely because both
        # were considered "disallowed" we compare by entity IDs.
        top_memories_per_device.sort(
            key=lambda x: (x.score, x.entity_id), reverse=True)
        top_memories_per_device = top_memories_per_device[:self.k_top_device]
        top_memories_query.extend(top_memories_per_device)

      top_memories_query.sort(
          key=lambda x: (x.score, x.entity_id), reverse=True)
      if k_top_post_selection is not None:
        top_memories_query = top_memories_query[:k_top_post_selection]

      if max_text_identifiers is not None:
        num_current_disallowed = 0
        text_id = text_identifiers[query_id // self.n_mentions,
                                   query_id % self.n_mentions].item()
        for i in range(len(top_memories_query)):
          if top_memories_query[i].memory_hash == text_id:
            num_current_disallowed += 1

          if ((same_passage_memory_policy == 'disallow' and
               top_memories_query[i].memory_hash == text_id) or
              (same_passage_memory_policy == 'only' and
               top_memories_query[i].memory_hash != text_id)):
            top_memories_query[i] = top_memories_query[i]._replace(
                score=-_LARGE_NUMBER)
        num_disallowed_per_device[query_id //
                                  self.n_mentions] += num_current_disallowed
        top_memories_query.sort(
            key=lambda x: (x.score, x.global_index), reverse=True)

      actual_entity_ids = loss_helpers['top_entity_ids'][query_id //
                                                         self.n_mentions,
                                                         query_id %
                                                         self.n_mentions]
      actual_memory_ids = loss_helpers['top_memory_ids'][query_id //
                                                         self.n_mentions,
                                                         query_id %
                                                         self.n_mentions]
      actual_attention_weights = loss_helpers['memory_attention_weights'][
          query_id // self.n_mentions, query_id % self.n_mentions]

      # We sort retrieved results first by the attention score and then
      # by memory ID.
      p = list(range(len(actual_attention_weights)))
      # pylint: disable=cell-var-from-loop
      p.sort(
          key=lambda i: (actual_attention_weights[i], actual_memory_ids[i]),
          reverse=True)
      # pylint: enable=cell-var-from-loop
      p = np.array(p)

      actual_entity_ids = list(actual_entity_ids[p])
      actual_attention_weights = list(actual_attention_weights[p])
      actual_memory_ids = list(actual_memory_ids[p])

      expected_entity_ids = [x.entity_id for x in top_memories_query]
      self.assertSequenceEqual(expected_entity_ids, actual_entity_ids)

      expected_attention_weights = scipy.special.softmax(
          [x.score for x in top_memories_query])
      self.assertSequenceAlmostEqual(expected_attention_weights,
                                     actual_attention_weights, 5)

      expected_memory_ids = [x.global_index for x in top_memories_query]
      self.assertSequenceEqual(expected_memory_ids, actual_memory_ids)

      actual_top_text_entities = loss_helpers['memory_top_text_entities'][
          query_id // self.n_mentions, query_id % self.n_mentions]
      actual_top_text_entities = actual_top_text_entities[p]

      expected_top_text_entities = [
          x.memory_passage_text_entities for x in top_memories_query
      ]
      self.assertEqual(
          len(actual_top_text_entities), len(expected_top_text_entities))

      # Comparing `actual_top_text_entities` and `expected_top_text_entities`
      # directly is troublesome since we cannot gurantee the order for
      # retrieval results with the same attention weights. Therefore, we first
      # sort both `actual_top_text_entities` and `expected_attention_weights`
      # by the attention weight first and then by their elements.
      def sort_entities(top_text_entities):
        result = [(expected_attention_weights[i], list(top_text_entities[i]))
                  for i in range(len(top_text_entities))]
        result.sort()
        return [x[1] for x in result]

      actual_top_text_entities = sort_entities(actual_top_text_entities)
      expected_top_text_entities = sort_entities(expected_top_text_entities)

      for i in range(len(actual_top_text_entities)):
        self.assertSequenceEqual(
            list(actual_top_text_entities[i]),
            list(expected_top_text_entities[i]))

    if max_text_identifiers is not None:
      self.assertSequenceEqual(
          list(num_disallowed_per_device),
          list(logging_helpers['n_disallowed']))


if __name__ == '__main__':
  absltest.main()

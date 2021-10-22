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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import batch_memory_attention_layer
import ml_collections


class BatchMemoryAttentionLayerTest(parameterized.TestCase):
  """Memory attention layer test."""

  dtype = jnp.float32
  memory_key_dim = 32
  memory_value_dim = 64
  input_dim = 64
  memory_update_type = 'additive'
  query_size = 16
  table_size = 128
  rows = 4
  splits = 2
  seq_len = 20
  bsz = 2
  n_devices = 4
  n_mentions = 5
  entity_vocab_size = 10

  memory_update_config = ml_collections.FrozenConfigDict({})

  @parameterized.parameters(
      (2),
      (None),
  )
  def test_model_shape(self, k_top):
    """Test batch memory attention layer output shape as expected."""

    model = batch_memory_attention_layer.BatchMemoryAttentionLayer(
        memory_key_dim=self.memory_key_dim,
        input_dim=self.input_dim,
        memory_update_type=self.memory_update_type,
        memory_update_config=self.memory_update_config,
        dtype=self.dtype,
        k_top=k_top,
        rows=self.rows,
    )

    rng = jax.random.PRNGKey(0)
    encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.input_dim), dtype=self.dtype)

    mention_batch_positions = jnp.tile(
        jnp.arange(self.bsz).reshape(-1, 1), (1, 3)).reshape(-1)
    mention_start_positions = jnp.tile(jnp.asarray([0, 5, 10]), (self.bsz))
    mention_end_positions = jnp.tile(jnp.asarray([2, 7, 12]), (self.bsz))
    n_mentions = mention_start_positions.shape[-1]

    mention_mask = jnp.tile(jnp.asarray([1, 1, 1]), (self.bsz))

    memory_keys = jnp.ones((self.table_size, self.memory_key_dim),
                           dtype=self.dtype)
    memory_values = jnp.ones((self.table_size, self.memory_value_dim),
                             dtype=self.dtype)
    memory_entity_ids = jnp.arange(self.table_size)
    memory_mask = jnp.ones((self.table_size))

    (encoded_output, loss_helpers, _), _ = model.init_with_output(
        rng,
        encoding=encoded_input,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        memory_keys=memory_keys,
        memory_values=memory_values,
        memory_mask=memory_mask,
        memory_entity_ids=memory_entity_ids,
        deterministic=True,
    )

    attention_weights = loss_helpers['memory_attention_weights']
    entity_ids = loss_helpers['top_entity_ids']

    # Check shapes as expected
    self.assertSequenceEqual(encoded_output.shape,
                             (self.bsz, self.seq_len, self.input_dim))

    if k_top is None:
      return_shape = self.table_size
    else:
      return_shape = k_top

    self.assertSequenceEqual(attention_weights.shape,
                             (n_mentions, return_shape))

    self.assertSequenceEqual(entity_ids.shape, (n_mentions, return_shape))

  @parameterized.parameters(
      (2),
      (None),
  )
  def test_model_backward(self, k_top):
    model = batch_memory_attention_layer.BatchMemoryAttentionLayer(
        memory_key_dim=self.memory_key_dim,
        input_dim=self.input_dim,
        memory_update_type=self.memory_update_type,
        memory_update_config=self.memory_update_config,
        dtype=self.dtype,
        k_top=k_top,
        rows=self.rows,
    )

    rng = jax.random.PRNGKey(0)
    encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.input_dim), dtype=self.dtype)

    mention_batch_positions = jnp.tile(
        jnp.arange(self.bsz).reshape(-1, 1), (1, 3)).reshape(-1)
    mention_start_positions = jnp.tile(jnp.asarray([0, 5, 10]), (self.bsz))
    mention_end_positions = jnp.tile(jnp.asarray([2, 7, 12]), (self.bsz))

    mention_mask = jnp.tile(jnp.asarray([1, 1, 1]), (self.bsz))

    memory_keys = jnp.ones((self.table_size, self.memory_key_dim),
                           dtype=self.dtype)
    memory_values = jnp.ones((self.table_size, self.memory_value_dim),
                             dtype=self.dtype)
    memory_mask = jnp.ones((self.table_size))
    memory_entity_ids = jnp.arange(self.table_size)

    initial_parameters = model.init(
        rng,
        encoding=encoded_input,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        memory_keys=memory_keys,
        memory_values=memory_values,
        memory_mask=memory_mask,
        memory_entity_ids=memory_entity_ids,
        deterministic=True,
    )

    def step_fn(
        params,
        encoded_input,
        mention_batch_positions,
        mention_start_positions,
        mention_end_positions,
        mention_mask,
        memory_keys,
        memory_values,
        memory_mask,
        memory_entity_ids,
    ):

      def loss_fn(params):
        encoded_output, _, _ = model.apply(
            {'params': params},
            rngs=None,
            encoding=encoded_input,
            mention_batch_positions=mention_batch_positions,
            mention_start_positions=mention_start_positions,
            mention_end_positions=mention_end_positions,
            mention_mask=mention_mask,
            memory_keys=memory_keys,
            memory_values=memory_values,
            memory_mask=memory_mask,
            memory_entity_ids=memory_entity_ids,
            deterministic=True,
        )
        return encoded_output.sum()

      loss, grad = jax.value_and_grad(loss_fn)(params)
      return loss, grad

    _ = step_fn(
        initial_parameters['params'],
        encoded_input=encoded_input,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        memory_keys=memory_keys,
        memory_values=memory_values,
        memory_mask=memory_mask,
        memory_entity_ids=memory_entity_ids,
    )


if __name__ == '__main__':
  absltest.main()

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
"""Tests for mention memory encoder."""

import copy
import os

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import mention_memory_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import mention_memory_task
from language.mentionmemory.utils import checkpoint_utils
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np

# easiest to define as constant here
MENTION_SIZE = 2


class MentionMemoryEncoderTest(parameterized.TestCase):
  """Tests for mention memory encoder."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'memory_key_dim': 4,
      'memory_value_dim': 4,
      'separate_memory_values': False,
      'memory_update_type': 'additive',
      'memory_update_config': {},
      'same_passage_memory_policy': 'disallow',
      'k_top_device': 2,
      'rows': 4,
      'splits': 2,
      'max_length': 128,
      'max_positions': 128,
      'hidden_size': 4,
      'intermediate_dim': 8,
      'num_attention_heads': 2,
      'num_initial_layers': 1,
      'num_final_layers': 1,
      'dropout_rate': 0.1,
      'n_memory_text_entities': 2,
      'final_k_top_device': 2,
      'final_splits': 2,
  }

  model_config = {
      'encoder_config': encoder_config,
  }

  config = {
      'model_config': model_config,
      'seed': 0,
      'per_device_batch_size': 2,
      'samples_per_example': 1,
      'mask_rate': 0.2,
      'mention_mask_rate': 0.2,
      'mlm_weight': 0.5,
      'el_im_weight': 0.25,
      'coref_res_weight': 0.25,
      'max_mention_targets': 5,
      'max_mlm_targets': 25,
      'max_mentions': 10,
      # Enable same-entity-set-retrieval loss so `memory_text_entities`
      # will be loaded.
      'same_entity_set_retrieval_weight': 0.1,
  }

  n_devices = 4
  table_size = 1024

  text_length = 100
  n_mentions = 5
  n_linked_mentions = 3

  @parameterized.parameters(
      {},
      {'separate_memory_values': True},
      {'num_intermediate_layers': 1},
  )
  def test_model_shape(
      self,
      separate_memory_values=False,
      num_intermediate_layers=None,
  ):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    config['model_config']['encoder_config'][
        'separate_memory_values'] = separate_memory_values
    config['model_config']['encoder_config'][
        'num_intermediate_layers'] = num_intermediate_layers
    config = ml_collections.FrozenConfigDict(config)

    model_config = config.model_config
    encoder_config = model_config.encoder_config

    rows = encoder_config.rows
    preprocess_fn = mention_memory_task.MentionMemoryTask.make_preprocess_fn(config)  # pylint: disable=line-too-long
    collater_fn = mention_memory_task.MentionMemoryTask.make_collater_fn(config)

    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    model = mention_memory_encoder.MentionMemoryEncoder(**encoder_config)
    dummy_input = mention_memory_task.MentionMemoryTask.dummy_input(config)
    dummy_input = jax.device_put_replicated(dummy_input, devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    memory_table = np.random.rand(rows, self.table_size // rows,
                                  encoder_config.memory_key_dim)
    memory_keys = jax.device_put_replicated(memory_table, devices)
    memory_values = memory_table.reshape(-1, encoder_config.memory_key_dim)
    memory_values = jax.device_put_replicated(memory_values, devices)
    memory_identifiers = np.arange(self.table_size)
    memory_identifiers = jax.device_put_replicated(memory_identifiers, devices)
    memory_entity_ids = memory_identifiers
    memory_text_entities = np.zeros(
        (self.table_size, encoder_config.n_memory_text_entities),
        dtype=np.int32)
    memory_text_entities = jax.device_put_replicated(memory_text_entities,
                                                     devices)

    def model_init(*args, **kwargs):
      return model.init(*args, method=model.forward, **kwargs)

    initial_variables = jax.pmap(
        model_init, 'batch', static_broadcasted_argnums=2)(
            split_rng,
            dummy_input,
            True,
        )
    initial_variables = {'params': initial_variables['params']}
    initial_variables['constants'] = {
        'memory_keys': memory_keys,
        'memory_values': memory_values,
        'memory_identifiers': memory_identifiers,
        'memory_entity_ids': memory_entity_ids,
        'memory_text_entities': memory_text_entities,
    }

    raw_example = test_utils.gen_mention_pretraining_sample(
        self.text_length,
        self.n_mentions,
        self.n_linked_mentions,
        max_length=encoder_config.max_length)
    processed_example = preprocess_fn(raw_example)
    batch = {
        key: np.tile(value, (config.per_device_batch_size, 1))
        for key, value in processed_example.items()
    }
    batch = collater_fn(batch)
    batch = {
        key: test_utils.tensor_to_numpy(value) for key, value in batch.items()
    }
    batch = {
        key: jax.device_put_replicated(value, devices)
        for key, value in batch.items()
    }

    def model_apply(*args, **kwargs):
      return model.apply(*args, method=model.forward, **kwargs)

    papply = jax.pmap(model_apply, 'batch', static_broadcasted_argnums=(2))
    encoded_output, loss_helpers, _ = papply(
        {
            'params': initial_variables['params'],
            'constants': initial_variables['constants'],
        },
        batch,
        True,
    )

    self.assertEqual(encoded_output.shape,
                     (self.n_devices, config.per_device_batch_size,
                      encoder_config.max_length, encoder_config.hidden_size))

    memory_value_dim = encoder_config.memory_value_dim
    memory_key_dim = encoder_config.memory_key_dim
    memory_size = memory_value_dim if memory_value_dim else memory_key_dim
    self.assertEqual(loss_helpers['target_mention_encodings'].shape,
                     (self.n_devices, config.max_mention_targets *
                      config.per_device_batch_size, memory_size))

  @parameterized.parameters(
      {},
      {'separate_memory_values': True},
      {'memory_only': True},
  )
  def test_load_weights(self, separate_memory_values=False, memory_only=False):
    """Test saving and loading model recovers original parameters."""

    config = copy.deepcopy(self.config)
    config['model_config']['encoder_config'][
        'separate_memory_values'] = separate_memory_values
    config = ml_collections.ConfigDict(config)

    model_config = config.model_config
    encoder_config = model_config.encoder_config
    rows = encoder_config.rows
    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()
    model = mention_memory_encoder.MentionMemoryEncoder(**encoder_config)
    dummy_input = mention_memory_task.MentionMemoryTask.dummy_input(config)
    dummy_input = jax.device_put_replicated(dummy_input, devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    memory_table = np.random.rand(rows, self.table_size // rows,
                                  encoder_config.memory_key_dim)
    memory_keys = jax.device_put_replicated(memory_table, devices)
    memory_values = memory_table.reshape(-1, encoder_config.memory_key_dim)
    memory_values = jax.device_put_replicated(memory_values, devices)
    memory_identifiers = np.arange(self.table_size)
    memory_identifiers = jax.device_put_replicated(memory_identifiers, devices)
    memory_entity_ids = memory_identifiers
    memory_text_entities = np.zeros(
        (self.table_size, encoder_config.n_memory_text_entities),
        dtype=np.int32)
    memory_text_entities = jax.device_put_replicated(memory_text_entities,
                                                     devices)

    def model_init(*args, **kwargs):
      return model.init(*args, method=model.forward, **kwargs)

    initial_variables = jax.pmap(
        model_init, 'batch', static_broadcasted_argnums=2)(
            split_rng,
            dummy_input,
            True,
        )
    initial_variables = {'params': initial_variables['params']}
    initial_variables['constants'] = {
        'memory_keys': memory_keys,
        'memory_values': memory_values,
        'memory_identifiers': memory_identifiers,
        'memory_entity_ids': memory_entity_ids,
        'memory_text_entities': memory_text_entities,
    }
    n_shards = 4

    tempdir_obj = self.create_tempdir()
    tempdir = tempdir_obj.full_path

    memory_key_base = os.path.join(tempdir, 'memory_keys')
    memory_value_base = os.path.join(tempdir, 'memory_values')
    memory_id_base = os.path.join(tempdir, 'memory_id')
    memory_entity_id_base = os.path.join(tempdir, 'memory_entity_id')
    memory_text_entities_base = os.path.join(tempdir, 'memory_text_entities')

    unreplicated_variables = jax_utils.unreplicate(initial_variables)
    unreplicated_variables['params'] = flax.core.unfreeze(
        unreplicated_variables['params']
    )

    if memory_only:
      load_weights = 'memory_only'
    else:
      load_weights = os.path.join(tempdir, 'weights')
      checkpoint_utils.save_weights(
          load_weights, unreplicated_variables['params']
      )

    memory_keys = initial_variables['constants']['memory_keys']
    memory_keys = memory_keys.reshape(
        n_shards, -1, encoder_config.memory_key_dim
    )
    memory_values = initial_variables['constants']['memory_values']
    memory_values = memory_values.reshape(
        n_shards, -1, encoder_config.memory_key_dim
    )
    memory_ids = initial_variables['constants']['memory_identifiers'].reshape(
        n_shards, -1
    )
    memory_entity_ids = initial_variables['constants'][
        'memory_entity_ids'
    ].reshape(n_shards, -1)
    memory_text_entities = initial_variables['constants'][
        'memory_text_entities'
    ].reshape(n_shards, -1, encoder_config.n_memory_text_entities)

    for shard in range(n_shards):
      np.save(memory_key_base + str(shard), memory_keys[shard])
      np.save(memory_value_base + str(shard), memory_values[shard])
      np.save(memory_id_base + str(shard), memory_ids[shard])
      np.save(memory_entity_id_base + str(shard), memory_entity_ids[shard])
      np.save(memory_entity_id_base + str(shard), memory_entity_ids[shard])
      np.save(memory_text_entities_base + str(shard),
              memory_text_entities[shard])

    config.memory_key_pattern = memory_key_base + '*'
    config.memory_value_pattern = memory_value_base + '*'
    config.memory_id_pattern = memory_id_base + '*'
    config.memory_entity_id_pattern = memory_entity_id_base + '*'
    config.memory_text_entities_pattern = memory_text_entities_base + '*'
    config.load_weights = load_weights

    loaded_variables = mention_memory_encoder.MentionMemoryEncoder.load_weights(
        config)

    arrayeq = lambda x, y: jnp.all(x == y)
    constants = {
        key: value
        for key, value in initial_variables['constants'].items()
        if not (key == 'memory_values' and not separate_memory_values)
    }
    comparison_variables = {'constants': constants}
    if not memory_only:
      comparison_variables['params'] = flax.core.unfreeze(
          initial_variables['params']
      )

    self.assertTrue(
        jax.tree_map(arrayeq, loaded_variables, comparison_variables)
    )


if __name__ == '__main__':
  absltest.main()

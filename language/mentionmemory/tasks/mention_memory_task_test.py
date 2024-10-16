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
"""Tests for mention memory model."""

import copy
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import mention_memory_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import mention_memory_task
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import test_utils
from language.mentionmemory.utils.custom_types import Array
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np

# easiest to define as constant here
MENTION_SIZE = 2


def stack(examples):
  features = examples[0].keys()
  return {k: np.stack([example[k] for example in examples]) for k in features}


class MentionMemoryTaskTest(test_utils.TestCase):
  """Tests for mention memory task."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'memory_key_dim': 1,
      'memory_value_dim': 1,
      'separate_memory_values': False,
      'memory_update_type': 'additive',
      'memory_update_config': {},
      'same_passage_memory_policy': 'disallow',
      'k_top_device': 2,
      'k_top_post_selection': None,
      'rows': 4,
      'splits': 2,
      'max_length': 128,
      'max_positions': 128,
      'hidden_size': 2,
      'intermediate_dim': 4,
      'num_attention_heads': 1,
      'num_initial_layers': 1,
      'num_final_layers': 1,
      'dropout_rate': 0.1,
      'n_memory_text_entities': 2,
      'final_k_top_device': 2,
      'final_k_top_post_selection': None,
      'final_splits': 2,
  }

  model_config = {
      'encoder_config': encoder_config,
  }

  config = {
      'model_config': model_config,
      'seed': 0,
      'per_device_batch_size': 5,
      'mask_rate': 0.2,
      'mention_mask_rate': 0.2,
      'mlm_weight': 0.5,
      'el_im_weight': 0.1,
      'coref_res_weight': 0.15,
      'same_passage_weight': 0.05,
      'mtb_im_weight': 0.05,
      'mtb_final_weights': 0.1,
      'mtb_score_mode': 'dot',
      'max_mention_targets': 4,
      'max_mlm_targets': 5,
      'max_mentions': 6,
      'save_k_retrieval': None,
      'same_entity_set_retrieval_weight': 0.05,
      'same_entity_set_target_threshold': 2,
      'el_final_weight': 0.05,
      'memory_reduction': 1,
      'memory_prop': None
  }

  n_devices = 3
  table_size = 64
  memory_text_length = 2

  text_length = 110
  n_mentions = 5
  n_linked_mentions = 3

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)
    self.devices = jax.local_devices()

  def save_sharded_array(self, array: Array, name: str) -> str:
    tmp_dir = self.create_tempdir()
    prefix = os.path.join(tmp_dir.full_path, name)
    for device_index in range(self.n_devices):
      data_utils.save_sharded_array(array[device_index], prefix, self.n_devices,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                    self.n_devices, device_index, 1)
    return '%s-?????-of-%05d' % (prefix, self.n_devices)

  def run_model(self, config, entity_vocab_size):
    """Initialize and run the model once, perform sanity checks."""
    np.random.seed(0)

    # Save arrays to test retrieval saver.
    memory_identifiers = np.arange(self.table_size)
    memory_identifiers = jax.device_put_replicated(memory_identifiers,
                                                   self.devices)
    memory_entity_ids = memory_identifiers
    config['memory_entity_id_pattern'] = self.save_sharded_array(
        memory_entity_ids, 'memory_entity_id')
    memory_text = np.random.randint(
        config['model_config']['encoder_config']['vocab_size'],
        size=(self.n_devices, self.table_size, self.memory_text_length),
        dtype=np.int32)
    config['memory_text_pattern'] = self.save_sharded_array(
        memory_text, 'memory_text')
    memory_positions = np.random.randint(
        self.memory_text_length,
        size=(self.n_devices, self.table_size, 2),
        dtype=np.int32)
    config['memory_positions_pattern'] = self.save_sharded_array(
        memory_positions, 'memory_positions')

    config = ml_collections.FrozenConfigDict(config)
    model_config = config.model_config
    encoder_config = model_config.encoder_config

    rows = encoder_config.rows
    preprocess_fn = mention_memory_task.MentionMemoryTask.make_preprocess_fn(config)  # pylint: disable=line-too-long
    collater_fn = mention_memory_task.MentionMemoryTask.make_collater_fn(config)
    postprocess_fn = mention_memory_task.MentionMemoryTask.make_output_postprocess_fn(
        config)

    model = mention_memory_task.MentionMemoryTask.build_model(model_config)
    dummy_input = mention_memory_task.MentionMemoryTask.dummy_input(config)
    dummy_input = jax.device_put_replicated(dummy_input, self.devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    memory_table = np.random.rand(rows, self.table_size // rows,
                                  encoder_config.memory_key_dim)
    memory_keys = jax.device_put_replicated(memory_table, self.devices)
    memory_values = memory_table.reshape(-1, encoder_config.memory_key_dim)
    memory_values = jax.device_put_replicated(memory_values, self.devices)

    # `memory_text_entities` are assumed to contain unique IDs in the last dim.
    memory_text_entities = np.zeros((self.n_devices, self.table_size,
                                     encoder_config.n_memory_text_entities),
                                    np.int32)
    for device_index in range(self.n_devices):
      for t_index in range(self.table_size):
        current_text_entities = np.random.choice(
            entity_vocab_size,
            size=(min(encoder_config.n_memory_text_entities,
                      entity_vocab_size)),
            replace=False)
        memory_text_entities[
            device_index,
            t_index, :len(current_text_entities)] = current_text_entities

    memory_text_entities = jax.device_put_sharded(
        list(memory_text_entities), self.devices)

    initial_variables = jax.pmap(
        model.init, 'batch', static_broadcasted_argnums=2)(
            split_rng,
            dummy_input,
            True,
        )
    initial_variables = {'params': initial_variables['params']}
    initial_variables['constants'] = {
        'encoder': {
            'memory_keys': memory_keys,
            'memory_values': memory_values,
            'memory_identifiers': memory_identifiers,
            'memory_entity_ids': memory_entity_ids,
            'memory_text_entities': memory_text_entities,
        }
    }

    def sample_batch():
      processed_examples = []
      for _ in range(config.per_device_batch_size):
        raw_example = test_utils.gen_mention_pretraining_sample(
            self.text_length,
            self.n_mentions,
            self.n_linked_mentions,
            entity_vocab_size=entity_vocab_size,
            max_length=encoder_config.max_length)
        processed_example = preprocess_fn(raw_example)
        processed_examples.append(processed_example)
      batch = stack(processed_examples)
      batch = collater_fn(batch)
      batch = {
          key: test_utils.tensor_to_numpy(value)
          for key, value in batch.items()
      }
      text_ids = batch['text_ids']
      for i in range(config.per_device_batch_size):
        for j in range(config.max_mlm_targets):
          if batch['mlm_target_weights'][i, j] > 0:
            text_ids[i, batch['mlm_target_positions'][
                i, j]] = batch['mlm_target_ids'][i, j]
      mention_batch_positions = batch['mention_batch_positions']
      text_identifiers = batch['text_identifiers'].astype(np.int32).tolist()
      expected_text_identifiers = [
          mention_preprocess_utils.text_hash(
              text_ids[mention_batch_positions[index]]).astype(np.int32)
          for index in range(len(mention_batch_positions))
      ]
      self.assertSequenceEqual(text_identifiers, expected_text_identifiers)
      return batch

    batch = stack([sample_batch() for _ in range(self.n_devices)])
    batch = {
        key: jax.device_put_sharded(list(value), self.devices)
        for key, value in batch.items()
    }

    loss_fn = jax.pmap(
        mention_memory_task.MentionMemoryTask.make_loss_fn(config),
        'batch',
        static_broadcasted_argnums=(0, 4))
    _, metrics, auxiliary_output = loss_fn(
        model_config,
        initial_variables['params'],
        {'constants': initial_variables['constants']},
        batch,
        True,
    )

    metrics_per_first_device = jax.tree.map(lambda x: x[0], metrics)
    self.assertEqual(metrics_per_first_device['mlm']['denominator'],
                     batch['mlm_target_weights'][0].sum())

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])

    n_mentions_per_device = (config.per_device_batch_size * config.max_mentions)
    if config.save_k_retrieval is not None:
      k_top_saved = min(config.save_k_retrieval,
                        encoder_config.k_top_post_selection)
    else:
      k_top_saved = (
          encoder_config.k_top_post_selection or
          encoder_config.k_top_device * self.n_devices)
    self.assertSequenceEqual(
        np.array(features['memory_text']).shape, [
            self.n_devices, n_mentions_per_device, k_top_saved,
            self.memory_text_length
        ])
    self.assertSequenceEqual(
        np.array(features['memory_positions']).shape,
        [self.n_devices, n_mentions_per_device, k_top_saved, 2])

    if encoder_config.get('num_intermediate_layers') is not None:
      self.assertSequenceEqual(
          np.array(features['second_memory_text']).shape, [
              self.n_devices, n_mentions_per_device, k_top_saved,
              self.memory_text_length
          ])
      self.assertSequenceEqual(
          np.array(features['second_memory_positions']).shape,
          [self.n_devices, n_mentions_per_device, k_top_saved, 2])

    return batch, initial_variables, metrics

  @parameterized.parameters(
      {},
      {'separate_memory_values': True},
      {'num_intermediate_layers': 1},
      {
          'num_intermediate_layers': 1,
          'el_second_im_weight': 0.1
      },
  )
  def test_loss_fn(
      self,
      separate_memory_values=False,
      num_intermediate_layers=None,
      el_second_im_weight=0.0,
  ):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    config['model_config']['encoder_config'][
        'separate_memory_values'] = separate_memory_values
    config['model_config']['encoder_config'][
        'num_intermediate_layers'] = num_intermediate_layers
    config['el_second_im_weight'] = el_second_im_weight
    self.run_model(config, entity_vocab_size=1000)

  @parameterized.parameters(
      (1, 120),
      (2, 120),
      (1, 150),
      (2, 150),
      (1, 180),
      (2, 180),
  )
  def test_same_entity_set_retrieval_loss(self,
                                          same_entity_set_target_threshold,
                                          entity_vocab_size):
    # We make the config such that the model retrieves ALL of memories
    config = copy.deepcopy(self.config)
    config['model_config']['encoder_config']['k_top_device'] = self.table_size
    config['model_config']['encoder_config']['rows'] = self.table_size
    config[
        'same_entity_set_target_threshold'] = same_entity_set_target_threshold

    batch, initial_variables, metrics = self.run_model(config,
                                                       entity_vocab_size)
    config = ml_collections.FrozenConfigDict(config)
    n_mentions_per_local_batch = self.n_mentions * config.per_device_batch_size

    memory_text_entities = initial_variables['constants']['encoder'][
        'memory_text_entities']
    memory_text_entities = memory_text_entities.reshape(
        -1, config.model_config.encoder_config.n_memory_text_entities)
    n_retrievals = memory_text_entities.shape[0]

    for device_index in range(self.n_devices):
      if metrics['disallowed']['per_mention'][device_index] > 0:
        # Ignore this device since `disallowed` might affect the denominator
        # for same-entity-set-retrieval loss.
        continue
      expected_same_entity_set_retrieval = 0

      mention_target_batch_positions = batch['mention_target_batch_positions'][
          device_index]
      mention_target_ids = batch['mention_target_ids'][device_index]
      mention_target_weights = batch['mention_target_weights'][device_index]
      mention_batch_positions = batch['mention_batch_positions'][device_index]
      mention_mask = batch['mention_mask'][device_index]

      expected_same_entity_set_retrieval_per_mention = [0] * (
          n_mentions_per_local_batch)
      num_commons = np.zeros((n_mentions_per_local_batch, n_retrievals))
      for batch_index in range(config.per_device_batch_size):
        sample_ids = mention_target_ids[mention_target_batch_positions ==
                                        batch_index]
        sample_weights = mention_target_weights[mention_target_batch_positions
                                                == batch_index]
        sample_ids = sample_ids[sample_weights > 0]
        sample_ids = set([x for x in sample_ids.tolist() if x != 0])

        for m_index in range(n_mentions_per_local_batch):
          if mention_batch_positions[m_index] != batch_index:
            continue
          if mention_mask[m_index] == 0:
            continue

          n_correct_retrievals, n_incorrect_retrievals = 0, 0
          for r_index in range(n_retrievals):
            common_ids = set(
                memory_text_entities[r_index].tolist()).intersection(sample_ids)
            num_commons[m_index, r_index] = len(common_ids)
            if len(common_ids) >= config.same_entity_set_target_threshold:
              n_correct_retrievals += 1
            else:
              n_incorrect_retrievals += 1

          if n_correct_retrievals > 0 and n_incorrect_retrievals > 0:
            expected_same_entity_set_retrieval += 1
            expected_same_entity_set_retrieval_per_mention[m_index] += 1

      self.assertEqual(
          metrics['same_entity_set_retrieval']['denominator'][device_index],
          expected_same_entity_set_retrieval)


if __name__ == '__main__':
  absltest.main()

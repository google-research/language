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
"""Tests for EaE task."""

import copy
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import eae_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import memory_generation_task
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np

# easiest to define as constant here
MENTION_SIZE = 2


def gen_eae_test_list():

  text_lengths = [0, 50, 128]
  n_mention_list = [0, 5, 10, 15]
  n_linked_mention_list = [0, 3, 5, 8, 10, 12, 15]

  # pylint: disable=g-complex-comprehension
  test_list = [
      (text_length, n_mentions, n_linked_mentions)
      for (
          text_length,
          n_mentions,
          n_linked_mentions,
      ) in itertools.product(text_lengths, n_mention_list,
                             n_linked_mention_list)
      if not (n_mentions *
              MENTION_SIZE >= text_length or n_linked_mentions > n_mentions)
  ]

  return test_list


class MemoryGenerationTaskTest(parameterized.TestCase):
  """Tests for MemoryGeneration task."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'entity_vocab_size': 1000,
      'max_positions': 512,
      'max_length': 128,
      'hidden_size': 64,
      'intermediate_dim': 128,
      'entity_dim': 3,
      'num_attention_heads': 2,
      'num_initial_layers': 1,
      'num_final_layers': 1,
      'dropout_rate': 0.1,
  }

  model_config = {
      'encoder_config': encoder_config,
      'encoder_name': 'eae',
      'dtype': 'float32',
  }

  config = {
      'model_config': model_config,
      'task_name': 'memory_generation',
      'seed': 0,
      'per_device_batch_size': 2,
      'samples_per_example': 1,
      'memory_dim': 3,
      'mask_rate': 0,
      'mention_mask_rate': 0,
      'max_mlm_targets': 0,
      'max_mention_targets': 10,
      'max_mentions': 20,
      'max_mentions_per_sample': 11,
      'min_distance_from_passage_boundary': 2,
  }

  @parameterized.parameters(gen_eae_test_list())
  def test_prediction_fn(self, text_length, n_mentions, n_linked_mentions):
    """Test loss function runs and produces expected values."""

    config = copy.deepcopy(self.config)
    config = ml_collections.FrozenConfigDict(config)

    max_length = config.model_config.encoder_config.max_length
    preprocess_fn = memory_generation_task.MemoryGenerationTask.make_preprocess_fn(
        config)
    collater_fn = memory_generation_task.MemoryGenerationTask.make_collater_fn(
        config)

    model = memory_generation_task.MemoryGenerationTask.build_model(
        config.model_config)
    dummy_input = memory_generation_task.MemoryGenerationTask.dummy_input(
        config)
    init_rng = jax.random.PRNGKey(0)
    init_parameters = model.init(init_rng, dummy_input, True)

    raw_example = test_utils.gen_mention_pretraining_sample(
        text_length, n_mentions, n_linked_mentions, max_length=max_length)
    processed_example = preprocess_fn(raw_example)
    batch = {
        key: np.tile(value, (config.per_device_batch_size, 1))
        for key, value in processed_example.items()
    }
    batch = collater_fn(batch)
    batch = jax.tree.map(np.asarray, batch)

    predict_fn = memory_generation_task.MemoryGenerationTask.make_prediction_fn(
        config)
    predictions = predict_fn(
        model_config=config.model_config,
        model_params=init_parameters['params'],
        model_vars={},
        batch=batch,
    )

    self.assertSequenceEqual(predictions['values'].shape,
                             (config.max_mention_targets *
                              config.per_device_batch_size, config.memory_dim))


def gen_memory_saver_test_list():

  num_total_memories_list = [1, 20, 50]
  batch_size_list = [1, 5, 10]
  n_mentions_per_batch_list = [1, 10]
  num_shards_list = [1, 2, 3]
  n_devices_list = [1, 2, 3]
  shard_size_divisible_list = [1, 3, 5]

  test_list = list(
      itertools.product(num_total_memories_list, batch_size_list,
                        n_mentions_per_batch_list, n_devices_list,
                        num_shards_list, shard_size_divisible_list))

  return test_list


class MemorySaverTest(parameterized.TestCase):
  """Tests for MemoryGeneration task."""

  memory_dim = 3
  memory_key_dim = 2
  text_length = 2
  max_mentions_per_sample = 3

  def _stack(self, d):
    self.assertNotEmpty(d)
    keys = d[0].keys()
    result = {}
    for key in keys:
      result[key] = np.stack([x[key] for x in d])
    return result

  def _sample_batch(self, batch_index, batch_size, n_mentions_per_batch):
    mention_target_weights = np.random.randint(2, size=(n_mentions_per_batch,))
    # Unique mention ID per sample
    mention_target_ids = 1 + np.arange(
        n_mentions_per_batch) + batch_index * n_mentions_per_batch
    mention_batch_positions = np.random.randint(
        batch_size, size=(n_mentions_per_batch,))
    text_identifiers = 1 + mention_batch_positions + batch_index * batch_size
    # mention hashes and encodings are same as entity IDs
    mention_encodings = np.expand_dims(mention_target_ids, 1)
    mention_encodings = np.tile(mention_encodings, self.memory_dim)
    text_ids = 1 + np.arange(batch_size) + batch_index * batch_size
    text_ids = np.expand_dims(text_ids, 1)
    text_ids = np.tile(text_ids, self.text_length)

    # Collect unique entity IDs per every passage
    text_entities = [set() for _ in range(batch_size)]
    for m_index in range(n_mentions_per_batch):
      if mention_target_weights[m_index] > 0:
        text_entities[mention_batch_positions[m_index]].add(
            mention_target_ids[m_index])

    unique_mention_ids = np.zeros((batch_size, self.max_mentions_per_sample),
                                  dtype=np.int32)
    # pylint:disable=g-explicit-length-test
    for i in range(batch_size):
      text_entities[i] = np.array(list(text_entities[i]), dtype=np.int32)
      num_unique_entities = len(text_entities[i])
      if num_unique_entities > self.max_mentions_per_sample:
        unique_mention_ids[i] = text_entities[i][:self.max_mentions_per_sample]
      elif num_unique_entities > 0:
        unique_mention_ids[i, :num_unique_entities] = text_entities[i]
      else:
        # i-th sample doesn't contain any entities
        pass

    batch = {
        'mention_target_weights': mention_target_weights,
        'mention_target_ids': mention_target_ids,
        'target_text_identifiers': text_identifiers,
        'target_mention_hashes': mention_target_ids,
        'text_ids': text_ids,
        'mention_target_batch_positions': mention_batch_positions,
        'mention_target_start_positions': mention_target_ids,
        'mention_target_end_positions': mention_target_ids,
        'unique_mention_ids': unique_mention_ids,
    }

    predictions = {
        'values': mention_encodings,
        'keys': mention_encodings[:, :self.memory_key_dim],
    }
    return batch, predictions

  @parameterized.parameters(gen_memory_saver_test_list())
  def test_memory_saver(self, num_total_memories, batch_size,
                        n_mentions_per_batch, n_devices, num_shards,
                        shard_size_divisible):

    memory_saver = memory_generation_task.MemorySaver(
        num_total_memories, self.memory_dim, self.text_length,
        self.max_mentions_per_sample, self.memory_key_dim)

    mention_to_batch = []
    batch_to_set_of_mentions = {}
    batch_index = 0
    while True:
      all_batch = []
      all_predictions = []
      for _ in range(n_devices):
        batch, predictions = self._sample_batch(batch_index, batch_size,
                                                n_mentions_per_batch)
        for i in range(n_mentions_per_batch):
          if batch['mention_target_weights'][i] == 1:
            current_batch_index = batch['target_text_identifiers'][i]
            entity_id = batch['mention_target_ids'][i]
            mention_to_batch.append((entity_id, current_batch_index))
            if current_batch_index not in batch_to_set_of_mentions:
              batch_to_set_of_mentions[current_batch_index] = set()
            batch_to_set_of_mentions[current_batch_index].add(entity_id)
        batch_index += 1

        all_batch.append(batch)
        all_predictions.append(predictions)
      all_batch = self._stack(all_batch)
      all_predictions = self._stack(all_predictions)
      memory_saver.add_memories(all_batch, all_predictions)
      if memory_saver.get_num_memories() >= num_total_memories:
        break

    # Keep only first num_total_memories memories
    mention_to_batch = dict(mention_to_batch[:num_total_memories])

    tmp_dir = self.create_tempdir()
    memory_saver.save(tmp_dir.full_path, num_shards, 1, 0, shard_size_divisible)

    def load_array(suffix):
      return data_utils.load_sharded_array(
          os.path.join(tmp_dir.full_path,
                       suffix + '-?????-of-%05d' % num_shards), 1, 0)

    mention_encodings = load_array('encodings')
    mention_target_ids = load_array('labels')
    text_identifiers = load_array('hashes')
    mention_hashes = load_array('mention_hashes')
    texts = load_array('texts')
    positions = load_array('positions')
    text_entities = load_array('text_entities')

    self.assertSetEqual(
        set(mention_to_batch.keys()),
        set(mention_target_ids[mention_target_ids > 0]))
    for i in range(len(mention_target_ids)):
      if mention_target_ids[i] > 0:
        batch_index = mention_to_batch[mention_target_ids[i]]
        self.assertEqual(text_identifiers[i], batch_index)
        self.assertTrue(np.all(texts[i] == batch_index))
        self.assertEqual(mention_hashes[i], mention_target_ids[i])
        self.assertTrue(np.all(mention_encodings[i] == mention_target_ids[i]))
        self.assertTrue(np.all(positions[i] == mention_target_ids[i]))

        current_text_entities = [x for x in text_entities[i] if x != 0]
        self.assertSequenceEqual(
            sorted(current_text_entities),
            sorted(list(set(current_text_entities))))
        current_text_entities = set(current_text_entities)
        # These two sets might not be exactly equal, since `text_entities`
        # contains at most `max_mentions_per_sample` unique entities for every
        # mention.
        self.assertContainsSubset(current_text_entities,
                                  batch_to_set_of_mentions[batch_index])


if __name__ == '__main__':
  absltest.main()

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
"""Tests for MentionBasedEntityQATask task."""

import copy
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import mention_memory_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import mention_based_entity_qa_task
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import test_utils
from language.mentionmemory.utils.custom_types import Array
import ml_collections
import numpy as np


def stack(examples):
  features = examples[0].keys()
  return {k: np.stack([example[k] for example in examples]) for k in features}


class PredictionFunctionsTest(test_utils.TestCase):

  entity_vocab_size = 100

  @parameterized.parameters(
      (1, 100),
      (10, 6),
      (20, 1),
      (20, 20),
  )
  def test_get_predictions_max(self, n_mentions, n_retrievals):
    shape = (n_mentions, n_retrievals)
    attention_weights = np.random.random(shape)
    attention_weights /= attention_weights.sum(axis=1, keepdims=True)
    memory_entity_ids = np.random.randint(self.entity_vocab_size, size=shape)
    weights = np.random.randint(2, size=(n_mentions))

    actual_predictions = mention_based_entity_qa_task.get_predictions_max(
        attention_weights, memory_entity_ids, weights)
    for m_index in range(n_mentions):
      if weights[m_index] == 0:
        self.assertEqual(actual_predictions[m_index], 0)
      else:
        top_memory = np.argmax(attention_weights[m_index])
        self.assertEqual(actual_predictions[m_index],
                         memory_entity_ids[m_index, top_memory])

  @parameterized.parameters(
      (10, 6),
      (20, 1),
      (20, 20),
      (20, 30),
      (1, 100),
  )
  def test_get_predictions_sum(self, n_mentions, n_retrievals):
    shape = (n_mentions, n_retrievals)
    attention_weights = np.random.random(shape)
    attention_weights /= attention_weights.sum(axis=1, keepdims=True)
    memory_entity_ids = np.random.randint(self.entity_vocab_size, size=shape)
    weights = np.random.randint(2, size=(n_mentions))

    actual_predictions = mention_based_entity_qa_task.get_predictions_sum(
        attention_weights, memory_entity_ids, weights, self.entity_vocab_size)

    for m_index in range(n_mentions):
      if weights[m_index] == 0:
        self.assertEqual(actual_predictions[m_index], 0)
      else:
        attention_weights_per_entity = np.zeros(self.entity_vocab_size)
        for r_index in range(n_retrievals):
          attention_weights_per_entity[memory_entity_ids[
              m_index, r_index]] += attention_weights[m_index, r_index]
        top_entity = np.argmax(attention_weights_per_entity)
        self.assertEqual(actual_predictions[m_index], top_entity)


class MentionBasedEntityQATaskTest(test_utils.TestCase):
  """Tests for mention memory QA task."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'memory_key_dim': 4,
      'memory_value_dim': 4,
      'separate_memory_values': False,
      'memory_update_type': 'additive',
      'memory_update_config': {},
      'same_passage_memory_policy': 'allow',
      'k_top_device': 2,
      'k_top_post_selection': 2,
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
      'n_memory_text_entities': None,
      'final_k_top_device': 2,
      'final_splits': 2,
      'final_k_top_post_selection': 2,
  }

  model_config = {
      'encoder_config': encoder_config,
  }

  config = {
      'model_config': model_config,
      'seed': 0,
      'per_device_batch_size': 6,
      'mask_rate': 0.2,
      'mention_mask_rate': 0.2,
      'mlm_weight': 0.5,
      'el_im_weight': 0.1,
      'coref_res_weight': 0.15,
      'same_passage_weight': 0.05,
      'entity_vocab_size': 100,
      'mtb_im_weight': 0.05,
      'mtb_final_weights': 0.1,
      'mtb_score_mode': 'dot',
      'max_mention_targets': 1,
      'max_mlm_targets': 5,
      'max_mentions': 6,
      'apply_answer_mask': True,
      'same_entity_set_retrieval_weight': 0.05,
      'same_entity_set_target_threshold': 2,
      'save_k_top_retrievals': 3,
      'memory_reduction': 1,
      'memory_prop': None
  }

  n_devices = 3
  table_size = 128
  memory_text_length = 3

  text_length = 110
  n_mentions = 5
  n_linked_mentions = 3

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)
    self.devices = jax.local_devices()

  def save_sharded_array(self, array, name):
    tmp_dir = self.create_tempdir()
    prefix = os.path.join(tmp_dir.full_path, name)
    for device_index in range(self.n_devices):
      data_utils.save_sharded_array(array[device_index], prefix, self.n_devices,
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
    preprocess_fn = mention_based_entity_qa_task.MentionBasedEntityQATask.make_preprocess_fn(config)  # pylint: disable=line-too-long
    collater_fn = mention_based_entity_qa_task.MentionBasedEntityQATask.make_collater_fn(
        config)
    postprocess_fn = mention_based_entity_qa_task.MentionBasedEntityQATask.make_output_postprocess_fn(
        config)

    model = mention_based_entity_qa_task.MentionBasedEntityQATask.build_model(
        model_config)
    dummy_input = mention_based_entity_qa_task.MentionBasedEntityQATask.dummy_input(
        config)
    dummy_input = jax.device_put_replicated(dummy_input, self.devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    memory_table = np.random.rand(rows, self.table_size // rows,
                                  encoder_config.memory_key_dim)
    memory_keys = jax.device_put_replicated(memory_table, self.devices)
    memory_values = memory_table.reshape(-1, encoder_config.memory_key_dim)
    memory_values = jax.device_put_replicated(memory_values, self.devices)

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
      return batch

    batch = stack([sample_batch() for _ in range(self.n_devices)])
    batch = {
        key: jax.device_put_sharded(list(value), self.devices)
        for key, value in batch.items()
    }

    loss_fn = jax.pmap(
        mention_based_entity_qa_task.MentionBasedEntityQATask.make_loss_fn(
            config),
        'batch',
        static_broadcasted_argnums=(0, 4))
    _, metrics, auxiliary_output = loss_fn(
        model_config,
        initial_variables['params'],
        {'constants': initial_variables['constants']},
        batch,
        True,
    )

    self.assertArrayEqual(metrics['agg']['denominator'],
                          batch['mention_target_weights'].sum(1))

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])

    n_mentions_per_device = (
        config.per_device_batch_size * config.max_mention_targets)
    k_top_final = (
        encoder_config.final_k_top_post_selection or
        encoder_config.final_k_top_device * self.n_devices)
    self.assertSequenceEqual(
        np.array(features['memory_text']).shape, [
            self.n_devices, n_mentions_per_device, k_top_final,
            self.memory_text_length
        ])
    self.assertSequenceEqual(
        np.array(features['memory_positions']).shape,
        [self.n_devices, n_mentions_per_device, k_top_final, 2])

    return batch, initial_variables, metrics

  @parameterized.parameters(
      (False, 2, None),
      (True, 2, None),
      (False, 3, 5),
      (True, 3, 5),
  )
  def test_loss_fn(
      self,
      separate_memory_values,
      qa_k_top_device,
      qa_k_top_post_selection,
  ):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    config['model_config']['encoder_config']['k_top_device'] = qa_k_top_device
    config['model_config']['encoder_config'][
        'k_top_post_selection'] = qa_k_top_post_selection
    config['model_config']['encoder_config'][
        'separate_memory_values'] = separate_memory_values
    if qa_k_top_post_selection is None:
      config['save_k_retrieval'] = None
    self.run_model(config, 1000)


if __name__ == '__main__':
  absltest.main()

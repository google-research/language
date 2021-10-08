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
"""Tests for ReadTwice Task."""

import copy
import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import readtwice_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import readtwice_task
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np

# easiest to define as constant here
MENTION_SIZE = 2


class ReadTwiceTaskTest(test_utils.TestCase):
  """Tests for mention memory task."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'memory_key_dim': 4,
      'memory_value_dim': 4,
      'memory_update_type': 'additive',
      'memory_update_config': {},
      'rows': 4,
      'max_length': 128,
      'max_positions': 128,
      'hidden_size': 8,
      'intermediate_dim': 16,
      'num_attention_heads': 2,
      'num_initial_layers': 1,
      'num_final_layers': 1,
      'num_initial_layers_second': 1,
      'num_intermediate_layers_second': 1,
      'num_final_layers_second': 1,
      'dropout_rate': 0.1,
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
      'mlm_first_weight': 0.1,
      'el_im_weight': 0.1,
      'coref_key_weight': 0.1,
      'coref_value_weight': 0.1,
      'coref_final_weight': 0.1,
      'mtb_key_weight': 0.1,
      'mtb_value_weight': 0.1,
      'mtb_final_weight': 0.1,
      'max_mention_targets': 10,
      'max_mlm_targets': 25,
      'max_mentions': 20,
  }

  n_devices = 4
  text_length = 128
  n_mentions = 5
  n_linked_mentions = 3

  @parameterized.parameters(
      {
          'k_top': None,
      },
      {
          'k_top': 2,
      },
      {
          'k_top': 2,
          'num_intermediate_layers': 1,
      },
      {
          'k_top': 2,
          'shared_initial_encoder': False,
      },
      {
          'k_top': 2,
          'shared_final_encoder': False,
      },
      {
          'k_top': 2,
          'num_intermediate_layers': 1,
          'shared_intermediate_encoder': False,
      },
      {
          'k_top': 2,
          'no_retrieval': True,
      },
      {
          'k_top': 2,
          'no_retrieval_for_masked_mentions': True,
      },
      {
          'k_top': None,
          'same_passage_retrieval_policy': 'disallow',
      },
      {
          'k_top': None,
          'same_passage_retrieval_policy': 'only',
      },
      {
          'k_top': None,
          'no_retrieval_for_masked_mentions': True,
      },
      {
          'k_top': None,
          'num_intermediate_layers': 1,
          'same_passage_retrieval_policy': 'disallow',
      },
      {
          'k_top': None,
          'num_intermediate_layers': 1,
          'same_passage_retrieval_policy': 'disallow',
          'shared_intermediate_encoder': False,
      },
  )
  def test_loss_fn(
      self,
      k_top,
      num_intermediate_layers=None,
      shared_initial_encoder=True,
      shared_intermediate_encoder=True,
      shared_final_encoder=True,
      no_retrieval=False,
      same_passage_retrieval_policy='allow',
      extract_unlinked_mentions=False,
      no_retrieval_for_masked_mentions=False,
  ):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    encoder_config = copy.deepcopy(self.encoder_config)
    encoder_config['k_top'] = k_top
    encoder_config['num_intermediate_layers'] = num_intermediate_layers
    encoder_config['shared_initial_encoder'] = shared_initial_encoder
    encoder_config['shared_intermediate_encoder'] = shared_intermediate_encoder
    encoder_config['shared_final_encoder'] = shared_final_encoder
    encoder_config['no_retrieval'] = no_retrieval
    encoder_config[
        'same_passage_retrieval_policy'] = same_passage_retrieval_policy
    encoder_config['extract_unlinked_mentions'] = extract_unlinked_mentions
    encoder_config[
        'no_retrieval_for_masked_mentions'] = no_retrieval_for_masked_mentions
    config['model_config']['encoder_config'] = encoder_config
    if no_retrieval:
      config['el_im_weight'] = 0
    if num_intermediate_layers is not None:
      config['second_el_im_weight'] = 0.1
    config = ml_collections.FrozenConfigDict(config)

    model_config = config.model_config
    encoder_config = model_config.encoder_config

    preprocess_fn = readtwice_task.ReadTwiceTask.make_preprocess_fn(config)  # pylint: disable=line-too-long
    collater_fn = readtwice_task.ReadTwiceTask.make_collater_fn(config)
    postprocess_fn = readtwice_task.ReadTwiceTask.make_output_postprocess_fn(
        config)

    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    model = readtwice_task.ReadTwiceTask.build_model(model_config)
    dummy_input = readtwice_task.ReadTwiceTask.dummy_input(config)
    dummy_input = jax.device_put_replicated(dummy_input, devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    initial_variables = jax.pmap(
        model.init, 'batch', static_broadcasted_argnums=2)(
            split_rng,
            dummy_input,
            True,
        )
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

    loss_fn = jax.pmap(
        readtwice_task.ReadTwiceTask.make_loss_fn(config),
        'batch',
        static_broadcasted_argnums=(0, 4))
    _, metrics, auxiliary_output = loss_fn(
        model_config,
        initial_variables['params'],
        {},  # model vars
        batch,
        True,  # deterministic
    )

    take_first = lambda x: x[0]
    metrics = jax.tree_map(take_first, metrics)
    np_batch = jax.tree_map(take_first, batch)

    # mlm losses
    expected_mlm_denom = np_batch['mlm_target_weights'].sum()
    expected_mlm_mention_denom = (np_batch['mlm_target_weights'] *
                                  np_batch['mlm_target_is_mention']).sum()
    expected_mlm_non_mention_denom = (
        np_batch['mlm_target_weights'] *
        (1 - np_batch['mlm_target_is_mention'])).sum()
    self.assertEqual(metrics['mlm']['denominator'], expected_mlm_denom)
    self.assertEqual(metrics['mlm_mention']['denominator'],
                     expected_mlm_mention_denom)
    self.assertEqual(metrics['mlm_non_mention']['denominator'],
                     expected_mlm_non_mention_denom)
    self.assertEqual(metrics['mlm_first']['denominator'], expected_mlm_denom)
    self.assertEqual(metrics['mlm_mention_first']['denominator'],
                     expected_mlm_mention_denom)
    self.assertEqual(metrics['mlm_non_mention_first']['denominator'],
                     expected_mlm_non_mention_denom)

    # same entity retrieval loss
    if not no_retrieval:
      expected_same_entity_denom = np_batch['mention_target_weights'].sum()
      self.assertEqual(metrics['el_intermediate']['denominator'],
                       expected_same_entity_denom)
      if num_intermediate_layers is not None:
        self.assertEqual(metrics['second_el_intermediate']['denominator'],
                         expected_same_entity_denom)

    # coref losses
    expected_coref_denom = np_batch['mention_target_weights'].sum()
    expected_coref_masked_denom = (np_batch['mention_target_weights'] *
                                   np_batch['mention_target_is_masked']).sum()
    expected_coref_non_masked_denom = (
        np_batch['mention_target_weights'] *
        (1 - np_batch['mention_target_is_masked'])).sum()

    for coref_type in {'key', 'value', 'final'}:
      self.assertEqual(metrics[coref_type + '_coref_resolution']['denominator'],
                       expected_coref_denom)
      self.assertEqual(
          metrics[coref_type + '_coref_resolution_masked']['denominator'],
          expected_coref_masked_denom)
      self.assertEqual(
          metrics[coref_type + '_coref_resolution_non_masked']['denominator'],
          expected_coref_non_masked_denom)

    # mtb losses
    for mtb_type in {'key', 'value', 'final'}:
      self.assertIn(mtb_type + '_mtb', metrics)
      self.assertIn(mtb_type + '_mtb_masked', metrics)
      self.assertIn(mtb_type + '_mtb_non_masked', metrics)

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])


if __name__ == '__main__':
  absltest.main()

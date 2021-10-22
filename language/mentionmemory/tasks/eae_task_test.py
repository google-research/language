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
import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import eae_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import eae_task
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np

# easiest to define as constant here
MENTION_SIZE = 2


def gen_eae_test_list():

  text_lengths = [0, 50, 128]
  n_mention_list = [0, 5, 10, 15]
  n_linked_mention_list = [0, 3, 5, 8, 10, 12, 15]
  no_entity_attention = [True, False]

  # pylint: disable=g-complex-comprehension
  test_list = [
      (text_length, n_mentions, n_linked_mentions, no_entity_attention)
      for (
          text_length,
          n_mentions,
          n_linked_mentions,
          no_entity_attention,
      ) in itertools.product(text_lengths, n_mention_list,
                             n_linked_mention_list, no_entity_attention)
      if not (n_mentions *
              MENTION_SIZE >= text_length or n_linked_mentions > n_mentions)
  ]

  return test_list


class EaETaskTest(test_utils.TestCase):
  """Tests for EaE model."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'entity_vocab_size': 1000,
      'max_positions': 512,
      'max_length': 128,
      'hidden_size': 64,
      'intermediate_dim': 128,
      'entity_dim': 32,
      'num_attention_heads': 8,
      'num_initial_layers': 4,
      'num_final_layers': 8,
      'dropout_rate': 0.1,
  }

  model_config = {
      'encoder_config': encoder_config,
  }

  config = {
      'model_config': model_config,
      'task_name': 'eae',
      'seed': 0,
      'per_device_batch_size': 2,
      'samples_per_example': 1,
      'mask_rate': 0.2,
      'mention_mask_rate': 0.2,
      'mlm_weight': 0.5,
      'el_im_weight': 0.15,
      'el_final_weight': 0.15,
      'mtb_im_weight': 0.1,
      'mtb_final_weights': 0.1,
      'mtb_score_mode': 'dot',
      'max_mlm_targets': 25,
      'max_mention_targets': 5,
      'max_mentions': 10,
  }

  @parameterized.parameters(gen_eae_test_list())
  def test_loss_fn(self, text_length, n_mentions, n_linked_mentions,
                   no_entity_attention):
    """Test loss function runs and produces expected values."""

    model_config = copy.deepcopy(self.model_config)
    model_config['encoder_config']['no_entity_attention'] = no_entity_attention
    model_config = ml_collections.FrozenConfigDict(model_config)
    config = ml_collections.FrozenConfigDict(self.config)

    max_length = model_config.encoder_config.max_length
    preprocess_fn = eae_task.EaETask.make_preprocess_fn(config)
    collater_fn = eae_task.EaETask.make_collater_fn(config)
    postprocess_fn = eae_task.EaETask.make_output_postprocess_fn(config)

    model = eae_task.EaETask.build_model(model_config)
    dummy_input = eae_task.EaETask.dummy_input(config)
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
    batch = jax.tree_map(np.asarray, batch)

    loss_fn = eae_task.EaETask.make_loss_fn(config)
    _, metrics, auxiliary_output = loss_fn(
        model_config=model_config,
        model_params=init_parameters['params'],
        model_vars={},
        batch=batch,
        deterministic=True,
    )

    self.assertEqual(metrics['mlm']['denominator'],
                     batch['mlm_target_weights'].sum())
    self.assertEqual(metrics['el_intermediate']['denominator'],
                     batch['mention_target_weights'].sum())
    if batch['mention_target_weights'].sum() > 0:
      self.assertFalse(np.isnan(metrics['el_intermediate']['loss']))
    self.assertEqual(metrics['el_final']['denominator'],
                     batch['mention_target_weights'].sum())
    if batch['mention_target_weights'].sum() > 0:
      self.assertFalse(np.isnan(metrics['el_final']['loss']))

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])


if __name__ == '__main__':
  absltest.main()

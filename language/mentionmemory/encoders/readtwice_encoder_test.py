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
"""Tests for ReadTwice encoder."""

import copy

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


class ReadTwiceEncoderTest(parameterized.TestCase):
  """Tests for mention memory encoder."""

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'memory_key_dim': 3,
      'memory_value_dim': 3,
      'memory_update_type': 'additive',
      'memory_update_config': {},
      'rows': 4,
      'max_length': 128,
      'max_positions': 128,
      'hidden_size': 2,
      'intermediate_dim': 4,
      'num_attention_heads': 2,
      'num_initial_layers': 1,
      'num_initial_layers_second': 2,
      'num_intermediate_layers_second': 1,
      'num_final_layers': 2,
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
      'el_im_weight': 0.25,
      'coref_res_weight': 0.25,
      'max_mention_targets': 2,
      'max_mlm_targets': 25,
      'max_mentions': 5,
  }

  n_devices = 4
  text_length = 128
  n_mentions = 5
  n_linked_mentions = 3

  @parameterized.parameters({
      'k_top': None,
  }, {
      'k_top': 2,
  }, {
      'k_top': 2,
      'num_intermediate_layers': 1,
  }, {
      'k_top': 2,
      'shared_initial_encoder': False,
  }, {
      'k_top': 2,
      'shared_final_encoder': False,
  }, {
      'k_top': 2,
      'num_intermediate_layers': 1,
      'shared_intermediate_encoder': False,
  }, {
      'k_top': 2,
      'no_retrieval': True,
  }, {
      'k_top': 2,
      'no_retrieval_for_masked_mentions': True,
  }, {
      'k_top': None,
      'same_passage_retrieval_policy': 'disallow',
  }, {
      'k_top': None,
      'same_passage_retrieval_policy': 'only',
  }, {
      'k_top': None,
      'extract_unlinked_mentions': True,
  }, {
      'k_top': None,
      'no_retrieval_for_masked_mentions': True,
  })
  def test_model_shape(
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
    config = ml_collections.FrozenConfigDict(config)

    model_config = config.model_config
    encoder_config = model_config.encoder_config

    preprocess_fn = readtwice_task.ReadTwiceTask.make_preprocess_fn(config)  # pylint: disable=line-too-long
    collater_fn = readtwice_task.ReadTwiceTask.make_collater_fn(config)

    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    model = readtwice_encoder.ReadTwiceEncoder(**encoder_config)
    dummy_input = readtwice_task.ReadTwiceTask.dummy_input(config)
    dummy_input = jax.device_put_replicated(dummy_input, devices)
    init_rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(init_rng, self.n_devices)

    def model_init(*args, **kwargs):
      return model.init(*args, method=model.forward, **kwargs)

    initial_variables = jax.pmap(
        model_init, 'batch', static_broadcasted_argnums=2)(
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

    def model_apply(*args, **kwargs):
      return model.apply(*args, method=model.forward, **kwargs)

    papply = jax.pmap(model_apply, 'batch', static_broadcasted_argnums=(2))
    encoded_output, loss_helpers, _ = papply(
        {
            'params': initial_variables['params'],
        },
        batch,
        True,
    )

    self.assertEqual(encoded_output.shape,
                     (self.n_devices, config.per_device_batch_size,
                      encoder_config.max_length, encoder_config.hidden_size))

    self.assertEqual(
        loss_helpers['target_mention_encodings'].shape,
        (self.n_devices, config.max_mention_targets *
         config.per_device_batch_size, encoder_config.memory_value_dim))


if __name__ == '__main__':
  absltest.main()

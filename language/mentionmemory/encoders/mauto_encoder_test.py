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
"""Tests for mauto encoder."""

from absl.testing import absltest
import jax
from language.mentionmemory.encoders import mauto_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import mauto_task
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import tensorflow as tf

# easiest to define as constant here
MENTION_SIZE = 2


class MautoEncoderTest(absltest.TestCase):
  """Tests for mauto task."""

  retrieval_table_size = 512
  retrieval_dim = 32
  test_dir = '/tmp/test/mentionauto/'
  table_path = test_dir + 'retrieval_table'
  hash_path = test_dir + 'retrieval_hash'
  entity_path = test_dir + 'retrieval_entity_ids'
  mentions_per_sample = 5
  linked_mentions_per_sample = 3

  encoder_config_dict = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'retrieval_dim': retrieval_dim,
      'retrieval_update_type': 'concat_mlp',
      'retrieval_update_config': {
          'hidden_dim': 128,
          'retrieval_dim': retrieval_dim,
          'n_additional_concat_layers': 1,
          'n_pooled_layers': 1,
          'dropout_rate': 0.1
      },
      'max_length': 128,
      'max_positions': 512,
      'hidden_size': 64,
      'intermediate_dim': 128,
      'num_attention_heads': 8,
      'num_initial_layers': 4,
      'num_final_layers': 8,
      'dropout_rate': 0.1,
  }

  model_config_dict = {
      'encoder_config': encoder_config_dict,
  }

  config_dict = {
      'model_config': model_config_dict,
      'model_class': 'mention_autoencoder',
      'seed': 0,
      'per_device_batch_size': 4,
      'samples_per_example': 1,
      'mask_rate': 0.2,
      'mention_mask_rate': 0.5,
      'mlm_weight': 1.0,
      'coref_res_weight': 1.0,
      'memory_pattern': table_path + '.npy',
      'memory_hash_pattern': hash_path + '.npy',
      'memory_reduction': 1,
      'max_mlm_targets': 25,
      'max_mentions': 5,
      'max_mention_targets': 3,
      'max_retrieval_indices': 12,
  }

  n_devices = 4
  table_size = 1024

  text_length = 128
  n_mentions = 5
  n_linked_mentions = 3

  def setUp(self):
    super().setUp()
    self.config = ml_collections.ConfigDict(self.config_dict)
    self.model_config = self.config.model_config
    self.encoder_config = self.model_config.encoder_config
    self.preprocess_fn = mauto_task.MautoTask.make_preprocess_fn(self.config)  # pylint: disable=line-too-long
    raw_examples = [
        test_utils.gen_mention_pretraining_sample(  # pylint: disable=g-complex-comprehension
            self.text_length,
            self.n_mentions,
            self.n_linked_mentions,
            max_length=self.encoder_config['max_length'],
        ) for _ in range(self.config.per_device_batch_size)
    ]

    processed_examples = [
        self.preprocess_fn(raw_example) for raw_example in raw_examples
    ]

    self.raw_batch = {
        key: tf.stack([example[key] for example in processed_examples
                      ]) for key in processed_examples[0]
    }

  def test_model_shape(self):
    # Construct retrieval table
    hash_values = np.random.randint(
        10000, size=(self.retrieval_table_size), dtype=np.int32)
    retrieval_table = np.random.rand(self.retrieval_table_size,
                                     self.retrieval_dim)

    # generate and save synthetic memory
    tf.io.gfile.makedirs(self.test_dir)
    np.save(self.table_path, retrieval_table)
    np.save(self.hash_path, hash_values)

    collate_fn = mauto_task.MautoTask.make_collater_fn(self.config)

    batch = collate_fn(self.raw_batch)
    batch = jax.tree.map(np.asarray, batch)

    model = mauto_task.MautoTask.build_model(self.model_config)

    rng = jax.random.PRNGKey(0)
    (loss_helpers, _), _ = model.init_with_output(rng, batch, True)

    self.assertEqual(
        loss_helpers['mlm_logits'].shape,
        (self.config.per_device_batch_size, self.config.max_mlm_targets,
         self.encoder_config.vocab_size))

    self.assertEqual(
        loss_helpers['target_mention_encodings'].shape,
        (self.config.max_mention_targets * self.config.per_device_batch_size,
         self.encoder_config.retrieval_dim))


if __name__ == '__main__':
  absltest.main()

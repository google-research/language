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
"""Tests for mauto task."""

import copy
import json

from absl.testing import absltest
import jax
from language.mentionmemory.encoders import mauto_encoder  # pylint: disable=unused-import
from language.mentionmemory.tasks import mauto_task
from language.mentionmemory.tasks import mention_memory_task
from language.mentionmemory.utils import mention_preprocess_utils
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import tensorflow as tf

# easiest to define as constant here
MENTION_SIZE = 2


class MautoTaskTest(test_utils.TestCase):
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
          'hidden_dim': 8,
          'retrieval_dim': retrieval_dim,
          'n_additional_concat_layers': 1,
          'n_pooled_layers': 1,
          'dropout_rate': 0.1
      },
      'max_length': 128,
      'max_positions': 512,
      'hidden_size': 4,
      'intermediate_dim': 8,
      'num_attention_heads': 1,
      'num_initial_layers': 1,
      'num_final_layers': 1,
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
      'max_mentions': mentions_per_sample,
      'max_mention_targets': linked_mentions_per_sample,
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

  def test_loss_fn(self):

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
    postprocess_fn = mauto_task.MautoTask.make_output_postprocess_fn(
        self.config)

    batch = collate_fn(self.raw_batch)
    batch = jax.tree.map(np.asarray, batch)

    model = mauto_task.MautoTask.build_model(self.model_config)

    rng = jax.random.PRNGKey(0)
    initial_parameters = model.init(rng, batch, True)
    loss_fn = mauto_task.MautoTask.make_loss_fn(self.config)
    _, metrics, auxiliary_output = loss_fn(
        model_config=self.model_config,
        model_params=initial_parameters['params'],
        model_vars={},
        batch=batch,
        deterministic=True,
    )

    self.assertEqual(metrics['mlm']['denominator'],
                     batch['mlm_target_weights'].sum())

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])

  def test_collater_fn(self):

    mm_collater = mention_memory_task.MentionMemoryTask.make_collater_fn(
        self.config)
    mm_batch = mm_collater(self.raw_batch)

    # Construct retrieval table
    text_identifiers = tf.gather(mm_batch['text_identifiers'],
                                 mm_batch['mention_target_indices'])
    start_positions = tf.gather(mm_batch['mention_start_positions'],
                                mm_batch['mention_target_indices'])
    batch_positions = tf.gather(mm_batch['mention_batch_positions'],
                                mm_batch['mention_target_indices'])

    hash_values = mention_preprocess_utils.modified_cantor_pairing(
        tf.cast(start_positions, tf.int64), text_identifiers)
    hash_values = tf.cast(hash_values, tf.int32)
    entity_ids = mm_batch['mention_target_ids']

    # Set hash values for one sample to max int value to test that collater
    # gracefully handles mentions that are not in table. Those mentions should
    # retrieve default value instead.
    first_sample_mask = tf.equal(batch_positions, 0)
    first_sample_indices = tf.where(first_sample_mask)
    first_sample_values = tf.fill(first_sample_indices.shape[0],
                                  np.iinfo(np.int32).max - 1)

    hash_values = tf.tensor_scatter_nd_update(hash_values, first_sample_indices,
                                              first_sample_values)

    retrieval_table = tf.constant(
        np.random.rand(len(start_positions), self.retrieval_dim))

    # generate and save synthetic memory
    tf.io.gfile.makedirs(self.test_dir)
    np.save(self.table_path, retrieval_table)
    np.save(self.hash_path, hash_values)
    np.save(self.entity_path, entity_ids)

    test_config = copy.deepcopy(self.config)
    test_config.memory_entity_pattern = self.entity_path + '.npy'
    collate_fn = mauto_task.MautoTask.make_collater_fn(test_config)

    dataset = tf.data.Dataset.from_tensor_slices(self.raw_batch)
    dataset = dataset.batch(self.config.per_device_batch_size)
    dataset = dataset.map(collate_fn, num_parallel_calls=2)
    batch = list(dataset)[0]

    collated_retrieval_values = batch['retrieval_mention_values'].numpy()

    # Check that retrievals of masked mentions have weight 0
    retrieval_mention_is_masked = batch['mention_target_is_masked'].numpy()[
        batch['mention_retrieval_indices'].numpy()]
    masked_mention_has_weight = retrieval_mention_is_masked * batch[
        'retrieval_mention_mask'].numpy()
    self.assertTrue(np.all(masked_mention_has_weight == 0))

    # Check that we retrieve default value with weight 0 for samples for which
    # we removed the hash from the table.
    default_retrieval_vector = retrieval_table.numpy()[0]
    indices_expected_default_retrieval = (
        batch['retrieval_mention_batch_positions'].numpy() == 0).nonzero()[0]
    self.assertTrue(
        np.all(
            np.expand_dims(default_retrieval_vector, axis=0) ==
            collated_retrieval_values[indices_expected_default_retrieval]))

    self.assertTrue(
        np.all(batch['retrieval_mention_mask'].numpy()
               [indices_expected_default_retrieval] == 0))

    # Check that for samples where we did not remove hash, we retrieve
    # corresponding value from table.
    retrieval_indices = (batch['retrieval_mention_batch_positions'].numpy() !=
                         0).nonzero()[0]
    expected_indices_present_in_retrieval = (batch_positions.numpy() !=
                                             0).nonzero()[0]

    # Indices are not consistent between these arrays, so we just have to check
    # that the same arrays are present overall regardless of order.
    reference_values = retrieval_table.numpy(
    )[expected_indices_present_in_retrieval]
    retrieval_values = collated_retrieval_values[retrieval_indices]
    expanded_equal = (
        np.expand_dims(reference_values,
                       1) == np.expand_dims(retrieval_values, 0))

    n_equal_values = np.sum(np.all(expanded_equal, axis=2), axis=1)
    self.assertTrue(np.all(n_equal_values == 1))


if __name__ == '__main__':
  absltest.main()

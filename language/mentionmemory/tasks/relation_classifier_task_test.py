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
"""Tests for Relation Classifier model and task functions."""

import copy
import json
import os


from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import relation_classifier_task
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import sklearn.metrics
import tensorflow.compat.v2 as tf



class TacredEvaluationTest(test_utils.TestCase):
  """Test to reproduce results on TACRED dataset."""

  def _get_test_data_path(self, file_name):
    path = os.path.join('language/mentionmemory/tasks/testdata/tacred',
                        file_name)
    return path

  def test_tacred_evaluation(self):
    targets_path = self._get_test_data_path('test.gold')
    with tf.io.gfile.GFile(targets_path, 'r') as targets_file:
      targets = [line.strip() for line in targets_file]
    predictions_path = self._get_test_data_path('spanbert_tacred_test.txt')
    with tf.io.gfile.GFile(predictions_path, 'r') as predictions_file:
      predictions = [line.strip() for line in predictions_file]

    labels_dict = {}
    for label in targets + predictions:
      if label not in labels_dict:
        new_index = len(labels_dict)
        labels_dict[label] = new_index

    labels_list = list(labels_dict.keys())
    labels_list.remove('no_relation')
    expected_precision, expected_recall, expected_f1, _ = sklearn.metrics.precision_recall_fscore_support(
        targets, predictions, labels=labels_list, average='micro')

    # See https://arxiv.org/abs/2004.14855, Table 5.
    self.assertAlmostEqual(expected_f1, 0.708, places=3)

    targets_array = jnp.asarray([labels_dict[x] for x in targets])
    predictions_array = jnp.asarray([labels_dict[x] for x in predictions])

    actual_tp, actual_fp, actual_fn = metric_utils.compute_tp_fp_fn_weighted(
        predictions_array, targets_array, jnp.ones_like(targets_array),
        labels_dict['no_relation'])
    actual_precision = actual_tp / (actual_tp + actual_fp)
    actual_recall = actual_tp / (actual_tp + actual_fn)
    actual_f1 = 2 * actual_precision * actual_recall / (
        actual_precision + actual_recall)
    self.assertAlmostEqual(actual_precision, expected_precision, places=8)
    self.assertAlmostEqual(actual_recall, expected_recall, places=8)
    self.assertAlmostEqual(actual_f1, expected_f1, places=8)


class RelationClassifierTaskTest(parameterized.TestCase):
  """Tests for RelationClassifierTask task."""

  encoder_config = {
      'dtype': 'bfloat16',
      'vocab_size': 1000,
      'max_positions': 128,
      'max_length': 128,
      'hidden_size': 4,
      'intermediate_dim': 8,
      'mention_encoding_dim': 4,
      'num_attention_heads': 2,
      'num_layers': 1,
      'dropout_rate': 0.1,
  }

  model_config = {
      'encoder_config': encoder_config,
      'encoder_name': 'bert',
      'dtype': 'bfloat16',
      'num_classes': 7,
      'num_layers': 2,
      'input_dim': 8,
      'hidden_dim': 9,
      'dropout_rate': 0.1,
  }

  config = {
      'model_config': model_config,
      'ignore_label': 0,
      'seed': 0,
  }

  def assertArrayEqual(self, expected, actual):
    expected = expected.ravel().tolist()
    actual = actual.ravel().tolist()
    self.assertSequenceEqual(expected, actual)

  def _gen_raw_sample(
      self, config):
    """Generate raw example."""

    features = {}

    # Generate text
    max_length = config.model_config.encoder_config.max_length
    features['text_ids'] = np.random.randint(
        low=1,
        high=config.model_config.encoder_config.vocab_size,
        size=(max_length),
        dtype=np.int64)
    features['text_mask'] = np.ones_like(features['text_ids'])

    # Generate labels
    features['target'] = np.random.randint(
        config.model_config.num_classes, size=1)

    # Generate mentions
    mention_positions = np.random.choice(
        max_length, size=(2 * config.max_mentions_per_sample), replace=False)
    mention_positions.sort()
    mention_mask = np.random.randint(
        2, size=(config.max_mentions_per_sample), dtype=np.int64)
    if mention_mask.sum() < 2:
      # There should be at least two mentions
      mention_mask[0] = 1
      mention_mask[1] = 1
    mention_start_positions = mention_positions[0::2] * mention_mask
    mention_end_positions = mention_positions[1::2] * mention_mask

    # Shuffle mentions
    p = np.random.permutation(config.max_mentions_per_sample)
    mention_start_positions = mention_start_positions[p]
    mention_end_positions = mention_end_positions[p]
    mention_mask = mention_mask[p]

    self.assertTrue(np.all(mention_start_positions[mention_mask == 0] == 0))
    self.assertTrue(np.all(mention_end_positions[mention_mask == 0] == 0))
    self.assertTrue(np.all(mention_mask[mention_mask == 0] == 0))

    features['mention_start_positions'] = mention_start_positions
    features['mention_end_positions'] = mention_end_positions
    features['mention_mask'] = mention_mask

    # Sample object and subject mentions
    mention_target_indices = np.random.choice(
        np.nonzero(mention_mask)[0], size=(2), replace=False)
    features['object_mention_indices'] = mention_target_indices[0]
    features['subject_mention_indices'] = mention_target_indices[1]

    return features

  def _gen_raw_batch(
      self, config):
    samples = [
        self._gen_raw_sample(config)
        for _ in range(config.per_device_batch_size)
    ]
    features = {}
    for feature_name in samples[0].keys():
      features[feature_name] = np.stack(
          [sample[feature_name] for sample in samples])
    return features

  @parameterized.parameters([
      (1, 2, 2, None),
      (1, 2, 2, 150),
      (2, 2, 2, None),
      (2, 2, 2, 150),
      (2, 3, 2, None),
      (2, 3, 2, 150),
      (5, 10, 2, None),
      (5, 10, 2, 150),
      (5, 10, 7, None),
      (5, 10, 7, 150),
      (10, 20, 10, None),
      (10, 20, 10, 170),
  ])
  def test_loss_fn(self, per_device_batch_size, max_mentions_per_sample,
                   max_mentions, max_length_with_entity_tokens):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    config['per_device_batch_size'] = per_device_batch_size
    config['max_mentions_per_sample'] = max_mentions_per_sample
    config['max_mentions'] = max_mentions
    config['max_length_with_entity_tokens'] = max_length_with_entity_tokens
    config = ml_collections.ConfigDict(config)

    raw_batch = self._gen_raw_batch(config)
    collater_fn = relation_classifier_task.RelationClassifierTask.make_collater_fn(
        config)
    postprocess_fn = relation_classifier_task.RelationClassifierTask.make_output_postprocess_fn(
        config)

    batch = collater_fn(raw_batch)
    batch = jax.tree_map(jnp.asarray, batch)

    self.assertSequenceEqual(batch['mention_target_weights'].shape,
                             [2 * config.per_device_batch_size])
    self.assertArrayEqual(batch['mention_target_weights'],
                          np.ones(2 * config.per_device_batch_size))
    self.assertSequenceEqual(batch['mention_target_batch_positions'].shape,
                             [2 * config.per_device_batch_size])
    self.assertArrayEqual(
        batch['mention_target_batch_positions'],
        np.repeat(np.arange(config.per_device_batch_size), [2]))

    # Check start / end positions are correctly preserved if entity tokens
    # are not used. Otherwise, positions might change.
    if max_length_with_entity_tokens is None:
      for index in range(config.per_device_batch_size):
        subj_index = raw_batch['subject_mention_indices'][index]
        obj_index = raw_batch['object_mention_indices'][index]
        self.assertEqual(
            batch['mention_target_start_positions'][2 * index],
            raw_batch['mention_start_positions'][index, subj_index])
        self.assertEqual(batch['mention_target_end_positions'][2 * index],
                         raw_batch['mention_end_positions'][index, subj_index])
        self.assertEqual(batch['mention_target_start_positions'][2 * index + 1],
                         raw_batch['mention_start_positions'][index, obj_index])
        self.assertEqual(batch['mention_target_end_positions'][2 * index + 1],
                         raw_batch['mention_end_positions'][index, obj_index])
    expected_mention_target_indices = np.arange(config.per_device_batch_size *
                                                2)
    self.assertArrayEqual(batch['mention_target_indices'],
                          expected_mention_target_indices)
    self.assertArrayEqual(batch['mention_subject_indices'],
                          expected_mention_target_indices[0::2])
    self.assertArrayEqual(batch['mention_object_indices'],
                          expected_mention_target_indices[1::2])

    model = relation_classifier_task.RelationClassifierTask.build_model(
        config.model_config)
    dummy_input = relation_classifier_task.RelationClassifierTask.dummy_input(
        config)
    init_rng = jax.random.PRNGKey(0)
    initial_parameters = model.init(init_rng, dummy_input, True)

    loss_fn = relation_classifier_task.RelationClassifierTask.make_loss_fn(
        config)
    _, metrics, auxiliary_output = loss_fn(config.model_config,
                                           initial_parameters['params'], {},
                                           batch, True)
    self.assertEqual(metrics['agg']['denominator'],
                     config.per_device_batch_size)

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])


if __name__ == '__main__':
  absltest.main()

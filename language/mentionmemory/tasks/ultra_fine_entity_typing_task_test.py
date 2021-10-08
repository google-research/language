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
"""Tests for Ultra Fine Entity Typing model and evaluation functions."""

import copy
import json
import os


from absl.testing import absltest
from absl.testing import parameterized
from flax.training import common_utils
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import ultra_fine_entity_typing_task
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf



class UltraFineEntityTypingMetricsTest(test_utils.TestCase):
  """Test metrics processing and computations."""

  def _get_test_data_path(self, file_name):
    path = os.path.join(
        'language/mentionmemory/tasks/testdata/'
        'ultra_fine_entity_typing', file_name)

    return path

  def assertNumpyArraysEqual(self, actual, expected):
    self.assertSequenceEqual(actual.tolist(), expected.tolist())

  def setUp(self):
    super().setUp()
    # See https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html
    # for the dataset and files description.
    # Model predictions are downloaded from
    # http://nlp.cs.washington.edu/entity_type/model/best_model.tar.gz
    vocab_path = self._get_test_data_path('types.txt')
    with tf.io.gfile.GFile(vocab_path, 'r') as vocab_file:
      self.label_vocab = {
          label.strip(): index for index, label in enumerate(vocab_file)
      }

    data_path = self._get_test_data_path('dev.json')
    with tf.io.gfile.GFile(data_path, 'rb') as data_file:
      data = json.load(data_file)

    size = (len(data), len(self.label_vocab))
    self.predictions = np.zeros(size, dtype=np.int16)
    self.labels = np.zeros(size, dtype=np.int16)
    for index, value in enumerate(data.values()):
      current_labels = np.array(
          [self.label_vocab[label] for label in value['gold']])
      self.labels[index, current_labels] = 1
      current_predictions = np.array(
          [self.label_vocab[label] for label in value['pred']])
      self.predictions[index, current_predictions] = 1

  def test_get_predictions(self):
    batch_size = 3
    size = (batch_size, 13)
    logits = np.random.uniform(low=-1, high=1, size=size)
    expected_predictions = (logits > 0).astype(np.int32)
    actual_predictions = ultra_fine_entity_typing_task.get_predictions(logits)
    self.assertNumpyArraysEqual(actual_predictions, expected_predictions)
    all_negative_logits = -np.abs(logits)
    expected_predictions = np.zeros(size, dtype=np.int32)
    expected_predictions[np.arange(batch_size),
                         np.argmax(all_negative_logits, axis=-1)] = 1
    actual_predictions = ultra_fine_entity_typing_task.get_predictions(
        all_negative_logits)
    self.assertNumpyArraysEqual(actual_predictions, expected_predictions)

  @parameterized.parameters(
      (0, 3, 10, 1),
      (1, 3, 10, 3),
      (2, 3, 10, 10),
      (3, 3, 10, 0),
      (4, 10, 50, 1),
      (5, 10, 50, 10),
      (6, 10, 50, 50),
      (7, 10, 50, 0),
  )
  def test_get_mrr(self, seed, batch_size, num_total_labels,
                   num_correct_labels):
    np.random.seed(seed)
    logits = np.random.random((batch_size, num_total_labels))
    labels = np.zeros((batch_size, num_total_labels))
    for i in range(batch_size):
      correct_labels = np.random.choice(
          num_total_labels, size=(num_correct_labels,), replace=False)
      labels[i, correct_labels] = 1
    actual_metrics = ultra_fine_entity_typing_task.get_mrr(labels, logits)

    expected_mrr, expected_denom = 0, 0

    for i in range(batch_size):
      if labels[i].sum() == 0:
        continue
      expected_denom += 1
      logits_labels_list = list(zip(logits[i], labels[i]))
      logits_labels_list.sort(reverse=True)
      current_mrr = []
      for j in range(num_total_labels):
        if logits_labels_list[j][1] == 1:
          current_mrr.append(1 / (1 + j))
      current_mrr = np.array(current_mrr)
      expected_mrr += current_mrr.mean()

    self.assertEqual(actual_metrics['denominator'], expected_denom)
    self.assertAlmostEqual(actual_metrics['value'], expected_mrr, places=4)

  @parameterized.parameters((1,), (2,), (10,))
  def test_reproduce_paper_evals(self, num_chunks):
    """Reproduce results from https://www.aclweb.org/anthology/P18-1009.pdf."""
    num_samples = self.labels.shape[0]
    chunk_size = num_samples // num_chunks
    metrics = []
    for chunk_start in range(0, num_samples, chunk_size):
      chunk_end = min(chunk_start + chunk_size, num_samples)
      labels = self.labels[chunk_start:chunk_end]
      predictions = self.predictions[chunk_start:chunk_end]
      current_metrics = ultra_fine_entity_typing_task.get_prediction_recall_metrics(
          labels, predictions)
      current_metrics = jax.tree_map(lambda x: jnp.expand_dims(x, 0),
                                     current_metrics)
      metrics.append(current_metrics)

    metrics = common_utils.get_metrics(metrics)
    metrics_sum = jax.tree_map(jnp.sum, metrics)
    processed_metrics = metric_utils.process_metrics(metrics_sum)
    self.assertAlmostEqual(
        processed_metrics['total_precision_value'], 0.481, places=3)
    self.assertAlmostEqual(
        processed_metrics['total_recall_value'], 0.232, places=3)
    self.assertAlmostEqual(
        processed_metrics['coarse_grained_precision_value'], 0.603, places=3)
    self.assertAlmostEqual(
        processed_metrics['coarse_grained_recall_value'], 0.616, places=3)
    self.assertAlmostEqual(
        processed_metrics['fine_grained_precision_value'], 0.404, places=3)
    self.assertAlmostEqual(
        processed_metrics['fine_grained_recall_value'], 0.384, places=3)
    self.assertAlmostEqual(
        processed_metrics['ultra_fine_grained_precision_value'],
        0.428,
        places=3)
    self.assertAlmostEqual(
        processed_metrics['ultra_fine_grained_recall_value'], 0.088, places=3)


class UltraFineEntityTypingTaskTest(parameterized.TestCase):
  """Tests for UltraFineEntityTyping task."""

  encoder_config = {
      'dtype': 'bfloat16',
      'vocab_size': 1000,
      'entity_vocab_size': 1000,
      'max_positions': 128,
      'max_length': 128,
      'hidden_size': 4,
      'intermediate_dim': 8,
      'entity_dim': 4,
      'num_attention_heads': 2,
      'num_initial_layers': 1,
      'num_final_layers': 1,
      'dropout_rate': 0.1,
  }

  model_config = {
      'encoder_config': encoder_config,
      'encoder_name': 'eae',
      'dtype': 'bfloat16',
  }

  config = {
      'model_config': model_config,
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
    num_classes = ultra_fine_entity_typing_task.NUM_CLASSES
    target = np.random.randint(
        num_classes, size=(config.max_num_labels_per_sample))
    target_mask = np.random.randint(2, size=(config.max_num_labels_per_sample))
    if target_mask.sum() == 0:
      # There should be at least one valid label
      target_mask[0] = 1
    target = target * target_mask
    features['target'] = target
    features['target_mask'] = target_mask

    # Generate mentions
    mention_positions = np.random.choice(
        max_length, size=(2 * config.max_mentions_per_sample), replace=False)
    mention_positions.sort()
    mention_mask = np.random.randint(
        2, size=(config.max_mentions_per_sample), dtype=np.int64)
    if mention_mask.sum() == 0:
      # There should be at least one valid label
      mention_mask[0] = 1
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
    features['mention_target_indices'] = np.random.choice(
        np.nonzero(mention_mask)[0], size=(1))

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
      (2, 24, 10, 2),
      (5, 10, 1, 1),
      (5, 10, 1, 30),
      (5, 10, 10, 1),
      (10, 20, 10, 5),
  ])
  def test_loss_fn(self, per_device_batch_size, max_mentions_per_sample,
                   max_mentions, max_num_labels_per_sample):
    """Test loss function runs and produces expected values."""
    config = copy.deepcopy(self.config)
    config['per_device_batch_size'] = per_device_batch_size
    config['max_mentions_per_sample'] = max_mentions_per_sample
    config['max_mentions'] = max_mentions
    config['max_num_labels_per_sample'] = max_num_labels_per_sample
    config = ml_collections.ConfigDict(config)

    raw_batch = self._gen_raw_batch(config)
    collater_fn = ultra_fine_entity_typing_task.UltraFineEntityTypingTask.make_collater_fn(
        config)
    postprocess_fn = ultra_fine_entity_typing_task.UltraFineEntityTypingTask.make_output_postprocess_fn(
        config)

    batch = collater_fn(raw_batch)
    batch = jax.tree_map(jnp.asarray, batch)

    self.assertSequenceEqual(batch['mention_target_weights'].shape,
                             [config.per_device_batch_size])
    self.assertSequenceEqual(batch['mention_target_batch_positions'].shape,
                             [config.per_device_batch_size])
    self.assertArrayEqual(batch['mention_target_batch_positions'],
                          np.arange(config.per_device_batch_size))

    self.assertSequenceEqual(raw_batch['mention_target_indices'].shape,
                             [config.per_device_batch_size, 1])
    expected_mention_target_indices = (np.arange(config.per_device_batch_size),
                                       raw_batch['mention_target_indices'][:,
                                                                           0])
    expected_mention_target_start_positions = raw_batch[
        'mention_start_positions'][expected_mention_target_indices]
    expected_mention_target_end_positions = raw_batch['mention_end_positions'][
        expected_mention_target_indices]
    self.assertArrayEqual(batch['mention_target_start_positions'],
                          expected_mention_target_start_positions)
    self.assertArrayEqual(batch['mention_target_end_positions'],
                          expected_mention_target_end_positions)

    model = ultra_fine_entity_typing_task.UltraFineEntityTypingTask.build_model(
        config.model_config)
    dummy_input = ultra_fine_entity_typing_task.UltraFineEntityTypingTask.dummy_input(
        config)
    init_rng = jax.random.PRNGKey(0)
    initial_parameters = model.init(init_rng, dummy_input, True)

    loss_fn = ultra_fine_entity_typing_task.UltraFineEntityTypingTask.make_loss_fn(
        config)
    _, metrics, auxiliary_output = loss_fn(config.model_config,
                                           initial_parameters['params'], {},
                                           batch, True)
    self.assertEqual(metrics['agg']['denominator'],
                     config.per_device_batch_size)
    self.assertEqual(metrics['agg_mrr']['denominator'],
                     config.per_device_batch_size)

    features = postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])


if __name__ == '__main__':
  absltest.main()

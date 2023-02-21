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
"""Tests for metric utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.utils import metric_utils
import numpy as np

_LARGE_NUMBER = 1e12


class ComputeMetricsTest(absltest.TestCase):
  """Test whether metrics computations produce expected values."""

  batch_size = 32
  seq_len = 20
  vocab_size = 100

  def test_logit_values_as_expected(self):
    """Test whether metrics computations produce expected values."""

    logits = np.random.rand(self.batch_size, self.seq_len, self.vocab_size)
    targets = np.random.randint(
        self.vocab_size, size=(self.batch_size, self.seq_len))
    dense_targets = jax.nn.one_hot(targets, self.vocab_size)
    weights = np.random.randint(2, size=(self.batch_size, self.seq_len))

    # Check loss and denominator make sense for random values
    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        logits,
        targets,
        weights,
    )

    expected_loss = -jax.nn.log_softmax(logits, axis=-1) * dense_targets
    expected_loss = (expected_loss * np.expand_dims(weights, axis=-1)).sum()
    self.assertAlmostEqual(loss, expected_loss, 1)
    self.assertAlmostEqual(denominator, weights.sum(), 1)

    # Check loss makes sense for uniform and degenerate scores
    logits = np.ones(shape=(self.batch_size, self.seq_len, self.vocab_size))
    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        logits,
        targets,
        weights,
    )
    expected_loss = np.log(self.vocab_size)
    self.assertAlmostEqual(loss / denominator, expected_loss, 4)

    logits = np.zeros(shape=(self.batch_size, self.seq_len, self.vocab_size))
    logits = logits + (
        _LARGE_NUMBER * dense_targets - _LARGE_NUMBER * (1 - dense_targets))

    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        logits,
        targets,
        weights,
    )
    self.assertAlmostEqual(loss / denominator, 0.0, 4)

  def test_prob_values_as_expected(self):

    probs = np.random.rand(self.batch_size, self.seq_len, self.vocab_size)
    targets = np.random.randint(
        self.vocab_size, size=(self.batch_size, self.seq_len))
    dense_targets = jax.nn.one_hot(targets, self.vocab_size)
    weights = np.random.randint(2, size=(self.batch_size, self.seq_len))

    # Check loss and denominator make sense with probs as inputs
    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        probs,
        targets,
        weights,
        inputs_are_prob=True,
    )

    expected_loss = -np.log(probs) * dense_targets
    expected_loss = (expected_loss * np.expand_dims(weights, axis=-1)).sum()
    self.assertAlmostEqual(loss, expected_loss, 1)
    self.assertAlmostEqual(denominator, weights.sum(), 1)

    # Check loss makes sense for uniform and degenerate probabilities

    probs = np.ones(shape=(self.batch_size, self.seq_len, self.vocab_size))
    probs = probs / self.vocab_size

    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        probs,
        targets,
        weights,
        inputs_are_prob=True,
    )

    expected_loss = np.log(self.vocab_size)
    self.assertAlmostEqual(loss / denominator, expected_loss, 4)

    probs = np.zeros(shape=(self.batch_size, self.seq_len, self.vocab_size))
    probs = probs + dense_targets

    loss, denominator = metric_utils.compute_weighted_cross_entropy(
        probs,
        targets,
        weights,
        inputs_are_prob=True,
    )

    self.assertAlmostEqual(loss / denominator, 0.0, 4)

  def test_accuracy_as_expected(self):
    logits = np.random.rand(self.batch_size, self.seq_len, self.vocab_size)
    targets = np.random.randint(
        self.vocab_size, size=(self.batch_size, self.seq_len))
    dense_targets = jax.nn.one_hot(targets, self.vocab_size)
    weights = np.random.randint(2, size=(self.batch_size, self.seq_len))

    # Check accuracy and denominator make sense

    logits = np.ones((self.batch_size, self.seq_len, self.vocab_size),
                     dtype=np.float32)
    correct = np.random.randint(2, size=(self.batch_size, self.seq_len, 1))
    logits = logits + dense_targets * (0.5 * correct - 0.5 * (1 - correct))

    acc, denominator = metric_utils.compute_weighted_accuracy(
        logits,
        targets,
        weights,
    )

    expected_accuracy = (np.squeeze(correct) * weights).sum() / weights.sum()
    self.assertAlmostEqual(acc / denominator, expected_accuracy, 1)
    self.assertAlmostEqual(denominator, weights.sum(), 1)


class ComputeCrossEntropyTest(parameterized.TestCase):
  """Test whether loss and metrics computations produce expected values."""

  @parameterized.parameters(
      (0, 1, 29, 31, 31),
      # Tests with large score values
      (1, 1000000, 29, 31),
      (2, 1000000, 29, 31),
      # Tests with large number of positive, negatives and neutral classes
      (3, 100, 29, 1001),
      (4, 100, 323, 31),
      # Tests whether lack of positives affects the numerical stability
      (5, 1, 29, 31, 1, 31),
      (6, 1, 29, 31, 0, 31),
      (7, 1, 29, 31, 31, 1),
      (8, 1, 29, 31, 31, 0),
      (9, 1, 29, 31, 1, 1),
      (10, 1, 29, 31, 0, 0),
      (11, 1000000, 29, 31, 0, 0),
      (12, 100, 29, 1001, 0, 0),
      (13, 100, 323, 31, 0, 0),
  )
  def test_loss_and_metrics_as_expected(self,
                                        seed,
                                        scale,
                                        local_n_mentions,
                                        global_n_mentions,
                                        max_num_positives=None,
                                        max_num_negatives=None):
    """Test whether loss and metrics computation produces expected values."""
    np.random.seed(seed)
    max_num_negatives = max_num_negatives or global_n_mentions
    max_num_positives = max_num_positives or global_n_mentions

    shape = (local_n_mentions, global_n_mentions)
    scores = np.random.random(shape) * scale

    num_positives = np.random.randint(
        max_num_positives + 1, size=(local_n_mentions))
    num_positives[0] = 0
    num_positives[-1] = global_n_mentions

    num_negatives = np.random.randint(
        max_num_negatives + 1, size=(local_n_mentions))
    num_negatives = np.minimum(num_negatives, global_n_mentions - num_positives)

    positives = np.zeros(shape, dtype=np.bool_)
    negatives = np.zeros(shape, dtype=np.bool_)

    for index in range(local_n_mentions):
      ids = np.random.choice(
          global_n_mentions,
          num_positives[index] + num_negatives[index],
          replace=False)
      positives[index, ids[:num_positives[index]]] = True
      negatives[index, ids[num_positives[index]:]] = True
    self.assertEqual(np.logical_and(positives, negatives).sum(), 0)

    weights = np.logical_and(num_positives > 0, num_negatives > 0)

    (actual_loss, actual_metrics, (actual_acc_per_sample,
                                   actual_weights_per_sample)
    ) = metric_utils.compute_cross_entropy_loss_with_positives_and_negatives_masks(
        scores, positives, negatives)

    expected_loss, expected_acc, expected_denom = 0, 0, 0
    expected_acc_per_sample = []
    # Consider every sample independently
    for i in range(local_n_mentions):
      if not weights[i]:
        expected_acc_per_sample.append(0)
        continue
      # Collect positive and negative scores
      positive_scores, negative_scores = [], []
      for j in range(global_n_mentions):
        if positives[i, j]:
          positive_scores.append(scores[i, j])
        if negatives[i, j]:
          negative_scores.append(scores[i, j])
      self.assertNotEmpty(positive_scores)
      self.assertNotEmpty(negative_scores)
      n_pos = len(positive_scores)
      max_negative_scores = max(negative_scores)
      current_loss, current_acc = 0, 0
      # Consider positive class per sample independently
      # and compute loss using a naive softmax op
      for pos_index in range(n_pos):
        current_scores = np.array([positive_scores[pos_index]] +
                                  negative_scores)
        current_scores = jax.nn.log_softmax(current_scores)
        current_loss += -current_scores[0]
        current_acc += int(positive_scores[pos_index] > max_negative_scores)
      expected_loss += current_loss / n_pos
      expected_acc += current_acc / n_pos
      expected_denom += 1
      expected_acc_per_sample.append(current_acc / n_pos)

    self.assertAlmostEqual(actual_loss, expected_loss, delta=1)
    self.assertAlmostEqual(actual_metrics['loss'], expected_loss, delta=1)
    self.assertAlmostEqual(actual_metrics['acc'], expected_acc, places=4)
    self.assertAlmostEqual(
        actual_metrics['denominator'], expected_denom, places=4)
    self.assertTrue(np.all(weights == actual_weights_per_sample))
    self.assertSequenceAlmostEqual(
        actual_acc_per_sample, expected_acc_per_sample, places=4)


class ComputeMetricsFromDuplicatesTest(absltest.TestCase):
  """Test whether metrics computation produces expected values."""

  batch_size = 32
  seq_len = 20
  num_items = 100
  num_classes = 200

  def test_values_as_expected(self):
    """Test whether metrics computation produces expected values."""
    probs = np.ones((self.batch_size, self.seq_len, self.num_items),
                    dtype=np.float32) / self.num_items

    classes = np.ones((self.batch_size, self.seq_len, self.num_items),
                      dtype=np.int32)
    targets = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
    weights = np.random.randint(2, size=(self.batch_size, self.seq_len))

    # Check case where all classes are targets
    loss, avg_prob, denominator = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
        probs,
        classes,
        targets,
        weights,
    )

    self.assertAlmostEqual(loss / denominator, 0.0, 4)
    self.assertAlmostEqual(avg_prob / denominator, 1.0, 4)
    self.assertAlmostEqual(denominator, weights.sum(), 4)

    # Check case where no classes are targets
    targets = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
    loss, avg_prob, denominator = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
        probs,
        classes,
        targets,
        weights,
    )

    self.assertAlmostEqual(avg_prob / denominator, 0.0, 4)

    # Check random cases
    classes = np.random.randint(
        self.num_classes, size=(self.batch_size, self.seq_len, self.num_items))
    targets = np.random.randint(
        self.num_classes, size=(self.batch_size, self.seq_len))

    loss, avg_prob, denominator = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
        probs,
        classes,
        targets,
        weights,
    )

    correct_probs = (classes == np.expand_dims(targets, axis=-1)) * probs
    expected_avg_prob = (
        correct_probs * np.expand_dims(weights, axis=-1)).sum() / weights.sum()

    self.assertAlmostEqual(avg_prob / denominator, expected_avg_prob, 4)


class ProcessMetricsTest(absltest.TestCase):
  """Test metrics processing."""

  def test_values_as_expected(self):
    """Test whether processed dictionaries match expected values."""

    metric_dict = {
        'cat1': {
            'key': 2.0,
            'denominator': 1.0
        },
        'cat2': {
            'key': 2.0,
            'denominator': 2.0
        },
    }

    processed_metrics = metric_utils.process_metrics(metric_dict)  # pytype: disable=wrong-arg-types  # jax-ndarray
    expected_result = {
        'cat1_key': 2.0,
        'cat1_denom': 1.0,
        'cat2_key': 1.0,
        'cat2_denom': 2.0,
    }
    self.assertEqual(processed_metrics, expected_result)

    metric_dict = {
        'cat1': {
            'key': 2.0,
            'denominator': 1.0
        },
        'cat2': {
            'key': 2.0,
            'denominator': 2.0
        },
    }

    processed_metrics = metric_utils.process_metrics(metric_dict, prefix='pref')  # pytype: disable=wrong-arg-types  # jax-ndarray
    expected_result = {
        'pref/cat1_key': 2.0,
        'pref/cat1_denom': 1.0,
        'pref/cat2_key': 1.0,
        'pref/cat2_denom': 2.0,
    }
    self.assertEqual(processed_metrics, expected_result)


class UpdateMetricsDTypeTest(absltest.TestCase):
  """Test metrics processing."""

  def test_types_as_expected(self):
    """Test whether updated metrics match expected types."""

    metric_dict = {
        'cat1': {
            'key': jnp.asarray([1], dtype=jnp.int32),
            'denominator': jnp.asarray([1], dtype=jnp.int16)
        },
        'cat2': {
            'key': 2.0,
            'denominator': jnp.asarray([1], dtype=jnp.bfloat16)
        },
    }
    processed_metrics = metric_utils.update_metrics_dtype(metric_dict)
    self.assertEqual(processed_metrics['cat1']['key'].dtype, jnp.float32)
    self.assertEqual(processed_metrics['cat1']['denominator'].dtype,
                     jnp.float32)
    self.assertIsInstance(processed_metrics['cat2']['key'], float)
    self.assertEqual(processed_metrics['cat2']['denominator'].dtype,
                     jnp.float32)


if __name__ == '__main__':
  absltest.main()

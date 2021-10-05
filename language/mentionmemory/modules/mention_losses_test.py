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
"""Tests for losses for mention encodings."""

import functools


from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import language.mentionmemory.modules.mention_losses as mention_losses
from language.mentionmemory.utils import test_utils
import numpy as np
import scipy.spatial


class NumCommonUniqueItemsTest(parameterized.TestCase):

  n_devices = 3

  def assertArrayEqual(self, expected, actual):
    expected = expected.ravel().tolist()
    actual = actual.ravel().tolist()
    self.assertSequenceEqual(expected, actual)

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)
    self.devices = jax.local_devices()

  @parameterized.named_parameters(
      dict(
          testcase_name='same_positions_different_ids',
          batch_positions=[0, 0, 0],
          ids=[1, 2, 3],
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='same_positions_same_ids',
          batch_positions=[0, 0, 0],
          ids=[1, 1, 1],
          expected=[1, 0, 0],
      ),
      dict(
          testcase_name='different_positions_same_ids',
          batch_positions=[1, 2, 3],
          ids=[1, 1, 1],
          expected=[1, 1, 1],
      ),
      dict(
          testcase_name='different_positions_different_ids',
          batch_positions=[1, 2, 3],
          ids=[1, 2, 3],
          expected=[1, 2, 3],
      ),
  )
  def test_mask_duplicate_ids(self, batch_positions, ids, expected):
    batch_positions = jnp.asarray(batch_positions)
    ids = jnp.asarray(ids)
    expected = jnp.asarray(expected)
    actual = mention_losses.mask_duplicate_ids(batch_positions, ids)
    self.assertArrayEqual(expected, actual)

  @parameterized.named_parameters(
      dict(
          testcase_name='all_same',
          batch=[[1, 1], [1, 1], [1, 1]],
          expected=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
      ),
      dict(
          testcase_name='all_common',
          batch=[[1, 2], [1, 2], [1, 2]],
          expected=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
      ),
      dict(
          testcase_name='all_unique',
          batch=[[1, 2], [3, 4], [5, 6]],
          expected=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
      ),
      dict(
          testcase_name='simple_1',
          batch=[[0, 1, 2, 1], [1, 3, 0, 0]],
          expected=[[2, 1], [1, 2]],
      ),
  )
  def test_get_num_common_unique_items_single_device(self,
                                                     batch,
                                                     expected):
    batch_positions, ids = [], []
    for batch_index, sample in enumerate(batch):
      for x in sample:
        batch_positions.append(batch_index)
        ids.append(x)
    self.assertEqual(len(ids), len(batch_positions))
    batch_positions = jnp.asarray(batch_positions)
    ids = jnp.asarray(ids)
    actual, _ = mention_losses.get_num_common_unique_items(
        batch_positions, ids, len(batch))
    self.assertArrayEqual(jnp.asarray(expected), actual)

  @parameterized.parameters(
      # All PADs mentions (vocab only contains PAD, so its size is 1)
      (0, 2, 2, 1),
      (1, 5, 5, 1),
      # Single mention per batch (n_mentions is 1)
      (2, 1, 1, 5),
      (3, 5, 1, 5),
      # Single passage per batch (batch_size is 1)
      (4, 1, 1, 5),
      (5, 1, 5, 5),
      # Batch size is bigger than number of mentions
      (6, 5, 2, 5),
      (7, 10, 5, 5),
      # Number of mentions is bigger than batch size
      (8, 2, 5, 5),
      (9, 5, 10, 5),
      # Examples with larger size
      (10, 20, 20, 5),
      (11, 20, 20, 20),
  )
  def test_get_num_common_unique_items_multi_devices(self, seed,
                                                     batch_size,
                                                     n_mentions,
                                                     vocab_size):
    np.random.seed(seed)

    batch_positions_sharded = jax.device_put_sharded(
        list(np.random.randint(batch_size, size=(self.n_devices, n_mentions))),
        self.devices)
    ids_sharded = jax.device_put_sharded(
        list(np.random.randint(vocab_size, size=(self.n_devices, n_mentions))),
        self.devices)

    fn = functools.partial(
        mention_losses.get_num_common_unique_items, batch_size=batch_size)

    actual_per_sample, actual_per_mention = jax.pmap(
        fn, axis_name='batch')(batch_positions_sharded, ids_sharded)

    batches = []
    for i in range(self.n_devices):
      batches.append([])
      for j in range(batch_size):
        batches[i].append(set())
      for j in range(n_mentions):
        if ids_sharded[i, j] > 0:
          batches[i][batch_positions_sharded[i, j]].add(ids_sharded[i,
                                                                    j].item())

    for d_i in range(self.n_devices):
      for b_i in range(batch_size):
        for d_j in range(self.n_devices):
          for b_j in range(batch_size):
            self.assertLen(
                batches[d_i][b_i].intersection(batches[d_j][b_j]),
                actual_per_sample[d_i, b_i, d_j * batch_size + b_j].item())

    for d_i in range(self.n_devices):
      for m_i in range(n_mentions):
        b_i = batch_positions_sharded[d_i, m_i].item()
        for d_j in range(self.n_devices):
          for m_j in range(n_mentions):
            b_j = batch_positions_sharded[d_j, m_j].item()
            self.assertLen(
                batches[d_i][b_i].intersection(batches[d_j][b_j]),
                actual_per_mention[d_i, m_i, d_j * n_mentions + m_j].item())


class MentionLossesTest(parameterized.TestCase):
  """Mention encoding losses tests."""

  entity_vocab_size = 100
  hidden_size = 3

  n_mentions = 37
  n_devices = 3
  batch_size = 5

  metrics_prefix = 'test_'

  def _gen_array(self, gen_fn):
    array = [gen_fn() for _ in range(len(self.devices))]
    array_sharded = jax.device_put_sharded(array, self.devices)
    array_stacked = np.stack(array)
    array_stacked = array_stacked.reshape([-1] + list(array_stacked.shape[2:]))
    return array_stacked, array_sharded

  def assertArrayEqual(self, expected, actual):
    expected = expected.ravel().tolist()
    actual = actual.ravel().tolist()
    self.assertSequenceEqual(expected, actual)

  def assertArrayAlmostEqual(self, expected, actual, places):
    expected = expected.ravel().tolist()
    actual = actual.ravel().tolist()
    self.assertSequenceAlmostEqual(expected, actual, places=places)

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)
    self.devices = jax.local_devices()
    # pylint: disable=g-long-lambda
    (self.mention_encodings_stacked,
     self.mention_encodings_sharded) = self._gen_array(
         lambda: 10.0 * np.random.random((self.n_mentions, self.hidden_size)))
    (self.mention_target_ids_stacked, self.mention_target_ids_sharded
    ) = self._gen_array(lambda: np.random.randint(
        self.entity_vocab_size, size=(self.n_mentions)))
    (self.mention_batch_positions_stacked,
     self.mention_batch_positions_sharded) = self._gen_array(
         lambda: np.random.randint(self.batch_size, size=(self.n_mentions)))
    (self.mention_target_is_masked_stacked,
     self.mention_target_is_masked_sharded
    ) = self._gen_array(lambda: np.random.randint(2, size=(self.n_mentions)))

  @parameterized.parameters((0, 5, 3), (1, 10, 1), (2, 10, 10))
  def test_get_globally_consistent_batch_positions(self, seed, batch_size,
                                                   n_mentions):
    np.random.seed(seed)

    mention_batch_positions_sharded = jax.device_put_sharded(
        list(np.random.randint(batch_size, size=(self.n_devices, n_mentions))),
        self.devices)

    fn = functools.partial(
        mention_losses.get_globally_consistent_batch_positions,
        batch_size=batch_size)

    # Test function in the multi-device setting
    (local_mention_batch_positions, global_mention_batch_positions) = jax.pmap(
        fn, axis_name='batch')(
            mention_batch_positions_sharded)

    for i in range(self.n_devices):
      self.assertArrayEqual(local_mention_batch_positions[i],
                            mention_batch_positions_sharded[i] + batch_size * i)

    local_mention_batch_positions = local_mention_batch_positions.reshape(-1)
    for i in range(self.n_devices):
      self.assertArrayEqual(global_mention_batch_positions[i],
                            local_mention_batch_positions)

    # Test function in the single-device setting
    for i in range(self.n_devices):
      (local_mention_batch_positions,
       global_mention_batch_positions) = fn(mention_batch_positions_sharded[i])
      self.assertArrayEqual(local_mention_batch_positions,
                            mention_batch_positions_sharded[i])
      self.assertArrayEqual(global_mention_batch_positions,
                            mention_batch_positions_sharded[i])

  def test_build_coref_positive_negative_mask(self):
    all_mention_target_ids = jax.device_put_replicated(
        self.mention_target_ids_stacked, self.devices)

    get_batch_positions = functools.partial(
        mention_losses.get_globally_consistent_batch_positions,
        batch_size=self.batch_size)
    get_batch_positions = jax.pmap(get_batch_positions, axis_name='batch')

    (local_mention_batch_positions,
     global_mention_batch_positions) = get_batch_positions(
         self.mention_batch_positions_sharded)

    (positive_mask, negative_mask) = jax.pmap(
        mention_losses.build_coref_positive_negative_mask,
        axis_name='batch')(local_mention_batch_positions,
                           global_mention_batch_positions,
                           self.mention_target_ids_sharded,
                           all_mention_target_ids)

    n_all_mentions = self.n_mentions * self.n_devices
    self.assertSequenceEqual(positive_mask.shape, negative_mask.shape)
    self.assertSequenceEqual(positive_mask.shape,
                             (self.n_devices, self.n_mentions, n_all_mentions))
    positive_mask = positive_mask.reshape(-1, n_all_mentions)
    negative_mask = negative_mask.reshape(-1, n_all_mentions)

    for i in range(n_all_mentions):
      for j in range(n_all_mentions):
        is_same_device = i // self.n_mentions == j // self.n_mentions
        is_same_passage = (
            self.mention_batch_positions_stacked[i] ==
            self.mention_batch_positions_stacked[j])
        is_same_passage = is_same_passage and is_same_device

        if (self.mention_target_ids_stacked[i] == 0 or
            self.mention_target_ids_stacked[j] == 0 or is_same_passage):
          self.assertEqual(positive_mask[i, j], 0)
          self.assertEqual(negative_mask[i, j], 0)
          continue

        self.assertEqual(
            positive_mask[i, j], self.mention_target_ids_stacked[i] ==
            self.mention_target_ids_stacked[j])
        self.assertEqual(
            negative_mask[i, j], self.mention_target_ids_stacked[i] !=
            self.mention_target_ids_stacked[j])

  @parameterized.parameters(('dot',), ('cos'), ('dot_sqrt'))
  def test_coref_resolution_loss_multiple_devices(self, mode):
    """Testing coreference resolution loss."""

    def compute_loss(mention_encodings, mention_batch_positions,
                     mention_target_is_masked, mention_ids):
      return mention_losses.coreference_resolution_loss(
          mention_encodings, mention_batch_positions, mention_ids,
          self.batch_size, mode, mention_target_is_masked, self.metrics_prefix)

    loss_sharded, metrics_sharded = jax.pmap(
        compute_loss, axis_name='batch')(self.mention_encodings_sharded,
                                         self.mention_batch_positions_sharded,
                                         self.mention_target_is_masked_sharded,
                                         self.mention_target_ids_sharded)

    num_total_mentions, hidden_dim = self.mention_encodings_stacked.shape
    scores = np.zeros((num_total_mentions, num_total_mentions))
    total_avg_scores, total_unnorm_avg_scores = [], []
    for i in range(num_total_mentions):
      current_avg_scores = []
      current_unnorm_avg_scores = []
      for j in range(num_total_mentions):
        if mode == 'dot':
          scores[i, j] = np.dot(self.mention_encodings_stacked[i],
                                self.mention_encodings_stacked[j])
        elif mode == 'dot_sqrt':
          scores[i, j] = np.dot(
              self.mention_encodings_stacked[i],
              self.mention_encodings_stacked[j]) / np.sqrt(hidden_dim)
        elif mode == 'cos':
          scores[i, j] = 1 - scipy.spatial.distance.cosine(
              self.mention_encodings_stacked[i],
              self.mention_encodings_stacked[j])
        else:
          raise ValueError('Unknown coreference resolution mode: ' + mode)
        if self.mention_target_ids_stacked[j] != 0:
          current_avg_scores.append(scores[i, j])
          current_unnorm_avg_scores.append(
              np.dot(self.mention_encodings_stacked[i],
                     self.mention_encodings_stacked[j]))
      # pylint: disable=g-explicit-length-test
      if len(current_avg_scores) > 0:
        current_avg_scores = np.array(current_avg_scores)
        total_avg_scores.append(current_avg_scores.mean())
        current_unnorm_avg_scores = np.array(current_unnorm_avg_scores)
        total_unnorm_avg_scores.append(current_unnorm_avg_scores.mean())
      else:
        total_avg_scores.append(0)
        total_unnorm_avg_scores.append(0)
    self.assertLen(total_avg_scores, len(self.mention_target_ids_stacked))

    expected_loss, expected_acc, expected_denom = 0, 0, 0
    expected_denom_masked, expected_denom_non_masked = 0, 0
    expected_acc_masked, expected_acc_non_masked = 0, 0
    expected_n_positives, expected_n_negatives = 0, 0
    expected_avg_scores, expected_unnorm_avg_scores = 0, 0
    expected_avg_norms = 0
    for i in range(len(self.mention_target_ids_stacked)):
      if self.mention_target_ids_stacked[i] == 0:
        continue
      positive_scores, negative_scores = [], []
      for j in range(len(self.mention_target_ids_stacked)):
        if self.mention_target_ids_stacked[j] == 0:
          continue
        is_same_device = i // self.n_mentions == j // self.n_mentions
        is_same_passage = (
            self.mention_batch_positions_stacked[i] ==
            self.mention_batch_positions_stacked[j])
        is_same_passage = is_same_passage and is_same_device
        if is_same_passage:
          continue
        if (self.mention_target_ids_stacked[i] ==
            self.mention_target_ids_stacked[j]):
          positive_scores.append(scores[i, j])
        else:
          negative_scores.append(scores[i, j])
      n_pos = len(positive_scores)
      n_neg = len(negative_scores)
      max_negative_scores = max(negative_scores)
      if n_pos == 0 or n_neg == 0:
        continue
      current_loss, current_acc = 0, 0
      for pos_index in range(n_pos):
        current_scores = np.array([positive_scores[pos_index]] +
                                  negative_scores)
        current_scores = jax.nn.log_softmax(current_scores)
        current_loss += -current_scores[0]
        current_acc += int(positive_scores[pos_index] > max_negative_scores)

      expected_loss += current_loss / n_pos
      expected_acc += current_acc / n_pos
      expected_denom += 1
      if self.mention_target_is_masked_stacked[i] > 0:
        expected_denom_masked += 1
        expected_acc_masked += current_acc / n_pos
      else:
        expected_denom_non_masked += 1
        expected_acc_non_masked += current_acc / n_pos

      expected_n_positives += n_pos
      expected_n_negatives += n_neg
      expected_avg_scores += total_avg_scores[i]
      expected_unnorm_avg_scores += total_unnorm_avg_scores[i]
      expected_avg_norms += np.linalg.norm(self.mention_encodings_stacked[i])

    metrics_sharded = jax.tree_map(jnp.sum, metrics_sharded)
    metrics_sharded_masked = metrics_sharded[self.metrics_prefix +
                                             'coref_resolution_masked']
    metrics_sharded_non_masked = metrics_sharded[self.metrics_prefix +
                                                 'coref_resolution_non_masked']
    metrics_sharded = metrics_sharded[self.metrics_prefix + 'coref_resolution']
    loss_sharded = jnp.sum(loss_sharded)

    self.assertAlmostEqual(loss_sharded, expected_loss, places=2)
    self.assertAlmostEqual(metrics_sharded['loss'], expected_loss, places=2)
    self.assertAlmostEqual(metrics_sharded['acc'], expected_acc, places=3)
    self.assertEqual(metrics_sharded['denominator'], expected_denom)
    self.assertEqual(metrics_sharded['n_positive'], expected_n_positives)
    self.assertEqual(metrics_sharded['n_negative'], expected_n_negatives)
    self.assertAlmostEqual(
        metrics_sharded['avg_score'], expected_avg_scores, places=2)
    self.assertAlmostEqual(
        metrics_sharded['avg_unnorm_score'],
        expected_unnorm_avg_scores,
        places=2)
    self.assertAlmostEqual(
        metrics_sharded['avg_norm'], expected_avg_norms, places=2)

    self.assertAlmostEqual(
        metrics_sharded_masked['acc'], expected_acc_masked, places=3)
    self.assertEqual(metrics_sharded_masked['denominator'],
                     expected_denom_masked)
    self.assertAlmostEqual(
        metrics_sharded_non_masked['acc'], expected_acc_non_masked, places=3)
    self.assertEqual(metrics_sharded_non_masked['denominator'],
                     expected_denom_non_masked)

  def test_coref_resolution_loss_multiple_vs_single_devices(self):
    """Comparing coreference resolution loss on multiple vs single devices."""

    def compute_loss(mention_encodings, mention_batch_positions, mention_ids,
                     mention_target_is_masked):
      return mention_losses.coreference_resolution_loss(
          mention_encodings, mention_batch_positions, mention_ids,
          self.batch_size, 'dot', mention_target_is_masked, self.metrics_prefix)

    loss_sharded, metrics_sharded = jax.pmap(
        compute_loss, axis_name='batch')(self.mention_encodings_sharded,
                                         self.mention_batch_positions_sharded,
                                         self.mention_target_ids_sharded,
                                         self.mention_target_is_masked_sharded)

    mention_batch_positions_stacked = (
        self.mention_batch_positions_stacked.reshape(self.n_devices, -1))
    mention_batch_positions_stacked = mention_batch_positions_stacked.copy()
    mention_batch_positions_stacked += (
        np.expand_dims(np.arange(self.n_devices), 1) * self.batch_size)
    mention_batch_positions_stacked = mention_batch_positions_stacked.reshape(
        -1)
    loss_stacked, metrics_stacked = compute_loss(
        self.mention_encodings_stacked, mention_batch_positions_stacked,
        self.mention_target_ids_stacked, self.mention_target_is_masked_stacked)

    loss_sharded = jnp.sum(loss_sharded)
    metrics_sharded = jax.tree_map(jnp.sum, metrics_sharded)

    self.assertAlmostEqual(loss_sharded, loss_stacked, places=2)
    for metric_group_name in metrics_stacked:
      for metric_name in metrics_stacked[metric_group_name]:
        self.assertAlmostEqual(
            metrics_sharded[metric_group_name][metric_name],
            metrics_stacked[metric_group_name][metric_name],
            places=2)

  @parameterized.parameters(('dot',), ('cos'), ('dot_sqrt'))
  def test_mtb_loss_multiple_devices(self, mode):
    """Testing MTB loss."""

    def compute_loss(mention_encodings, mention_batch_positions, mention_ids,
                     mention_target_is_masked):
      return mention_losses.mtb_loss(mention_encodings, mention_batch_positions,
                                     mention_ids, self.batch_size, mode,
                                     mention_target_is_masked,
                                     self.metrics_prefix)

    loss_sharded, metrics_sharded = jax.pmap(
        compute_loss, axis_name='batch')(self.mention_encodings_sharded,
                                         self.mention_batch_positions_sharded,
                                         self.mention_target_ids_sharded,
                                         self.mention_target_is_masked_sharded)

    batches = []
    for i in range(self.n_devices):
      batches.append([])
      for j in range(self.batch_size):
        batches[i].append(set())
      for j in range(self.n_mentions):
        if self.mention_target_ids_sharded[i, j] > 0:
          batches[i][self.mention_batch_positions_sharded[i, j]].add(
              self.mention_target_ids_sharded[i, j].item())

    num_total_mentions, hidden_dim = self.mention_encodings_stacked.shape

    # Compute the scores between mentions.
    scores = np.zeros((num_total_mentions, num_total_mentions))
    total_avg_scores, total_unnorm_avg_scores = [], []
    for i in range(num_total_mentions):
      current_avg_scores = []
      current_unnorm_avg_scores = []
      for j in range(num_total_mentions):
        if mode == 'dot':
          scores[i, j] = np.dot(self.mention_encodings_stacked[i],
                                self.mention_encodings_stacked[j])
        elif mode == 'dot_sqrt':
          scores[i, j] = np.dot(
              self.mention_encodings_stacked[i],
              self.mention_encodings_stacked[j]) / np.sqrt(hidden_dim)
        elif mode == 'cos':
          scores[i, j] = 1 - scipy.spatial.distance.cosine(
              self.mention_encodings_stacked[i],
              self.mention_encodings_stacked[j])
        else:
          raise ValueError('Unknown coreference resolution mode: ' + mode)
        if self.mention_target_ids_stacked[j] != 0:
          current_avg_scores.append(scores[i, j])
          current_unnorm_avg_scores.append(
              np.dot(self.mention_encodings_stacked[i],
                     self.mention_encodings_stacked[j]))
      # pylint: disable=g-explicit-length-test
      if len(current_avg_scores) > 0:
        current_avg_scores = np.array(current_avg_scores)
        total_avg_scores.append(current_avg_scores.mean())
        current_unnorm_avg_scores = np.array(current_unnorm_avg_scores)
        total_unnorm_avg_scores.append(current_unnorm_avg_scores.mean())
      else:
        total_avg_scores.append(0)
        total_unnorm_avg_scores.append(0)
    self.assertLen(total_avg_scores, len(self.mention_target_ids_stacked))

    # Compute the loss and metrics.
    expected_loss, expected_acc, expected_denom = 0, 0, 0
    expected_n_positives, expected_n_negatives = 0, 0
    expected_n_hard_negatives = 0
    expected_avg_scores, expected_unnorm_avg_scores = 0, 0
    expected_denom_masked, expected_denom_non_masked = 0, 0
    expected_acc_masked, expected_acc_non_masked = 0, 0
    expected_avg_norms = 0
    for i in range(len(self.mention_target_ids_stacked)):
      if self.mention_target_ids_stacked[i] == 0:
        continue
      device_i = i // self.n_mentions
      unique_entities_i = (
          batches[device_i][self.mention_batch_positions_sharded[
              device_i, i % self.n_mentions]])

      positive_scores, hard_negative_scores, negative_scores = [], [], []
      for j in range(len(self.mention_target_ids_stacked)):
        if self.mention_target_ids_stacked[j] == 0:
          continue
        device_j = j // self.n_mentions
        is_same_device = device_i == device_j
        is_same_passage = (
            self.mention_batch_positions_stacked[i] ==
            self.mention_batch_positions_stacked[j])
        is_same_passage = is_same_passage and is_same_device
        if is_same_passage:
          continue
        if (self.mention_target_ids_stacked[i] ==
            self.mention_target_ids_stacked[j]):
          unique_entities_j = (
              batches[device_j][self.mention_batch_positions_sharded[
                  device_j, j % self.n_mentions]])
          num_common_entities = len(
              unique_entities_i.intersection(unique_entities_j))
          if num_common_entities >= 2:
            positive_scores.append(scores[i, j])
          else:
            hard_negative_scores.append(scores[i, j])
        else:
          negative_scores.append(scores[i, j])
      negative_scores = negative_scores + hard_negative_scores
      n_pos = len(positive_scores)
      n_neg = len(negative_scores)
      n_hard_neg = len(hard_negative_scores)
      max_negative_scores = max(negative_scores)
      if n_pos == 0 or n_hard_neg == 0:
        continue
      current_loss, current_acc = 0, 0
      for pos_index in range(n_pos):
        current_scores = np.array([positive_scores[pos_index]] +
                                  negative_scores)
        current_scores = jax.nn.log_softmax(current_scores)
        current_loss += -current_scores[0]
        current_acc += int(positive_scores[pos_index] > max_negative_scores)

      expected_loss += current_loss / n_pos
      expected_acc += current_acc / n_pos
      expected_denom += 1
      if self.mention_target_is_masked_stacked[i] > 0:
        expected_denom_masked += 1
        expected_acc_masked += current_acc / n_pos
      else:
        expected_denom_non_masked += 1
        expected_acc_non_masked += current_acc / n_pos
      expected_n_positives += n_pos
      expected_n_negatives += n_neg
      expected_n_hard_negatives += n_hard_neg
      expected_avg_scores += total_avg_scores[i]
      expected_unnorm_avg_scores += total_unnorm_avg_scores[i]
      expected_avg_norms += np.linalg.norm(self.mention_encodings_stacked[i])

    metrics_sharded = jax.tree_map(jnp.sum, metrics_sharded)
    metrics_sharded_masked = metrics_sharded[self.metrics_prefix + 'mtb_masked']
    metrics_sharded_non_masked = metrics_sharded[self.metrics_prefix +
                                                 'mtb_non_masked']
    metrics_sharded = metrics_sharded[self.metrics_prefix + 'mtb']

    loss_sharded = jnp.sum(loss_sharded)
    self.assertAlmostEqual(loss_sharded, expected_loss, places=2)
    self.assertAlmostEqual(metrics_sharded['loss'], expected_loss, places=2)
    self.assertAlmostEqual(metrics_sharded['acc'], expected_acc, places=3)
    self.assertEqual(metrics_sharded['denominator'], expected_denom)
    self.assertEqual(metrics_sharded['n_positive'], expected_n_positives)
    self.assertEqual(metrics_sharded['n_negative'], expected_n_negatives)
    self.assertEqual(metrics_sharded['n_hard_negative'],
                     expected_n_hard_negatives)
    self.assertAlmostEqual(
        metrics_sharded['avg_score'], expected_avg_scores, places=2)
    self.assertAlmostEqual(
        metrics_sharded['avg_unnorm_score'],
        expected_unnorm_avg_scores,
        places=2)
    self.assertAlmostEqual(
        metrics_sharded['avg_norm'], expected_avg_norms, places=2)
    self.assertAlmostEqual(
        metrics_sharded_masked['acc'], expected_acc_masked, places=3)
    self.assertEqual(metrics_sharded_masked['denominator'],
                     expected_denom_masked)
    self.assertAlmostEqual(
        metrics_sharded_non_masked['acc'], expected_acc_non_masked, places=3)
    self.assertEqual(metrics_sharded_non_masked['denominator'],
                     expected_denom_non_masked)

  def test_mtb_loss_multiple_vs_single_devices(self):
    """Comparing MTB loss on multiple vs single devices."""

    def loss_fn_multi_device(mention_encodings, mention_batch_positions,
                             mention_ids, mention_target_is_masked):
      return mention_losses.mtb_loss(mention_encodings, mention_batch_positions,
                                     mention_ids, self.batch_size, 'dot',
                                     mention_target_is_masked,
                                     self.metrics_prefix)

    loss_sharded, metrics_sharded = jax.pmap(
        loss_fn_multi_device,
        axis_name='batch')(self.mention_encodings_sharded,
                           self.mention_batch_positions_sharded,
                           self.mention_target_ids_sharded,
                           self.mention_target_is_masked_sharded)

    mention_batch_positions_stacked = (
        self.mention_batch_positions_stacked.reshape(self.n_devices, -1))
    mention_batch_positions_stacked = mention_batch_positions_stacked.copy()
    mention_batch_positions_stacked += (
        np.expand_dims(np.arange(self.n_devices), 1) * self.batch_size)
    mention_batch_positions_stacked = mention_batch_positions_stacked.reshape(
        -1)
    loss_stacked, metrics_stacked = mention_losses.mtb_loss(
        self.mention_encodings_stacked, mention_batch_positions_stacked,
        self.mention_target_ids_stacked, self.batch_size * self.n_devices,
        'dot', self.mention_target_is_masked_stacked, self.metrics_prefix)

    loss_sharded = jnp.sum(loss_sharded)
    metrics_sharded = jax.tree_map(jnp.sum, metrics_sharded)

    self.assertAlmostEqual(loss_sharded, loss_stacked, places=2)
    for metric_group_name in metrics_stacked:
      for metric_name in metrics_stacked[metric_group_name]:
        self.assertAlmostEqual(
            metrics_sharded[metric_group_name][metric_name],
            metrics_stacked[metric_group_name][metric_name],
            places=2)

  @parameterized.parameters(('dot',), ('cos'), ('dot_sqrt'))
  def test_entity_linking_loss(self, mode):
    n_mentions = 5
    n_entities = 10
    hidden_size = 3
    mention_encodings = np.random.random((n_mentions, hidden_size))
    entity_embeddings = np.random.random((n_entities, hidden_size))
    mention_target_ids = np.random.randint(n_entities, size=(n_mentions))
    mention_target_weights = np.random.randint(2, size=(n_mentions))

    (actual_loss, actual_metrics,
     (actual_acc_per_mention,
      actual_weight_per_mention)) = mention_losses.entity_linking_loss(
          mention_encodings, entity_embeddings, mention_target_ids,
          mention_target_weights, mode)

    # Simple consistency checks
    self.assertArrayEqual(mention_target_weights, actual_weight_per_mention)
    self.assertEqual(actual_metrics['loss'], actual_loss)
    self.assertAlmostEqual(
        actual_metrics['acc'], actual_acc_per_mention.sum(), places=6)
    self.assertAlmostEqual(
        actual_metrics['denominator'], mention_target_weights.sum(), places=8)

    scores = np.matmul(mention_encodings, np.transpose(entity_embeddings))
    if mode == 'dot_sqrt':
      scores /= np.sqrt(hidden_size)
    if mode == 'cos':
      scores /= np.expand_dims(np.linalg.norm(mention_encodings, axis=-1), 1)
      scores /= np.expand_dims(np.linalg.norm(entity_embeddings, axis=-1), 0)

    log_probs = np.log(scipy.special.softmax(scores, axis=-1))
    expected_loss = 0
    expected_acc_per_mention = []
    expected_cos_sim_per_mention = []
    for i in range(n_mentions):
      if mention_target_weights[i] == 1:
        expected_loss += -log_probs[i, mention_target_ids[i]]
        is_correct = int(np.argmax(log_probs[i]) == mention_target_ids[i])
        expected_acc_per_mention.append(is_correct)
        expected_cos_sim_per_mention.append(1 - scipy.spatial.distance.cosine(
            mention_encodings[i], entity_embeddings[mention_target_ids[i]]))
      else:
        expected_acc_per_mention.append(0)
        expected_cos_sim_per_mention.append(0)
    expected_acc_per_mention = np.array(expected_acc_per_mention)
    expected_cos_sim_per_mention = np.array(expected_cos_sim_per_mention)
    self.assertAlmostEqual(actual_loss, expected_loss, places=4)
    self.assertAlmostEqual(
        actual_metrics['denominator'], mention_target_weights.sum(), places=8)
    self.assertArrayAlmostEqual(
        actual_acc_per_mention, expected_acc_per_mention, places=8)
    self.assertAlmostEqual(
        actual_metrics['cos_sim'], expected_cos_sim_per_mention.sum(), places=2)

  @parameterized.parameters([
      {
          'batch_size': 1,
      },
      {
          'batch_size': 11,
      },
      {
          'batch_size': 11,
          'entity_vocab_size': 1000000,
      },
      {
          'batch_size': 2,
          'n_target_mentions': 19,
      },
      {
          'batch_size': 2,
          'n_target_mentions': 19,
          'entity_vocab_size': 2,
      },
      {
          'batch_size': 2,
          'k_top': 1,
      },
      {
          'batch_size': 2,
          'n_mentions_per_memory_passage': 21,
      },
      {
          'batch_size': 10,
          'n_mentions_per_memory_passage': 41,
      },
      {
          'batch_size': 10,
          'n_mentions': 100,
      },
      {
          'batch_size': 2,
          'n_mentions': 1,
      },
      {
          'batch_size': 11,
          'p_memory_mask': 0,
      },
      {
          'batch_size': 11,
          'p_memory_mask': 1,
      },
  ])
  def test_same_entity_set_retrieval_loss(self,
                                          batch_size,
                                          n_target_mentions=11,
                                          n_mentions=21,
                                          entity_vocab_size=10,
                                          k_top=10,
                                          n_mentions_per_memory_passage=4,
                                          p_memory_mask=0.5):
    np.random.seed(0)
    mention_target_batch_positions = np.random.randint(
        batch_size, size=(n_target_mentions))
    mention_target_ids = np.random.randint(
        entity_vocab_size, size=(n_target_mentions))
    mention_target_weights = np.random.randint(2, size=(n_target_mentions))
    mention_batch_positions = np.random.randint(batch_size, size=(n_mentions))
    mention_mask = np.random.randint(2, size=(n_mentions))
    memory_mask = np.random.random((n_mentions, k_top)) < p_memory_mask
    memory_mask = memory_mask.astype(np.int32)

    # `memory_text_entities` are assumed to contain unique IDs in the last dim.
    memory_text_entities = np.zeros(
        (n_mentions, k_top, n_mentions_per_memory_passage), np.int32)
    for m_index in range(n_mentions):
      for r_index in range(k_top):
        current_text_entities = np.random.choice(
            entity_vocab_size,
            size=(min(n_mentions_per_memory_passage, entity_vocab_size)),
            replace=False)
        memory_text_entities[
            m_index,
            r_index, :len(current_text_entities)] = current_text_entities
    memory_attention_weights = np.random.random((n_mentions, k_top))
    memory_attention_weights /= memory_attention_weights.sum(
        axis=-1, keepdims=True)

    actual_entity_overlap = mention_losses.get_batch_and_retrievals_entity_overlap(
        mention_target_batch_positions=mention_target_batch_positions,
        mention_target_ids=mention_target_ids,
        mention_target_weights=mention_target_weights,
        memory_text_entities=memory_text_entities.reshape(
            [n_mentions * k_top, -1]),
        batch_size=batch_size,
    )
    actual_entity_overlap = actual_entity_overlap.reshape(
        [batch_size, n_mentions, k_top])

    expected_entity_overlap = np.zeros((batch_size, n_mentions, k_top))
    for batch_index in range(batch_size):
      sample_ids = mention_target_ids[mention_target_batch_positions ==
                                      batch_index]
      sample_weights = mention_target_weights[mention_target_batch_positions ==
                                              batch_index]
      sample_ids = sample_ids[sample_weights > 0]
      sample_ids = set([x for x in sample_ids if x != 0])
      for m_index in range(n_mentions):
        for r_index in range(k_top):
          common_ids = set(
              memory_text_entities[m_index, r_index]).intersection(sample_ids)
          expected_entity_overlap[batch_index, m_index,
                                  r_index] = len(common_ids)

    self.assertArrayEqual(expected_entity_overlap, actual_entity_overlap)

    for same_entity_set_target_threshold in [1, 2, 3]:
      (actual_loss, actual_avg_probs,
       actual_denom) = mention_losses.same_entity_set_retrieval_loss(
           mention_target_batch_positions=mention_target_batch_positions,
           mention_target_ids=mention_target_ids,
           mention_target_weights=mention_target_weights,
           mention_batch_positions=mention_batch_positions,
           mention_mask=mention_mask,
           memory_text_entities=memory_text_entities,
           memory_attention_weights=memory_attention_weights,
           memory_mask=memory_mask,
           batch_size=batch_size,
           same_entity_set_target_threshold=same_entity_set_target_threshold,
       )

      expected_loss, expected_avg_probs, expected_denom = 0, 0, 0
      for batch_index in range(batch_size):
        for m_index in range(n_mentions):
          if mention_batch_positions[m_index] != batch_index:
            continue
          if mention_mask[m_index] == 0:
            continue

          correct_prob, n_positive, n_negative = 0, 0, 0
          for r_index in range(k_top):
            if memory_mask[m_index, r_index] == 0:
              continue
            if (expected_entity_overlap[batch_index, m_index, r_index] >=
                same_entity_set_target_threshold):
              correct_prob += memory_attention_weights[m_index, r_index]
              n_positive += 1
            else:
              n_negative += 1

          if n_positive > 0 and n_negative > 0:
            expected_loss -= np.log(correct_prob + 1e-5)
            expected_avg_probs += correct_prob
            expected_denom += 1

      self.assertEqual(actual_denom, expected_denom)
      self.assertAlmostEqual(actual_loss, expected_loss, places=4)
      self.assertAlmostEqual(actual_avg_probs, expected_avg_probs, places=4)


if __name__ == '__main__':
  absltest.main()

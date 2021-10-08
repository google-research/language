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
"""Tests for mention utils."""

import functools
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.utils import mention_utils
from language.mentionmemory.utils import test_utils
import numpy as np


class NumCommonUniqueItemsTest(test_utils.TestCase):

  n_devices = 3

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
    actual = mention_utils.mask_duplicate_ids(batch_positions, ids)
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
                                                     batch: List[List[int]],
                                                     expected: List[List[int]]):
    batch_positions, ids = [], []
    for batch_index, sample in enumerate(batch):
      for x in sample:
        batch_positions.append(batch_index)
        ids.append(x)
    self.assertEqual(len(ids), len(batch_positions))
    batch_positions = jnp.asarray(batch_positions)
    ids = jnp.asarray(ids)
    actual, _ = mention_utils.get_num_common_unique_items(
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
  def test_get_num_common_unique_items_multi_devices(self, seed: int,
                                                     batch_size: int,
                                                     n_mentions: int,
                                                     vocab_size: int):
    np.random.seed(seed)

    batch_positions_sharded = jax.device_put_sharded(
        list(np.random.randint(batch_size, size=(self.n_devices, n_mentions))),
        self.devices)
    ids_sharded = jax.device_put_sharded(
        list(np.random.randint(vocab_size, size=(self.n_devices, n_mentions))),
        self.devices)

    fn = functools.partial(
        mention_utils.get_num_common_unique_items, batch_size=batch_size)

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


class GlobalPositionsTest(test_utils.TestCase):
  """Test for computations regarding positions in global batch."""

  n_mentions = 37
  n_devices = 3
  batch_size = 5

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)
    self.devices = jax.local_devices()
    mention_batch_positions = [
        np.random.randint(self.batch_size, size=(self.n_mentions))
        for _ in range(self.n_devices)
    ]
    self.mention_batch_positions_sharded = jax.device_put_sharded(
        mention_batch_positions, self.devices)

  @parameterized.parameters((0, 5, 3), (1, 10, 1), (2, 10, 10))
  def test_get_globally_consistent_batch_positions(self, seed, batch_size,
                                                   n_mentions):
    np.random.seed(seed)

    mention_batch_positions_sharded = jax.device_put_sharded(
        list(np.random.randint(batch_size, size=(self.n_devices, n_mentions))),
        self.devices)

    fn = functools.partial(
        mention_utils.get_globally_consistent_batch_positions,
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


if __name__ == '__main__':
  absltest.main()

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
"""Test utils."""

import os


from absl.testing import parameterized
from jax.lib import xla_bridge
import numpy as np


class TestCase(parameterized.TestCase):
  """Custom test class containing additional useful utility methods."""

  def assertArrayEqual(self, actual, expected):
    actual = actual.ravel().tolist()
    expected = expected.ravel().tolist()
    self.assertSequenceEqual(actual, expected)

  def assertArrayAlmostEqual(self,
                             actual,
                             expected,
                             places = 7):
    actual = actual.ravel().tolist()
    expected = expected.ravel().tolist()
    self.assertSequenceAlmostEqual(actual, expected, places=places)


def force_multi_devices(num_cpu_devices):
  """Run with set number of CPU devices."""
  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str +
        ' --xla_force_host_platform_device_count={}'.format(num_cpu_devices))
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def tensor_to_numpy(tensor):
  """Convert numpy if not already numpy array."""
  if isinstance(tensor, np.ndarray):
    return tensor
  else:
    return tensor.numpy()


def gen_mention_pretraining_sample(
    text_length,
    n_mentions,
    n_linked_mentions,
    max_length = 100,
    vocab_size = 100,
    entity_vocab_size = 1000,
    mention_size = 2,
):
  """Generate test raw decoded input for mention pre-training pipeline."""

  text_pad_shape = (0, max_length - text_length)
  text_ids = np.random.randint(
      low=1, high=vocab_size, size=text_length, dtype=np.int64)
  text_ids = np.pad(text_ids, pad_width=text_pad_shape, mode='constant')
  text_mask = np.pad(
      np.ones(shape=text_length, dtype=np.int64),
      pad_width=text_pad_shape,
      mode='constant')

  mention_start_positions = np.random.choice(
      text_length // mention_size, size=n_mentions,
      replace=False) * mention_size
  mention_start_positions.sort()
  mention_end_positions = mention_start_positions + mention_size - 1

  dense_span_starts = np.zeros(shape=max_length, dtype=np.int64)
  dense_span_starts[mention_start_positions] = 1

  dense_span_ends = np.zeros(shape=max_length, dtype=np.int64)
  dense_span_ends[mention_end_positions] = 1

  linked_mention_indices = np.arange(n_linked_mentions)

  linked_mention_position_slices = [
      np.arange(mention_start_positions[idx], mention_end_positions[idx] + 1)
      for idx in linked_mention_indices
  ]

  if n_linked_mentions > 0:
    dense_linked_mention_positions = np.concatenate(
        linked_mention_position_slices)
  else:
    dense_linked_mention_positions = np.arange(0)

  linked_mention_ids = np.random.randint(
      low=1, high=entity_vocab_size, size=len(linked_mention_indices))

  dense_mention_mask = np.zeros(shape=max_length, dtype=np.int64)
  dense_mention_mask[dense_linked_mention_positions] = 1
  dense_mention_ids = np.zeros(shape=max_length, dtype=np.int64)
  for idx, position_slice in enumerate(linked_mention_position_slices):
    dense_mention_ids[position_slice] = linked_mention_ids[idx]

  dense_answer_mask = np.ones_like(dense_mention_mask)

  raw_example = {
      'text_ids': text_ids,
      'text_mask': text_mask,
      'dense_span_starts': dense_span_starts,
      'dense_span_ends': dense_span_ends,
      'dense_mention_mask': dense_mention_mask,
      'dense_mention_ids': dense_mention_ids,
      'dense_answer_mask': dense_answer_mask,
  }
  return raw_example

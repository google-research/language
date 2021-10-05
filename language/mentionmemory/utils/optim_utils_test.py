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
"""Tests for optimization utils."""

from absl.testing import absltest
from absl.testing import parameterized

from language.mentionmemory.utils import optim_utils


class LrSchedulerTest(parameterized.TestCase):
  """Test whether scheduler produces expected values."""

  @parameterized.parameters(
      # constant LR
      (0, 0.1, 0.1, False, 1000, False, 1e4, 0.0),
      (2000, 0.1, 0.1, False, 1000, False, 1e4, 0.0),
      # warmup steps
      (0, 0, 0.1, True, 1000, False, 1e4, 0.0),
      (500, 0.05, 0.1, True, 1000, False, 1e4, 0.0),
      (1000, 0.1, 0.1, True, 1000, False, 1e4, 0.0),
      (2000, 0.1, 0.1, True, 1000, False, 1e4, 0.0),
      # linear decay
      (500, 0.1, 0.1, False, 1000, True, 1e4, 0.0),
      (1e4, 0.0, 0.1, False, 1000, True, 1e4, 0.0),
      # both
      (500, 0.05, 0.1, True, 1000, True, 1e4, 0.0),
      (1e4, 0.0, 0.1, True, 1000, True, 1e4, 0.0),
  )
  def test_key_values_correct(
      self,
      step,
      value,
      learning_rate,
      warmup,
      warmup_steps,
      linear_decay,
      max_steps,
      decay_minimum_factor,
  ):
    """Match scheduler output with expected value."""
    lr_scheduler = optim_utils.create_learning_rate_scheduler(
        learning_rate=learning_rate,
        warmup=warmup,
        warmup_steps=warmup_steps,
        linear_decay=linear_decay,
        max_steps=max_steps,
        decay_minimum_factor=decay_minimum_factor,
    )
    self.assertEqual(lr_scheduler(step), value)


class DictMaskTest(parameterized.TestCase):
  """Test whether dict masking produces expected values."""

  @parameterized.parameters(
      (
          # If no mask keys, return same dict
          {
              'A': 1,
              'B': {
                  'C': 2
              },
          },
          [],
          {
              'A': True,
              'B': {
                  'C': True
              },
          },
      ),
      (
          # Filter value
          {
              'A': 1,
              'B': {
                  'C': 2,
                  'Delta': 3
              },
          },
          ['D'],
          {
              'A': True,
              'B': {
                  'C': True,
                  'Delta': False
              },
          },
      ),
      (
          # Filter subtree
          {
              'A': 1,
              'B': {
                  'C': 2,
                  'Delta': 3
              },
          },
          ['B'],
          {
              'A': True,
              'B': {
                  'C': False,
                  'Delta': False
              },
          },
      ),
  )
  def test_key_values_correct(
      self,
      test_dict,
      mask_keys,
      expected_dict,
  ):
    """Match masked dict with expected value."""
    masked_dict = optim_utils.create_dict_mask(test_dict, mask_keys)
    self.assertDictEqual(masked_dict, expected_dict)


if __name__ == '__main__':
  absltest.main()

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
"""Tests for sample_utils."""

from absl.testing import absltest
from language.casper.utils import sample_utils


class SampleUtilsTest(absltest.TestCase):

  def test_geometric_sample(self):
    pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sampled = sample_utils.geometric_sample(pool, 5, 0.5)
    self.assertLen(sampled, 5)
    sampled = sample_utils.geometric_sample(pool, 99, 0.5)
    self.assertLen(sampled, 10)
    # Test extreme values.
    sampled = sample_utils.geometric_sample(pool, 7, 1.0)
    self.assertEqual(sampled, [0, 1, 2, 3, 4, 5, 6])
    sampled = sample_utils.geometric_sample(pool, 7, 0.0)
    self.assertEqual(sampled, [9, 8, 7, 6, 5, 4, 3])


if __name__ == '__main__':
  absltest.main()

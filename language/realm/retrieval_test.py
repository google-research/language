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
"""Tests for retrieval.py."""
from language.realm import retrieval
import tensorflow as tf


class RetrievalTest(tf.test.TestCase):

  def test_yield_predictions_from_estimator_restarts(self):
    predictions = [(0, 'a'),
                   (1, 'b'),
                   (2, 'c'),
                   (3, 'd'),
                   (0, 'a'),  # restart.
                   (1, 'b'),
                   (2, 'c'),
                   (2, 'c'),  # rewind / didn't move.
                   (3, 'd'),
                   (4, 'e')]

    predictions = [{'result_idx': i, 'val': val} for i, val in predictions]

    results = list(
        retrieval.yield_predictions_from_estimator(predictions, total=5))

    self.assertEqual(results, [{'val': val} for val in 'abcde'])

  def test_yield_predictions_from_estimator_not_enough(self):
    predictions = [(0, 'a'),
                   (1, 'b'),
                   (2, 'c')]

    predictions = [{'result_idx': i, 'val': val} for i, val in predictions]

    # Set total=5 -- imagine that we are expecting 5 results.
    msg = 'Estimator.predict terminated before we got all results.'
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      list(retrieval.yield_predictions_from_estimator(predictions, total=5))

  def test_yield_predictions_from_estimator_skip(self):
    predictions = [(0, 'a'),
                   (1, 'b'),
                   (2, 'c'),  # We skipped 'd'.
                   (4, 'e')]

    predictions = [{'result_idx': i, 'val': val} for i, val in predictions]

    msg = 'Estimator.predict has somehow missed a result.'
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      list(retrieval.yield_predictions_from_estimator(predictions, total=5))


if __name__ == '__main__':
  tf.test.main()

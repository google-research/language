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
"""Test for span_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.question_answering.utils import span_utils
import numpy as np
import tensorflow as tf


class SpanUtilsTest(tf.test.TestCase):

  def test_max_scoring_span(self):
    batch_size = 7
    seq_len = 13
    start_scores = np.random.uniform(size=[batch_size, seq_len])
    end_scores = np.random.uniform(size=[batch_size, seq_len])
    with tf.Graph().as_default():
      best_start, best_end, best_score = span_utils.max_scoring_span(
          start_scores=tf.constant(start_scores, dtype=tf.float32),
          end_scores=tf.constant(end_scores, dtype=tf.float32))
      with tf.Session() as sess:
        actual_best_start, actual_best_end, actual_best_score = (
            sess.run([best_start, best_end, best_score]))

    for ss, es, bs, be, b_score in zip(start_scores, end_scores,
                                       actual_best_start, actual_best_end,
                                       actual_best_score):
      expected_best_start = -1
      expected_best_end = -1
      expected_best_score = float("-inf")
      for i, start_score in enumerate(ss):
        for j in range(i, len(es)):
          total_score = start_score + es[j]
          if total_score > expected_best_score:
            expected_best_score = total_score
            expected_best_start = i
            expected_best_end = j
      self.assertAllEqual(expected_best_start, bs)
      self.assertAllEqual(expected_best_end, be)
      self.assertAllClose(expected_best_score, b_score)


if __name__ == "__main__":
  tf.test.main()

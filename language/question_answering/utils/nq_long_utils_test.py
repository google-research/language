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
"""Tests for nq_long_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.question_answering.utils import nq_long_utils

import tensorflow as tf


class NqLongUtilsTest(tf.test.TestCase):

  def test_compute_thresholded_labels(self):
    with tf.Graph().as_default():
      labels = tf.to_int32([[0, 1, 5, 0, 0], [0, 2, 0, 0, 0], [0, 3, 0, 0, 0],
                            [0, 4, 8, 0, 0]])
      thresholded_labels = nq_long_utils.compute_thresholded_labels(labels)
      with tf.Session("") as sess:
        tf_thresholded_labels = sess.run(thresholded_labels)
        self.assertAllEqual(tf_thresholded_labels,
                            [[0, 1, 5, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                             [0, 4, 8, 0, 0]])

  def test_compute_compute_match_stats(self):
    with tf.Graph().as_default():
      predictions = tf.to_int32([1, 2, 3, 9])
      labels = tf.to_int32([[0, 1, 5, 0, 0], [0, 2, 0, 0, 0], [7, 3, 0, 0, 0],
                            [0, 4, 8, 0, 0]])
      non_null_match, non_null_gold, non_null_predictions = (
          nq_long_utils.compute_match_stats(
              predictions=predictions, labels=labels))
      with tf.Session("") as sess:
        tf_non_null_match, tf_non_null_gold, tf_non_null_predictions = (
            sess.run([non_null_match, non_null_gold, non_null_predictions]))
        self.assertAllClose(tf_non_null_match, [1.0, 0.0, 1.0, 0.0])
        self.assertAllClose(tf_non_null_gold, [1.0, 0.0, 1.0, 1.0])
        self.assertAllClose(tf_non_null_predictions, [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
  tf.test.main()

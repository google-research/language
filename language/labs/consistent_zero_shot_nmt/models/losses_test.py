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
"""Tests for losses.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.consistent_zero_shot_nmt.models import losses

import tensorflow as tf


class LossesTest(tf.test.TestCase):

  def test_l2_distance(self):
    """Tests l2 distance."""
    with tf.Graph().as_default():
      x = [1.0, 2.0]
      y = [3.0, 4.0]
      dist = losses.l2_distance(x=x, y=y)
      normalize_dist = losses.l2_distance(x=x, y=y, normalize=True)
      with tf.Session("") as sess:
        tf_dist, tf_normalize_dist = sess.run([dist, normalize_dist])
        self.assertAllClose([tf_dist, tf_normalize_dist], [8.0, 0.0322602])


if __name__ == "__main__":
  tf.test.main()

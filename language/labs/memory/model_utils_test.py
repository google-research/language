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
"""Tests for model_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.memory.model_utils import hamming_loss

import tensorflow as tf


class ModelUtilsTest(tf.test.TestCase):

  def test_hamming_loss(self):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        preds = tf.constant([1, 2, 3, 4])
        labels = tf.constant([3, 2, 1, 0])

        val, update_op = hamming_loss(preds, labels)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(update_op)

        self.assertEqual(sess.run(val), .75)

  def test_hamming_loss_sign(self):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        preds = tf.constant([.25, .25, .25, .25])
        labels = tf.constant([-1, -1, 1, 1])

        val, update_op = hamming_loss(preds, labels, sign=True)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(update_op)

        self.assertEqual(sess.run(val), .5)


if __name__ == "__main__":
  tf.test.main()

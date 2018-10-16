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
# coding=utf-8
"""Tests for common_layers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.inputs import char_utils
from language.common.layers import common_layers
import tensorflow as tf


class CommonLayersTest(tf.test.TestCase):

  def test_ffnn(self):
    with tf.Graph().as_default():
      input_emb = tf.random_uniform([3, 5, 8])
      output_emb = common_layers.ffnn(
          input_emb=input_emb,
          hidden_sizes=[7, 9],
          dropout_ratio=0.2,
          mode=tf.estimator.ModeKeys.TRAIN)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_output_emb = sess.run(output_emb)
      self.assertAllEqual(actual_output_emb.shape, [3, 5, 9])

  def test_stacked_highway(self):
    with tf.Graph().as_default():
      input_emb = tf.random_uniform([3, 5, 8])
      output_emb = common_layers.stacked_highway(
          input_emb=input_emb,
          hidden_sizes=[7, 9],
          dropout_ratio=0.2,
          mode=tf.estimator.ModeKeys.TRAIN)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_output_emb = sess.run(output_emb)
      self.assertAllEqual(actual_output_emb.shape, [3, 5, 9])

  def test_character_cnn(self):
    with tf.Graph().as_default():
      input_words = [["google", "lumiere"],
                     [u"¯\\_(ツ)_/¯", u"(ᵔᴥᵔ)"],
                     [u"谷", u"歌"]]
      char_ids = char_utils.batch_word_to_char_ids(tf.constant(input_words), 10)
      output_emb = common_layers.character_cnn(char_ids, num_filters=5)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_output_emb = sess.run(output_emb)
      self.assertAllEqual(actual_output_emb.shape, [3, 2, 5])


if __name__ == "__main__":
  tf.test.main()

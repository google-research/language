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
"""Test for decomposable_attention.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.question_answering.layers import decomposable_attention as decatt
import tensorflow as tf


class DecomposableAttentionTest(tf.test.TestCase):

  def test_decomposable_attention(self):
    with tf.Graph().as_default():
      input1_emb = tf.random_uniform([3, 5, 7])
      input1_len = tf.constant([5, 2, 0])
      input2_emb = tf.random_uniform([3, 8, 7])
      input2_len = tf.constant([8, 6, 1])
      output_emb = decatt.decomposable_attention(
          emb1=input1_emb,
          len1=input1_len,
          emb2=input2_emb,
          len2=input2_len,
          hidden_size=5,
          hidden_layers=2,
          dropout_ratio=0.1,
          mode=tf.estimator.ModeKeys.TRAIN)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_output_emb = sess.run(output_emb)
      self.assertAllEqual(actual_output_emb.shape, [3, 5])

if __name__ == "__main__":
  tf.test.main()

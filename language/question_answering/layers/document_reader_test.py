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
"""Test for document_reader.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.question_answering.layers import document_reader
import tensorflow as tf


class DocumentReaderTest(tf.test.TestCase):

  def test_document_reader(self):
    with tf.Graph().as_default():
      start_score, end_score = document_reader.score_endpoints(
          question_emb=tf.random_uniform([3, 5, 7]),
          question_len=tf.constant([5, 2, 0]),
          context_emb=tf.random_uniform([3, 8, 7]),
          context_len=tf.constant([8, 6, 1]),
          hidden_size=9,
          num_layers=2,
          dropout_ratio=0.1,
          mode=tf.estimator.ModeKeys.TRAIN)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_start_score, actual_end_score = sess.run(
            [start_score, end_score])
      self.assertAllEqual(actual_start_score.shape, [3, 8])
      self.assertAllEqual(actual_end_score.shape, [3, 8])


if __name__ == "__main__":
  tf.test.main()

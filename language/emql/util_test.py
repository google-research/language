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
# Lint as: python3
"""Tests for emql.util."""

from language.emql import util
import numpy as np
import tensorflow.compat.v1 as tf


class UtilTest(tf.test.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()
    self.sess = tf.Session()
    self.logits = tf.constant([[2, 3, 1, -1], [4, 1, 9, 3]], dtype=tf.float32)
    self.labels = tf.constant([[0, 1, 0, 1], [1, 0, 0, 0]], dtype=tf.float32)

  def test_hits_at_k(self):
    hits_at_one = util.compute_hits_at_k(self.logits, self.labels, k=1)
    hits_at_two = util.compute_hits_at_k(self.logits, self.labels, k=2)

    self.assertAllEqual(
        hits_at_one.eval(session=self.sess), np.array([1, 0]))
    self.assertAllEqual(
        hits_at_two.eval(session=self.sess), np.array([1, 1]))

  def test_recall_at_k(self):
    recall_at_one = util.compute_recall_at_k(self.logits, self.labels, k=1)
    recall_at_two = util.compute_recall_at_k(self.logits, self.labels, k=2)

    self.assertAllEqual(
        recall_at_one.eval(session=self.sess), np.array([0.5, 0]))
    self.assertAllEqual(
        recall_at_two.eval(session=self.sess), np.array([0.5, 1]))

  def test_map_at_k(self):
    map_at_one = util.compute_average_precision_at_k(
        self.logits, self.labels, k=1)
    map_at_two = util.compute_average_precision_at_k(
        self.logits, self.labels, k=2)

    self.assertAllEqual(
        map_at_one.eval(session=self.sess), np.array([1.0, 0.0]))
    self.assertAllEqual(
        map_at_two.eval(session=self.sess), np.array([1.0, 0.5]))

  def test_get_nonzero_ids(self):
    nonzero_at_one = util.get_nonzero_ids(self.labels, k=1)
    nonzero_at_two = util.get_nonzero_ids(self.labels, k=2)

    self.assertAllEqual(
        nonzero_at_one.eval(session=self.sess), np.array([[1], [0]]))
    self.assertAllEqual(
        nonzero_at_two.eval(session=self.sess), np.array([[1, 3], [0, -1]]))

  def test_embedding_lookup_with_padding(self):
    tokens = tf.constant([0, -1], dtype=tf.int32)
    embeddings_mat = tf.constant([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    embs = util.embedding_lookup_with_padding(
        embeddings_mat, tokens, padding=-1)
    embs_np = embs.eval(session=self.sess)  # 2, 3
    self.assertAllClose(embs_np[0, :], [0.1, 0.2, 0.3])
    self.assertAllClose(embs_np[1, :], [0, 0, 0])

  def test_x_in_set(self):
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    s = tf.constant([[1, 2, 5], [4, 7, 8]], dtype=tf.int32)
    _, x_in_s = util.compute_x_in_set(x, s)
    self.assertAllEqual(
        x_in_s.eval(session=self.sess), np.array([[1, 1], [0, 1]]))

  def test_bert_tokenizer(self):
    text = 'hello world'
    bert_tokenizer = util.BertTokenizer()
    _, (token_ids, _, input_mask) = bert_tokenizer.tokenize(text)
    self.assertGreater(np.sum(input_mask), 2)
    self.assertAllEqual(token_ids != 0, input_mask)
    self.assertEqual(len(token_ids), bert_tokenizer.max_seq_length)


if __name__ == '__main__':
  tf.test.main()

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
"""Tests for embedding_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from language.common.inputs import embedding_utils

import numpy as np
import tensorflow as tf


class EmbeddingUtilsTest(tf.test.TestCase):

  def test_l2_normalize_random(self):
    v = np.random.uniform(size=[3])
    norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    expected_normalized = [v[0] / norm, v[1] / norm, v[2] / norm]
    self.assertAllClose(embedding_utils.l2_normalize(v), expected_normalized)

  def test_l2_normalize_zero(self):
    v = np.zeros([3])
    self.assertAllClose(embedding_utils.l2_normalize(v), np.zeros([3]))

  def test_l2_normalize_higherdim(self):
    v = np.array([[0, 0, 2]], dtype=np.float32)
    expected_normalized = np.array([[0, 0, 1]], dtype=np.float32)
    self.assertAllClose(embedding_utils.l2_normalize(v, axis=1),
                        expected_normalized)

  def _test_pretrained_word_embeddings(self, trainable, num_oov_buckets):
    temp_dir = tempfile.mkdtemp(prefix="embedding_utils_test")
    temp_path = os.path.join(temp_dir, "emb.txt")
    with tf.gfile.Open(temp_path, "w") as temp_file:
      temp_file.write("a 0.5 0.8 -0.1 0.0 1.0\n")
      temp_file.write("b -0.5 -0.8 0.1 -0.0 -1.0\n")
      temp_file.write("c -2.5 10 0.1 0 -0.005\n")
      temp_file.write("d -3.5 -3 1.1 0.5 1.5\n")
    max_vocab_size = 2
    embeddings = embedding_utils.PretrainedWordEmbeddings(
        temp_path,
        max_vocab_size=max_vocab_size,
        num_oov_buckets=num_oov_buckets)
    with tf.Graph().as_default():
      embedding_weights, embedding_scaffold = embeddings.get_params(
          trainable=trainable)
      dataset = tf.data.Dataset.from_tensor_slices({"s": ["a", "b", "<UNK>"]})
      dataset = dataset.map(embeddings.token_to_word_id_mapper(["s"]))
      self.assertDictEqual(dataset.output_types,
                           {"s": tf.string,
                            "s_wid": tf.int32})
      self.assertDictEqual(dataset.output_shapes,
                           {"s": [], "s_wid": []})

      dataset = dataset.batch(3)
      iterator = dataset.make_initializable_iterator()
      features = iterator.get_next()

      emb = tf.nn.embedding_lookup(embedding_weights, features["s_wid"])

      _, oov_metric = embeddings.oov_metric(features["s_wid"])

      with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),
                  tf.tables_initializer(),
                  iterator.initializer])
        embedding_scaffold.init_fn(sess)
        sess.run([tf.local_variables_initializer()])

        tf_embedding_weights = sess.run(embedding_weights)
        self.assertAllEqual(
            tf_embedding_weights.shape, [max_vocab_size + num_oov_buckets, 5])

        tf_s, tf_s_wid, tf_emb, tf_oov_metric = (
            sess.run([
                features["s"],
                features["s_wid"],
                emb,
                oov_metric]))

      self.assertAllEqual(tf_s, ["a", "b", "<UNK>"])
      self.assertAllEqual(tf_s_wid[:2], [0, 1])
      self.assertGreater(tf_s_wid[2], 1)

      expected_a_emb = embedding_utils.l2_normalize(
          [0.5, 0.8, -0.1, 0.0, 1.0])
      expected_b_emb = embedding_utils.l2_normalize(
          [-0.5, -0.8, 0.1, 0.0, -1.0])
      expected_emb = np.stack([expected_a_emb, expected_b_emb])
      self.assertAllClose(tf_emb[:2, :], expected_emb)

      if num_oov_buckets == 1:
        expected_unk_emb = np.zeros([5])
        self.assertAllClose(tf_emb[2, :], expected_unk_emb)

      self.assertListEqual(["a", "b"], embeddings.get_vocab())
      self.assertEqual(5, embeddings.get_dims())
      self.assertEqual(
          max_vocab_size + num_oov_buckets,
          embeddings.get_vocab_size_with_oov())

      self.assertAllClose(tf_oov_metric, 1. / 3)

  def test_pretrained_word_embeddings_single_bucket_trainable(self):
    self._test_pretrained_word_embeddings(trainable=True, num_oov_buckets=1)

  def test_pretrained_word_embeddings_single_bucket_fixed(self):
    self._test_pretrained_word_embeddings(trainable=False, num_oov_buckets=1)

  def test_pretrained_word_embeddings_multiple_buckets_trainable(self):
    self._test_pretrained_word_embeddings(trainable=True, num_oov_buckets=5)

  def test_pretrained_word_embeddings_multiple_buckets_fixed(self):
    self._test_pretrained_word_embeddings(trainable=False, num_oov_buckets=5)

if __name__ == "__main__":
  tf.test.main()

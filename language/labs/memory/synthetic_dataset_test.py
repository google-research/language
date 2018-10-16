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
"""Tests for synthetic_dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from language.labs.memory import synthetic_dataset

import tensorflow as tf


class PatternDataTest(tf.test.TestCase):

  def test_generate_pattern(self):
    random.seed(10)
    with tf.Graph().as_default():
      dataset = synthetic_dataset.get_pattern_dataset(
          n=2, num_patterns=3, pattern_size=5)
      features = dataset.make_one_shot_iterator().get_next()

      with tf.Session() as sess:
        seqs, targets = sess.run([features["seqs"], features["targets"]])

        # Test that we generate sequence of 3 random binary patterns and 1
        # degraded query pattern each with dimensionality 5.
        self.assertAllEqual(seqs, [[-1, 1, 1, -1, -1], [-1, -1, -1, 1, -1],
                                   [1, 1, 1, 1, -1], [-1, 0, -1, 1, 0]])

        # Test that the target pattern to retrieve given the degraded query is
        # correct.
        self.assertAllEqual(targets, [-1, -1, -1, 1, -1])

        # Test that we can get the second element of the batch.
        seqs2, targets2 = sess.run([features["seqs"], features["targets"]])
        self.assertAllEqual(seqs2, [[1, 1, 1, 1, -1], [-1, 1, 1, -1, -1],
                                    [1, 1, 1, 1, 1], [0, 1, 1, -1, 0]])
        self.assertAllEqual(targets2, [-1, 1, 1, -1, -1])

  def test_generate_pattern_selective(self):
    random.seed(10)
    with tf.Graph().as_default():
      dataset = synthetic_dataset.get_pattern_dataset(
          n=1,
          num_patterns=3,
          pattern_size=5,
          selective=True,
          num_patterns_store=2)
      features = dataset.make_one_shot_iterator().get_next()

      with tf.Session() as sess:
        seqs, targets = sess.run([features["seqs"], features["targets"]])

        # Test that we generate sequence of 3 random binary patterns.
        # remember_idxs should be [0, 1] - the last element of these patterns
        # should be 1 and 0 for the others.
        self.assertAllEqual(seqs,
                            [[-1, 1, 1, -1, -1, 1], [-1, -1, -1, 1, -1, 1],
                             [1, 1, 1, 1, -1, 0], [0, 0, -1, 1, -1, 1]])

        # Test that the target pattern to retrieve given the degraded query is
        # correct.
        self.assertAllEqual(targets, [-1, -1, -1, 1, -1, 1])

  def test_generate_symbolic_data(self):
    random.seed(10)

    vocab = synthetic_dataset.get_symbolic_vocab()

    with tf.Graph().as_default():
      dataset = synthetic_dataset.get_symbolic_dataset(True, n=1, num_pairs=3)
      features = dataset.make_one_shot_iterator().get_next()

      with tf.Session() as sess:
        seqs, targets = sess.run([features["seqs"], features["targets"]])

        # Test the sequence is the encoded string of k,v pairs.
        self.assertAllEqual(seqs, [6, 31, 0, 33, 21, 35, 36, 0])
        self.assertEqual("".join([vocab[i] for i in seqs]), "g5a7v9?a")

        # Test that the target is correct.
        self.assertEqual(targets, 33)
        self.assertEqual(vocab[targets], "7")


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for char_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.inputs import char_utils
import tensorflow as tf


class CharUtilsTest(tf.test.TestCase):

  def test_empty_word(self):
    words_to_test = [""]
    expected_char_ids = [[256, 257, 258, 258, 258]]
    self._test_words(words_to_test, expected_char_ids, 5)

  def test_ascii_word(self):
    words_to_test = []
    expected_char_ids = []

    words_to_test.append("pad")
    expected_char_ids.append([256, 112, 97, 100, 257, 258, 258])

    words_to_test.append("exact")
    expected_char_ids.append([256, 101, 120, 97, 99, 116, 257])

    words_to_test.append("truncated")
    expected_char_ids.append([256, 116, 114, 117, 110, 99, 257])

    self._test_words(words_to_test, expected_char_ids, 7)

  def test_unicode_word(self):
    words_to_test = []
    expected_char_ids = []

    words_to_test.append(u"谷")
    expected_char_ids.append(
        [256, 232, 176, 183, 257, 258, 258, 258, 258, 258, 258, 258])

    words_to_test.append(u"(ᵔᴥᵔ)")
    expected_char_ids.append(
        [256, 40, 225, 181, 148, 225, 180, 165, 225, 181, 148, 257])

    words_to_test.append(u"¯\\_(ツ)_/¯")
    expected_char_ids.append(
        [256, 194, 175, 92, 95, 40, 227, 131, 132, 41, 95, 257])

    self._test_words(words_to_test, expected_char_ids, 12)

  def _test_words(self, words, expected_char_ids, word_length):
    with tf.Graph().as_default():
      char_ids = char_utils.batch_word_to_char_ids(
          tf.constant(words), word_length)
      with tf.Session() as sess:
        actual_char_ids = sess.run(char_ids)

    for a_cid, e_cid in zip(actual_char_ids, expected_char_ids):
      self.assertAllEqual(a_cid, e_cid)

  def test_token_to_char_ids_mapper(self):
    with tf.Graph().as_default():
      dataset = tf.data.Dataset.from_tensor_slices({"s": ["a", "b"]})
      dataset = dataset.map(char_utils.token_to_char_ids_mapper(["s"], 4))
      self.assertDictEqual(dataset.output_types,
                           {"s": tf.string, "s_cid": tf.int32})
      self.assertDictEqual(dataset.output_shapes,
                           {"s": [], "s_cid": [4]})

      dataset = dataset.batch(2)
      features = dataset.make_one_shot_iterator().get_next()

      with tf.Session() as sess:
        tf_s, tf_s_cid = sess.run([features["s"], features["s_cid"]])

      self.assertAllEqual(tf_s, ["a", "b"])

      expected_a_emb = [char_utils.BOW_CHAR, 97, char_utils.EOW_CHAR,
                        char_utils.PAD_CHAR]
      expected_b_emb = [char_utils.BOW_CHAR, 98, char_utils.EOW_CHAR,
                        char_utils.PAD_CHAR]
      self.assertAllEqual(tf_s_cid, [expected_a_emb, expected_b_emb])

if __name__ == "__main__":
  tf.test.main()

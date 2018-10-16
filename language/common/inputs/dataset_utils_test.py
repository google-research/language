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
"""Tests for dataset_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.inputs import dataset_utils

import tensorflow as tf


class DatasetUtilsTest(tf.test.TestCase):

  def test_length_mapper(self):
    with tf.Graph().as_default():
      short_dataset = tf.data.Dataset.from_tensors({
          "s_tok": ["short", "phrase"]
      })
      long_dataset = tf.data.Dataset.from_tensors({
          "s_tok": ["a", "phrase", "that", "is", "longer"]
      })
      dataset = short_dataset.concatenate(long_dataset)
      dataset = dataset.map(dataset_utils.length_mapper(["s_tok"]))
      dataset = dataset.padded_batch(2, dataset.output_shapes)

      self.assertDictEqual(dataset.output_types, {"s_tok": tf.string,
                                                  "s_tok_len": tf.int32})
      features = dataset.make_one_shot_iterator().get_next()

      with tf.Session() as sess:
        tf_s_tok, tf_s_tok_len = sess.run(
            [features["s_tok"], features["s_tok_len"]])

      self.assertAllEqual(tf_s_tok,
                          [["short", "phrase", "", "", ""],
                           ["a", "phrase", "that", "is", "longer"]])
      self.assertAllEqual(tf_s_tok_len, [2, 5])

  def test_string_to_int_mapper(self):
    with tf.Graph().as_default():
      dataset = tf.data.Dataset.from_tensor_slices({
          "s": [["a", "b"], ["c", "d"]]
      })
      dataset = dataset.map(dataset_utils.string_to_int_mapper(
          ["s"], ["a", "c"]))
      dataset = dataset.batch(2)

      self.assertDictEqual(dataset.output_types, {"s": tf.string,
                                                  "s_id": tf.int32})
      iterator = dataset.make_initializable_iterator()
      features = iterator.get_next()

      with tf.Session() as sess:
        sess.run([tf.tables_initializer(), iterator.initializer])
        tf_s, tf_s_id = sess.run([features["s"], features["s_id"]])

      self.assertAllEqual(tf_s, [["a", "b"], ["c", "d"]])
      self.assertAllEqual(tf_s_id, [[0, 2], [1, 2]])


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for tensor_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.utils import tensor_utils
import numpy as np
import tensorflow as tf


class TensorUtilsTest(tf.test.TestCase):

  def test_shape_dynamic(self):
    with tf.Graph().as_default():
      tensor = tf.placeholder(tf.int64, [None, None])
      d0_single = tensor_utils.shape(tensor, 0)
      d1_single = tensor_utils.shape(tensor, 1)
      d0_full, d1_full = tensor_utils.shape(tensor)

      self.assertIsInstance(d0_single, tf.Tensor)
      self.assertIsInstance(d1_single, tf.Tensor)
      self.assertIsInstance(d0_full, tf.Tensor)
      self.assertIsInstance(d1_full, tf.Tensor)

      with tf.Session() as sess:
        feed_dict = {tensor: np.zeros((4, 7))}

        tf_d0_single, tf_d1_single = sess.run([d0_single, d1_single],
                                              feed_dict=feed_dict)
        self.assertEqual(tf_d0_single, 4)
        self.assertEqual(tf_d1_single, 7)

        tf_d0_full, tf_d1_full = sess.run([d0_full, d1_full],
                                          feed_dict=feed_dict)
        self.assertEqual(tf_d0_full, 4)
        self.assertEqual(tf_d1_full, 7)

  def test_shape_static(self):
    with tf.Graph().as_default():
      tensor = tf.placeholder(tf.int64, [4, 7])
      d0_single = tensor_utils.shape(tensor, 0)
      d1_single = tensor_utils.shape(tensor, 1)
      d0_full, d1_full = tensor_utils.shape(tensor)

      self.assertIsInstance(d0_single, int)
      self.assertIsInstance(d1_single, int)
      self.assertIsInstance(d0_full, int)
      self.assertIsInstance(d1_full, int)

      self.assertEqual(d0_single, 4)
      self.assertEqual(d1_single, 7)
      self.assertEqual(d0_full, 4)
      self.assertEqual(d1_full, 7)

  def test_shape_mixed(self):
    """Test for shape() with a mixture of static and dynamic sizes."""
    with tf.Graph().as_default():
      tensor = tf.placeholder(tf.int64, [4, None])
      d0_single = tensor_utils.shape(tensor, 0)
      d1_single = tensor_utils.shape(tensor, 1)
      d0_full, d1_full = tensor_utils.shape(tensor)

      self.assertIsInstance(d0_single, int)
      self.assertIsInstance(d1_single, tf.Tensor)
      self.assertIsInstance(d0_full, int)
      self.assertIsInstance(d1_full, tf.Tensor)

      self.assertEqual(d0_single, 4)
      self.assertEqual(d0_full, 4)

      with tf.Session() as sess:
        feed_dict = {tensor: np.zeros((4, 7))}

        tf_d1_single = sess.run(d1_single, feed_dict=feed_dict)
        self.assertEqual(tf_d1_single, 7)

        tf_d1_full = sess.run(d1_full, feed_dict=feed_dict)
        self.assertEqual(tf_d1_full, 7)

  def test_where_select_elements(self):
    """Test for where() when selecting individual elements."""
    with self.test_session(graph=tf.Graph()):
      condition = tf.constant([[True, False],
                               [False, True]])  # pyformat: disable
      if_true = tf.constant([[1, 2],
                             [3, 4]])  # pyformat: disable
      if_false = tf.constant([[5, 6],
                              [7, 8]])  # pyformat: disable

      result = tensor_utils.where(condition, if_true, if_false)

      self.assertAllEqual(result.eval(),
                          [[1, 6],
                           [7, 4]])  # pyformat: disable

  def test_where_select_axis_0(self):
    """Test for where() when selecting on axis 0."""
    with self.test_session(graph=tf.Graph()):
      condition = tf.constant([True, False])
      if_true = tf.constant([[1, 2],
                             [3, 4]])  # pyformat: disable
      if_false = tf.constant([[5, 6],
                              [7, 8]])  # pyformat: disable

      result = tensor_utils.where(condition, if_true, if_false)

      self.assertAllEqual(result.eval(),
                          [[1, 2],
                           [7, 8]])  # pyformat: disable

  def test_where_select_nontrivial(self):
    """Test for where() when selecting on an intermediate axis."""
    with self.test_session(graph=tf.Graph()):
      condition = tf.constant([[True, False],
                               [False, True]])  # pyformat: disable
      if_true = tf.constant([[[1, 1], [2, 2]],
                             [[3, 3], [4, 4]]])  # pyformat: disable
      if_false = tf.constant([[[5, 5], [6, 6]],
                              [[7, 7], [8, 8]]])  # pyformat: disable

      result = tensor_utils.where(condition, if_true, if_false)

      self.assertAllEqual(result.eval(),
                          [[[1, 1], [6, 6]],
                           [[7, 7], [4, 4]]])  # pyformat: disable

  def test_shaped_py_func(self):
    def _fn(x, y):
      return np.array([x + y, x * y])

    with tf.Graph().as_default():
      z, = tensor_utils.shaped_py_func(
          func=_fn,
          inputs=[tf.constant(4), tf.constant(7)],
          types=[tf.int32],
          shapes=[2])
      self.assertAllEqual(z.get_shape(), [2])

      with tf.Session() as sess:
        tf_z = sess.run(z)
      self.assertAllEqual(tf_z, [11, 28])

  def test_unflatten(self):
    with tf.Graph().as_default():
      tensor = tf.placeholder(tf.float32, [4, 7, 6, 3])
      w = tf.placeholder(tf.float32, [3, 9])

      flat_tensor, unflatten = tensor_utils.flatten(tensor)
      self.assertAllEqual(tensor_utils.shape(flat_tensor), [4 * 7 * 6, 3])

      flat_projected_tensor = tf.matmul(flat_tensor, w)
      projected_tensor = unflatten(flat_projected_tensor)
      self.assertAllEqual(tensor_utils.shape(projected_tensor), [4, 7, 6, 9])

  def test_transpose_batch_time_rank_1(self):
    with self.test_session(graph=tf.Graph()):
      tensor = tf.constant([1, 2, 3])

      transposed = tensor_utils.transpose_batch_time(tensor)

      self.assertAllEqual(transposed.eval(), [1, 2, 3])

  def test_transpose_batch_time_rank_2(self):
    with self.test_session(graph=tf.Graph()):
      tensor = tf.constant([[1, 2],
                            [3, 4]])  # pyformat: disable

      transposed = tensor_utils.transpose_batch_time(tensor)

      self.assertAllEqual(transposed.eval(),
                          [[1, 3],
                           [2, 4]])  # pyformat: disable

  def test_transpose_batch_time_rank_3(self):
    with self.test_session(graph=tf.Graph()):
      tensor = tf.constant([[[1, 2], [3, 4]],
                            [[5, 6], [7, 8]]])  # pyformat: disable

      transposed = tensor_utils.transpose_batch_time(tensor)

      self.assertAllEqual(transposed.eval(),
                          [[[1, 2], [5, 6]],
                           [[3, 4], [7, 8]]])  # pyformat: disable

  def test_transpose_batch_time_rank_unknown(self):
    with self.test_session(graph=tf.Graph()):
      tensor = tf.placeholder(tf.float32, shape=None)

      with self.assertRaisesRegexp(ValueError, "Tensor with unknown rank"):
        tensor_utils.transpose_batch_time(tensor)

  def test_sequence_mask(self):
    with self.test_session(graph=tf.Graph()):
      lengths = tf.constant([1, 2, 3])

      mask = tensor_utils.sequence_mask(lengths)
      transposed_mask = tensor_utils.sequence_mask(lengths, transpose=True)

      self.assertAllEqual(mask.eval(),
                          [[True, False, False],
                           [True, True, False],
                           [True, True, True]])  # pyformat: disable

      self.assertAllEqual(transposed_mask.eval(),
                          [[True, True, True],
                           [False, True, True],
                           [False, False, True]])  # pyformat: disable

  def test_sequence_mask_with_maxlen(self):
    with self.test_session(graph=tf.Graph()):
      lengths = tf.constant([1, 2])

      mask = tensor_utils.sequence_mask(lengths, maxlen=3)
      transposed_mask = tensor_utils.sequence_mask(
          lengths, maxlen=3, transpose=True)

      self.assertAllEqual(mask.eval(),
                          [[True, False, False],
                           [True, True, False]])  # pyformat: disable

      self.assertAllEqual(transposed_mask.eval(),
                          [[True, True],
                           [False, True],
                           [False, False]])  # pyformat: disable

  def test_sequence_mask_with_dtype(self):
    with self.test_session(graph=tf.Graph()):
      lengths = tf.constant([1, 2, 3])

      mask = tensor_utils.sequence_mask(lengths, dtype=tf.int32)
      transposed_mask = tensor_utils.sequence_mask(
          lengths, dtype=tf.int32, transpose=True)

      self.assertAllEqual(mask.eval(),
                          [[1, 0, 0],
                           [1, 1, 0],
                           [1, 1, 1]])  # pyformat: disable

      self.assertAllEqual(transposed_mask.eval(),
                          [[1, 1, 1],
                           [0, 1, 1],
                           [0, 0, 1]])  # pyformat: disable


if __name__ == "__main__":
  tf.test.main()

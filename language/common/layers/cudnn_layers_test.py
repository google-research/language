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
"""Tests for cudnn_layers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest

from language.common.layers import cudnn_layers
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test


class CudnnLayersTest(tf.test.TestCase):

  def test_stacked_bilstm(self):
    with tf.Graph().as_default():
      input_emb = tf.random_uniform([3, 5, 8])
      input_len = tf.constant([4, 5, 2])
      output_emb = cudnn_layers.stacked_bilstm(
          input_emb=input_emb,
          input_len=input_len,
          hidden_size=10,
          num_layers=3,
          dropout_ratio=0.2,
          mode=tf.estimator.ModeKeys.TRAIN)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_output_emb = sess.run(output_emb)
      self.assertAllEqual(actual_output_emb.shape, [3, 5, 10 * 2])

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_stacked_bilstm_compatibility(self):
    checkpoint_dir = tempfile.mkdtemp(prefix="checkpoint_dir")
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    hidden_size = 10
    num_layers = 3
    dropout_ratio = 0.0
    input_emb = np.random.uniform(size=[3, 5, 9]).astype(np.float32)
    input_len = [4, 5, 2]

    # Make sure we fail explicitly if the specified devices can't be used.
    config = tf.ConfigProto(
        allow_soft_placement=False, log_device_placement=True)

    with tf.Graph().as_default():
      with tf.device("/gpu:0"):
        output_emb = cudnn_layers.stacked_bilstm(
            input_emb=input_emb,
            input_len=input_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_ratio=dropout_ratio,
            mode=tf.estimator.ModeKeys.TRAIN,
            use_cudnn=True)
      saver = tf.train.Saver()
      with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        gpu_output_emb = sess.run(output_emb)
        saver.save(sess, checkpoint_path)

    with tf.Graph().as_default():
      with tf.device("/cpu:0"):
        output_emb = cudnn_layers.stacked_bilstm(
            input_emb=input_emb,
            input_len=input_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_ratio=dropout_ratio,
            mode=tf.estimator.ModeKeys.TRAIN,
            use_cudnn=False)
      saver = tf.train.Saver()
      with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint_path)
        cpu_output_emb = sess.run(output_emb)

    for c, g, l in zip(cpu_output_emb, gpu_output_emb, input_len):
      self.assertAllClose(c[:l], g[:l])


if __name__ == "__main__":
  tf.test.main()

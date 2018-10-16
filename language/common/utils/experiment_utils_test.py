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
"""Tests for experiment_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.utils import experiment_utils
import tensorflow as tf


class ExperimentUtilsTest(tf.test.TestCase):

  def _simple_model_fn(self, features, labels, mode):
    logits = tf.squeeze(tf.layers.dense(features, 1))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(labels), logits=logits))
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

  def _simple_input_function(self):
    features = [[1.0, 0.0, -1.0, 2.5],
                [0.5, 1.1, -0.8, 1.5]]
    labels = [0, 1]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat(2)
    dataset = dataset.batch(2)
    return dataset

  def test_run_experiment(self):
    experiment_utils.run_experiment(
        model_fn=self._simple_model_fn,
        train_input_fn=self._simple_input_function,
        eval_input_fn=self._simple_input_function)


if __name__ == "__main__":
  tf.test.main()

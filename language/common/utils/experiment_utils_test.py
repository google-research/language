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

from absl import flags
from language.common.utils import experiment_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

FLAGS = flags.FLAGS

FLAGS.set_default("num_train_steps", 2)
FLAGS.set_default("num_eval_steps", 2)


class ExperimentUtilsTest(tf.test.TestCase):

  def _simple_model_fn(self, features, labels, mode, params):
    logits = tf.squeeze(tf.layers.dense(features, 1))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(labels), logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    if params["use_tpu"]:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(
        loss, global_step=tf.train.get_or_create_global_step())
    if params["use_tpu"]:
      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    else:
      return tf_estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  def _simple_input_function(self, params):
    features = [[1.0, 0.0, -1.0, 2.5],
                [0.5, 1.1, -0.8, 1.5]]
    labels = [0, 1]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(params["batch_size"], drop_remainder=True)
    return dataset

  def test_run_experiment(self):
    experiment_utils.run_experiment(
        model_fn=self._simple_model_fn,
        train_input_fn=self._simple_input_function,
        eval_input_fn=self._simple_input_function)

  def test_run_experiment_tpu(self):
    params = dict(use_tpu=True)
    experiment_utils.run_experiment(
        model_fn=self._simple_model_fn,
        train_input_fn=self._simple_input_function,
        eval_input_fn=self._simple_input_function,
        params=params)


if __name__ == "__main__":
  tf.test.main()

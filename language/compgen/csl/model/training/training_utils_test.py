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
"""Tests for training_utils with different training strategies.

These tests also serve as a sort of integration test between
the generation of tf.Examples, input_utils, and training_utils.
"""

import os

from language.compgen.csl.model import test_utils
from language.compgen.csl.model import weighted_model
from language.compgen.csl.model.data import example_converter
from language.compgen.csl.model.training import input_utils
from language.compgen.csl.model.training import training_utils
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


def _write_tf_examples(examples_filepath, config, num_examples=8):
  """Write examples as test data."""
  rules = [
      qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
      qcfg_rule.rule_from_string("bar ### bar"),
      qcfg_rule.rule_from_string("foo bar ### foo bar"),
  ]

  converter = example_converter.ExampleConverter(rules, config)
  example = ("foo bar", "foo bar")

  writer = tf.io.TFRecordWriter(examples_filepath)
  for _ in range(num_examples):
    tf_example = converter.convert(example)
    writer.write(tf_example.SerializeToString())


def _run_model_with_strategy(strategy, config, dataset_fn):
  dataset_iterator = iter(
      strategy.experimental_distribute_datasets_from_function(dataset_fn))
  batch_size = int(config["batch_size"] / strategy.num_replicas_in_sync)
  with strategy.scope():
    model = weighted_model.Model(
        batch_size, config, training=True, verbose=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=config["learning_rate"])
    train_for_n_steps_fn = training_utils.get_train_for_n_steps_fn(
        config, strategy, optimizer, model)
    mean_loss = train_for_n_steps_fn(
        dataset_iterator,
        tf.convert_to_tensor(config["steps_per_iteration"], dtype=tf.int32))
    return mean_loss


def _run_model(config, dataset_fn):
  batch_size = config["batch_size"]
  model = weighted_model.Model(batch_size, config, training=True, verbose=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=config["learning_rate"])

  training_step = training_utils.get_training_step(config, optimizer, model)
  mean_loss = training_step(next(iter(dataset_fn(ctx=None))))
  return mean_loss


class TrainingUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(TrainingUtilsTest, self).setUp()
    self.config = test_utils.get_test_config()
    examples_filepath = os.path.join(self.get_temp_dir(), "examples.tfrecord")
    _write_tf_examples(examples_filepath, self.config)
    self.dataset_fn = input_utils.get_dataset_fn(examples_filepath, self.config)

  def test_model_no_strategy(self):
    mean_loss = _run_model(self.config, self.dataset_fn)
    self.assertIsNotNone(mean_loss)

  def test_model_one_device(self):
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    mean_loss = _run_model_with_strategy(strategy, self.config, self.dataset_fn)
    self.assertIsNotNone(mean_loss)

  def test_model_mirrored(self):
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1"])
    mean_loss = _run_model_with_strategy(strategy, self.config, self.dataset_fn)
    self.assertIsNotNone(mean_loss)

  def test_get_batch_emb_idxs(self):
    inputs = next(iter(self.dataset_fn(ctx=None)))
    batch_emb_idxs, batch_emb_idx_list = training_utils.get_batch_emb_idxs(
        inputs["lhs_emb_idx_list"], 4, self.config)
    # inputs["lhs_emb_idx_list"] = tf.convert_to_tensor(
    #     [[[-1, -1, 0, 2, 1, -1, 0, 0], [-1, -1, -1, -1, -1, -1, 0, 0]],
    #      [[-1, -1, 0, 2, 1, -1, 0, 0], [-1, -1, -1, -1, -1, -1, 0, 0]],
    #      [[-1, -1, 0, 2, 1, -1, 0, 0], [-1, -1, -1, -1, -1, -1, 0, 0]],
    #      [[-1, -1, 0, 2, 1, -1, 0, 0], [-1, -1, -1, -1, -1, -1, 0, 0]]])
    expected_emb_idxs = tf.convert_to_tensor([0, 0, 2, 1])
    expected_emb_idx_list = tf.convert_to_tensor(
        [[[-1, -1, 1, 2, 3, -1, 1, 1], [-1, -1, -1, -1, -1, -1, 1, 1]],
         [[-1, -1, 1, 2, 3, -1, 1, 1], [-1, -1, -1, -1, -1, -1, 1, 1]],
         [[-1, -1, 1, 2, 3, -1, 1, 1], [-1, -1, -1, -1, -1, -1, 1, 1]],
         [[-1, -1, 1, 2, 3, -1, 1, 1], [-1, -1, -1, -1, -1, -1, 1, 1]]])
    self.assertAllEqual(expected_emb_idxs, batch_emb_idxs)
    self.assertAllEqual(expected_emb_idx_list, batch_emb_idx_list)


def setUpModule():
  # Setup virtual CPUs.
  cpus = tf.config.list_physical_devices("CPU")
  tf.config.set_logical_device_configuration(
      cpus[-1], [tf.config.LogicalDeviceConfiguration()] * 2
  )


if __name__ == "__main__":
  tf.test.main()

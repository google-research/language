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
"""Run model training.

Currently, CPU and parallel GPU training is supported. TPU training is not
currently supported.
"""

import os

from absl import app
from absl import flags
from absl import logging
from language.compgen.csl.common import json_utils
from language.compgen.csl.model import weighted_model
from language.compgen.csl.model.training import input_utils
from language.compgen.csl.model.training import training_utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input", "",
    "TFRecord(s) of tf.Examples (use * for matching multiple files).")

flags.DEFINE_string("model_dir", "", "Directory to save model files.")

flags.DEFINE_string("config", "", "Config json file.")

flags.DEFINE_bool("restore_checkpoint", True,
                  "Whether to restore checkpoint if one exists in model_dir.")

flags.DEFINE_bool("use_gpu", False, "Whether to use GPU for training.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")


def train_model(strategy):
  """Run model training."""
  config = json_utils.json_file_to_dict(FLAGS.config)
  dataset_fn = input_utils.get_dataset_fn(FLAGS.input, config)

  writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "train"))

  dataset_iterator = iter(
      strategy.experimental_distribute_datasets_from_function(dataset_fn))

  batch_size = int(config["batch_size"] / strategy.num_replicas_in_sync)
  logging.info("num_replicas: %s.", strategy.num_replicas_in_sync)
  logging.info("per replica batch_size: %s.", batch_size)

  with strategy.scope():
    model = weighted_model.Model(
        batch_size, config, training=True, verbose=FLAGS.verbose)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    train_for_n_steps_fn = training_utils.get_train_for_n_steps_fn(
        config, strategy, optimizer, model)

    current_step = tf.Variable(
        0, trainable=False, name="current_step", dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, model=model, current_step=current_step)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
    if FLAGS.restore_checkpoint and latest_checkpoint:
      status = checkpoint.restore(latest_checkpoint)
      status.assert_existing_objects_matched()
      logging.info("Restoring %s", latest_checkpoint)
    current_step = checkpoint.current_step.numpy().item()
    logging.info("Start training at step %s", current_step)

    with writer.as_default():
      while current_step < config["training_steps"]:
        logging.info("current_step: %s.", current_step)
        # TODO(linluqiu): investigate how "steps_per_iteration" influences
        # model performance.
        mean_loss = train_for_n_steps_fn(
            dataset_iterator,
            tf.convert_to_tensor(config["steps_per_iteration"], dtype=tf.int32))
        logging.info("loss: %s.", mean_loss)
        logging.info("scores: %s.", mean_loss)
        tf.summary.scalar("loss", mean_loss, step=current_step)
        current_step += config["steps_per_iteration"]
        checkpoint.current_step.assign_add(config["steps_per_iteration"])

        if current_step and (current_step % config["save_checkpoint_every"] == 0
                             or current_step >= config["training_steps"]):
          checkpoint_prefix = os.path.join(FLAGS.model_dir,
                                           "ckpt-%s" % current_step)
          logging.info("Saving checkpoint to %s.", checkpoint_prefix)
          checkpoint.save(file_prefix=checkpoint_prefix)


def main(unused_argv):
  if FLAGS.use_gpu:
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Number of devices: %d", strategy.num_replicas_in_sync)
    train_model(strategy)
  else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    train_model(strategy)


if __name__ == "__main__":
  app.run(main)

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

from language.nqg.model.parser import config_utils
from language.nqg.model.parser import nqg_model
from language.nqg.model.parser.training import input_utils
from language.nqg.model.parser.training import training_utils

import tensorflow as tf

from official.nlp import optimization
from official.nlp.bert import configs


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input", "",
    "TFRecord(s) of tf.Examples (use * for matching multiple files).")

flags.DEFINE_string("model_dir", "", "Directory to save model files.")

flags.DEFINE_string(
    "bert_dir", "",
    "Directory for BERT, including config and (optionally) checkpoint.")

flags.DEFINE_string("config", "", "Config json file.")

flags.DEFINE_bool("restore_checkpoint", False,
                  "Whether to restore checkpoint if one exists in model_dir.")

flags.DEFINE_bool(
    "init_bert_checkpoint", True,
    "If True, init from checkpoint in bert_dir, otherwise use random init.")

flags.DEFINE_bool("use_gpu", False, "Whether to use GPU for training.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")


def train_model(strategy):
  """Run model training."""
  config = config_utils.json_file_to_dict(FLAGS.config)
  dataset_fn = input_utils.get_dataset_fn(FLAGS.input, config)

  writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "train"))

  dataset_iterator = iter(
      strategy.experimental_distribute_datasets_from_function(dataset_fn))

  bert_config = configs.BertConfig.from_json_file(
      os.path.join(FLAGS.bert_dir, "bert_config.json"))
  logging.info("Loaded BERT config: %s", bert_config.to_dict())
  batch_size = int(config["batch_size"] / strategy.num_replicas_in_sync)
  logging.info("num_replicas: %s.", strategy.num_replicas_in_sync)
  logging.info("per replica batch_size: %s.", batch_size)

  with strategy.scope():
    model = nqg_model.Model(
        batch_size, config, bert_config, training=True, verbose=FLAGS.verbose)
    optimizer = optimization.create_optimizer(config["learning_rate"],
                                              config["training_steps"],
                                              config["warmup_steps"])
    train_for_n_steps_fn = training_utils.get_train_for_n_steps_fn(
        strategy, optimizer, model)

    if FLAGS.init_bert_checkpoint:
      bert_checkpoint = tf.train.Checkpoint(model=model.bert_encoder)
      bert_checkpoint_path = os.path.join(FLAGS.bert_dir, "bert_model.ckpt")
      logging.info("Restoring bert checkpoint: %s", bert_checkpoint_path)
      logging.info("Bert vars: %s", model.bert_encoder.trainable_variables)
      logging.info("Checkpoint vars: %s",
                   tf.train.list_variables(bert_checkpoint_path))
      status = bert_checkpoint.restore(bert_checkpoint_path)
      status.assert_existing_objects_matched()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    current_step = 0

    if FLAGS.restore_checkpoint:
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
      # TODO(petershaw): This is a hacky way to read current step.
      current_step = int(latest_checkpoint.split("-")[-2])
      logging.info("Restoring %s at step %s.", latest_checkpoint, current_step)
      status = checkpoint.restore(latest_checkpoint)
      status.assert_existing_objects_matched()

    with writer.as_default():
      while current_step < config["training_steps"]:
        logging.info("current_step: %s.", current_step)
        mean_loss = train_for_n_steps_fn(
            dataset_iterator,
            tf.convert_to_tensor(config["steps_per_iteration"], dtype=tf.int32))
        tf.summary.scalar("loss", mean_loss, step=current_step)
        current_step += config["steps_per_iteration"]

        if current_step and current_step % config["save_checkpoint_every"] == 0:
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

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
"""Run Jax experiments."""

import json
import os


from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from language.mentionmemory.training.trainer import train
import ml_collections
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

# Training config.
flags.DEFINE_string("config", None, "Train configuration serialized as JSON.")
flags.DEFINE_string(
    "config_file", None,
    "Path to file that contains JSON serialized model configuration. If this"
    "is specified, ignores 'config' parameter.")
flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

# Hyper parameters
flags.DEFINE_float("learning_rate", None, "Learning rate")
flags.DEFINE_integer("per_device_batch_size", None, "Per device batch size")
flags.DEFINE_integer("num_train_steps", None, "Number of training steps")
flags.DEFINE_integer("warmup_steps", None, "Number of warmup training steps")


def validate_config_flags(flag_dict):
  return flag_dict["config"] is not None or flag_dict["config_file"] is not None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.model_dir, "model_dir")

  tf.io.gfile.makedirs(FLAGS.model_dir)

  # Process config here
  if FLAGS.config_file:
    with tf.io.gfile.GFile(FLAGS.config_file, "r") as reader:
      config = json.load(reader)
  else:
    config = json.loads(FLAGS.config)
    # # Save config to workdir if it's not yet exists
    if jax.process_index() == 0:
      config_file = os.path.join(FLAGS.model_dir, "config.json")
      with tf.io.gfile.GFile(config_file, "w") as writer:
        writer.write(json.dumps(config, indent=4))

  config["model_dir"] = FLAGS.model_dir
  if FLAGS.learning_rate is not None:
    config["learning_rate"] = FLAGS.learning_rate
  if FLAGS.per_device_batch_size is not None:
    config["per_device_batch_size"] = FLAGS.per_device_batch_size
  if FLAGS.num_train_steps is not None:
    config["num_train_steps"] = FLAGS.num_train_steps
  if FLAGS.warmup_steps is not None:
    config["warmup_steps"] = FLAGS.warmup_steps

  train(ml_collections.ConfigDict(config))


if __name__ == "__main__":
  flags.register_multi_flags_validator(["config", "config_file"],
                                       validate_config_flags,
                                       "Either --config or --config_file needs "
                                       "to be set.")
  app.run(main)

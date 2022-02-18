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
"""Main file for training or testing the model."""

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax

from language.gscan.xattn_model import predict
from language.gscan.xattn_model import train

from ml_collections import config_flags

import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', '', 'Training configuration.', lock_config=True)
flags.DEFINE_string('workdir', None, 'Work unit directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'predict'],
                  'Whether to do training or testing.')
flags.DEFINE_string('ckpt_path', None, 'The checkpoint to use.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  if FLAGS.mode == 'train':
    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  else:
    predict.predict_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.ckpt_path)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  app.run(main)

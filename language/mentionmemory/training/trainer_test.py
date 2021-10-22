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
"""Tests for trainer."""

from absl.testing import absltest
from language.mentionmemory.training import trainer
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class TrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    config_dict = {
        'task_name': 'example_task',
        'model_config': {
            'features': [32, 32],
            'dtype': 'float32',
        },
        'seed': 0,
        'num_train_steps': 5,
        'learning_rate': 1e-4,
        'warmup': False,
        'linear_decay': False,
        'weight_decay': 0.0,
        'weight_decay_exclude': ['layer_norm', 'bias'],
        'grad_clip': 1.0,
        'per_device_batch_size': 1,
        'train_data': [{
            'patterns': '/tmp/test/test_dataset.tfrecords',
            'samples_per_example': 1,
        }],
        'eval_data': [{
            'patterns': '/tmp/test/test_dataset.tfrecords',
            'samples_per_example': 1,
        }],
        'model_dir': '/tmp/test/',
        'restore_checkpoint': False,
        'save_checkpoints': False,
        'load_weights': None,
        'checkpoint_every_steps': 5,
        'save_every_steps': None,
        'save_samples_every_steps': 1,
        'eval_every_steps': 1,
        'num_eval_steps': 5,
        'ignore_k_nans': 2,
    }

    self.test_config = ml_collections.ConfigDict(config_dict)

    dataset_size = 8
    tf.io.gfile.makedirs(self.test_config.model_dir)
    writer = tf.io.TFRecordWriter(self.test_config.train_data[0]['patterns'])
    for _ in range(dataset_size):
      x = np.random.random(self.test_config.model_config.features[0])

      record_bytes = tf.train.Example(
          features=tf.train.Features(feature={
              'x': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
          })).SerializeToString()
      writer.write(record_bytes)

  def test_multi_node_training(self):
    test_utils.force_multi_devices(8)
    trainer.train(self.test_config)


if __name__ == '__main__':
  absltest.main()

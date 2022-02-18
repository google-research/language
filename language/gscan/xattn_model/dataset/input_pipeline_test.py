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
"""Tests for input_pipeline."""

from absl import flags
import jax

from language.gscan.xattn_model import test_utils
from language.gscan.xattn_model.dataset import input_pipeline

import tensorflow as tf

FLAGS = flags.FLAGS


class InputPipelineTest(tf.test.TestCase):

  def test_create_datasets(self):
    config = test_utils.get_dataset_test_config()
    batch_size = config.train_per_device_batch_size
    max_seq_len = config.max_seq_len
    max_target_seq_len = config.max_target_seq_len
    grid_size = config.grid_size
    img_dim = config.img_dim

    rng = jax.random.PRNGKey(0)
    ds, _ = input_pipeline.create_datasets(config, rng)
    actual_shapes = {k: v.shape.as_list() for k, v in ds.element_spec.items()}
    self.assertEqual(
        actual_shapes, {
            'index': [1, batch_size, 1],
            'token': [1, batch_size, max_seq_len],
            'txt_mask': [1, batch_size, max_seq_len],
            'target_token': [1, batch_size, max_target_seq_len],
            'target_txt_mask': [1, batch_size, max_target_seq_len],
            'image': [1, batch_size, grid_size, grid_size, img_dim]
        })
    actual_types = {k: v.dtype for k, v in ds.element_spec.items()}
    self.assertEqual(
        actual_types, {
            'index': tf.int64,
            'token': tf.int64,
            'txt_mask': tf.int64,
            'target_token': tf.int64,
            'target_txt_mask': tf.int64,
            'image': tf.float32
        })


if __name__ == '__main__':
  tf.test.main()

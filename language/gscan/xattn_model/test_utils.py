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
"""Utils for testing."""

import os

from absl import flags
from language.gscan.xattn_model.configs import compositional

FLAGS = flags.FLAGS


def get_dataset_test_config():
  """Returns the test config for dataset."""
  batch_size = 4
  grid_size = 4
  img_dim = 15
  config = compositional.get_dataset_config()
  data_dir = os.path.join(
      FLAGS.test_srcdir,
      'language/gscan/xattn_model/testdata')
  config.data_dir = data_dir
  config.grid_size = grid_size
  config.train_split = 'train'
  config.eval_split = 'train'
  config.test_splits = ['train']
  config.split = 'test'
  config.num_epochs = 1
  config.img_dim = img_dim
  config.train_per_device_batch_size = batch_size
  config.eval_per_device_batch_size = batch_size

  return config


def get_model_test_config():
  """Returns the test config for model."""
  vocab_size = 10
  hidden_dim = 16
  intermediate_dim = 32
  num_heads = 2
  num_layers = 1
  config = compositional.get_model_config()
  config.vocab_size = vocab_size
  config.target_vocab_size = vocab_size
  config.bi_hidden_dim = hidden_dim
  config.l_hidden_dim = hidden_dim
  config.v_hidden_dim = hidden_dim
  config.l_intermediate_dim = intermediate_dim
  config.v_intermediate_dim = intermediate_dim
  config.bi_num_heads = num_heads
  config.l_num_heads = num_heads
  config.v_num_heads = num_heads
  config.decode_num_heads = num_heads
  config.l_num_layers = num_layers
  config.v_num_layers = num_layers
  config.bi_num_layers = num_layers
  config.decode_num_layers = num_layers
  config.max_decode_step = 1

  return config


def get_test_config():
  """Returns the test config."""
  config = compositional.get_config()
  config.dataset = get_dataset_test_config()
  config.model = get_model_test_config()

  config.num_train_steps = 1
  config.num_eval_steps = 1
  config.num_test_steps = 1

  return config

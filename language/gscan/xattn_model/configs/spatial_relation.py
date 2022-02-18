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
"""A default config for gSCAN spatial relation splits."""

import ml_collections


def get_dataset_config():
  """Config for dataset and input pipeline."""
  config = ml_collections.ConfigDict()
  config.data_dir = 'data/spatial_relation_splits'
  config.grid_size = 6
  config.max_seq_len = 15
  config.max_target_seq_len = 25
  config.num_epochs = 40
  config.img_dim = 16
  config.eos_idx = 2
  config.train_split = 'train'
  config.eval_split = 'dev'
  config.test_splits = [
      'test', 'visual', 'relation', 'referent', 'relative_position_1',
      'relative_position_2'
  ]
  config.split = 'spatial_relation_splits'
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = True
  config.test_pad_last_batch = True
  config.shuffle_buffer_size = 1000
  config.train_per_device_batch_size = 128
  config.eval_per_device_batch_size = 1000
  config.test_per_device_batch_size = 1000

  return config


def get_model_config():
  """Config for Transformer."""
  config = ml_collections.ConfigDict()
  config.vocab_size = 23
  config.target_vocab_size = 8
  config.max_decode_step = 40
  config.cross_attn = True
  config.bi_hidden_dim = 128
  config.v_hidden_dim = 128
  config.l_hidden_dim = 128
  config.l_intermediate_dim = 256
  config.v_intermediate_dim = 256
  config.bi_num_heads = 8
  config.l_num_heads = 8
  config.v_num_heads = 8
  config.decode_num_heads = 8
  config.l_num_layers = 6
  config.v_num_layers = 6
  config.bi_num_layers = 6
  config.decode_num_layers = 6
  config.num_conv_channels = 50
  config.conv_kernel_sizes = (1, 5, 7)
  config.beam_size = 1

  return config


def get_config():
  """Config for training and evaluation."""
  config = ml_collections.ConfigDict()

  # Training config
  config.learning_rate = 16e-4
  config.learning_rate_schedule = 'step'
  config.learning_rate_step_boundaries = [0.5, 0.75]
  config.learning_rate_weight_decay = 0.01
  config.optimizer = 'AdamW'
  config.no_weight_decay = ['bias', 'LayerNorm/bias', 'LayerNorm/scale']
  config.warmup_proportion = 0.1
  config.grad_clip = 0.5

  # If num_train_steps == -1 then the number of training steps is calculated
  # from num_epochs.
  config.num_train_steps = -1
  config.num_eval_steps = -1
  config.num_test_steps = -1
  config.num_profile_steps = 1
  config.log_loss_every_steps = 20
  # If eval_every_steps == -1 then the model is not evaluated.
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 1000

  config.seed = 0
  config.dataset = get_dataset_config()
  config.model = get_model_config()

  return config



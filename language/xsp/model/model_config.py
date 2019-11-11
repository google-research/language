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
"""Defines a model configuration as a nested namedtuple."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import tensorflow.gfile as gfile


class DataOptions(
    collections.namedtuple(
        'DataOptions',
        ['bert_vocab_path', 'max_num_tokens', 'max_decode_length'])):
  pass


class ModelParameters(
    collections.namedtuple('ModelParameters', [
        'use_segment_ids', 'use_foreign_key_features', 'use_alignment_features',
        'pretrained_bert_dir', 'source_embedding_dims', 'target_embedding_dims',
        'encoder_dims', 'decoder_dims', 'max_decoder_relative_distance',
        'num_decoder_layers', 'num_heads', 'decoder_ff_layer_hidden_size'
    ])):
  pass


class TrainingOptions(
    collections.namedtuple('TrainingOptions', [
        'tpu_iterations_per_loop', 'batch_size', 'training_steps',
        'layer_dropout_rate', 'optimizer_learning_rate',
        'optimizer_warmup_steps', 'freeze_pretrained_steps',
        'after_restart_learning_rate'
    ])):
  pass


class ModelConfig(
    collections.namedtuple(
        'ModelConfig',
        ['data_options', 'model_parameters', 'training_options'])):

  def __str__(self):
    return json.dumps(
        {
            'data_options': self.data_options._asdict(),
            'model_parameters': self.model_parameters._asdict(),
            'training_options': self.training_options._asdict()
        },
        indent=4)


def load_config(filename):
  """Loads a serialized model config into a ModelConfig object."""
  with gfile.Open(filename) as infile:
    model_dict = json.load(infile)

  # Load the data options
  data_options = DataOptions(**model_dict['data_options'])
  model_parameters = ModelParameters(**model_dict['model_parameters'])
  training_options = TrainingOptions(**model_dict['training_options'])

  model_config = ModelConfig(
      data_options=data_options,
      model_parameters=model_parameters,
      training_options=training_options)

  return model_config

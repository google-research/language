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
"""Utilities for pre-training with BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bert import modeling
import tensorflow.compat.v1 as tf


def get_input_mask(max_seq_len, source_len):
  """Returns input mask for input to BERT model."""
  sequence_mask = tf.sequence_mask(source_len, max_seq_len)

  return tf.to_int32(sequence_mask)


def get_bert_config(bert_dir, reinitialize_type_embeddings=False):
  """Returns BertConfig given BERT directory."""
  bert_config_file = os.path.join(bert_dir, "bert_config.json")
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  if reinitialize_type_embeddings:
    # NOTE: Setting the type vocab size here to something much larger. We are
    # using segment IDs to represent spans of table information, so this needs
    # to be greater than two (the default). Apparently, there's a silent bug if
    # it's not set, so we are setting it now to something large.
    bert_config.type_vocab_size = 512

  return bert_config


def get_bert_embeddings(input_ids,
                        bert_config,
                        input_mask=None,
                        token_type_ids=None,
                        is_training=False,
                        use_one_hot_embeddings=False,
                        scope=None):
  """Returns embeddings for BERT."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=token_type_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope)
  return model.get_sequence_output()


def get_bert_pooled_output(input_ids,
                           input_mask,
                           bert_config,
                           is_training=False,
                           use_one_hot_embeddings=False,
                           scope=None):
  """Returns embeddings for BERT pooled for each input.

  Args:
      input_ids: <int32>[batch_size, max_seq_len].
      input_mask: <int32>[batch_size, max_seq_len].
      bert_config: BertConfig instance.
      is_training: Determines whether dropout is used.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.nn.embedding_lookup()`.
      scope: Variable scope to use.

  Returns:
    <float>[batch_size, hidden_size] corresponding to the pooled BERT
    transformer encoder hidden states.
  """
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope)
  return model.get_pooled_output()

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
"""A collection of attention mechanisms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.consistent_zero_shot_nmt.utils import common_utils as U
import tensorflow as tf


__all__ = [
    "get",
]


def get(attention_type, num_units, memory, memory_sequence_length,
        scope=None, reuse=None):
  """Returns attention mechanism according to the specified type."""
  with tf.variable_scope(scope, reuse=reuse):
    if attention_type == U.ATT_LUONG:
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units=num_units,
          memory=memory,
          memory_sequence_length=memory_sequence_length)
    elif attention_type == U.ATT_LUONG_SCALED:
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units=num_units,
          memory=memory,
          memory_sequence_length=memory_sequence_length,
          scale=True)
    elif attention_type == U.ATT_BAHDANAU:
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units=num_units,
          memory=memory,
          memory_sequence_length=memory_sequence_length)
    elif attention_type == U.ATT_BAHDANAU_NORM:
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units=num_units,
          memory=memory,
          memory_sequence_length=memory_sequence_length,
          normalize=True)
    else:
      raise ValueError("Unknown attention type: %s" % attention_type)
  return attention_mechanism

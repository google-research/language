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
"""Utilities for defining evaluation metrics during training."""
from language.xsp.model import decode_utils
import tensorflow.compat.v1 as tf


def _sequence_correct(labels, predictions):
  """Computes a per-example sequence accuracy."""
  target_decode_steps = decode_utils.decode_steps_from_labels(
      labels, trim_start_symbol=True)
  predicted_decode_steps = decode_utils.decode_steps_from_predictions(
      predictions)

  decode_utils.assert_shapes_match(target_decode_steps, predicted_decode_steps)

  equal_tokens = decode_utils.compare_decode_steps(target_decode_steps,
                                                   predicted_decode_steps)
  target_len = labels["target_len"] - 1
  loss_mask = tf.sequence_mask(
      lengths=tf.to_int32(target_len),
      maxlen=tf.to_int32(tf.shape(equal_tokens)[1]))
  equal_tokens = tf.logical_or(equal_tokens, tf.logical_not(loss_mask))
  all_equal = tf.cast(tf.reduce_all(equal_tokens, 1), tf.float32)
  return all_equal


def create_metrics_ops(labels, predictions):
  """Creates metrics ops for evaluation."""
  metric_ops = dict()

  sequence_correct = _sequence_correct(labels, predictions)
  metric_ops["sequence_correct"] = tf.metrics.mean(sequence_correct)

  return metric_ops

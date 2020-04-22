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
"""Utilities for expressing decoder steps.

We use a formulation similar to Pointer-Generator networks.
Therefore, we have two equivalent representations for decoder outputs:
1) As a DecodeSteps tuple
2) As Tensor of integer indices in the extended vocabulary, which is a
concatenation of the output vocabulary and the source, in that order.
"""

import collections

from language.xsp.model import constants
import tensorflow.compat.v1 as tf

# Represents a decoder step.
# Each tensor is expected to be shape [batch_size, output_length].
# action_types: Integer tensor corresponding to
#              internal_pb2.OutputStep enum values.
# action_ids: Integer tensor with semantics dependent on action_type.
DecodeSteps = collections.namedtuple("DecodeSteps",
                                     ["action_types", "action_ids"])

# Represents the range in the extended action id space for each action type.
# end_index is non-inclusive.
ActionTypeRange = collections.namedtuple(
    "ActionTypeRange", ["action_type", "start_index", "end_index"])


def _get_action_types_to_range(output_vocab_size, model_config):
  """Returns a list of ActionTypeRange tuples."""
  # Tuples of (action_type, length).
  action_type_tuples = [(constants.GENERATE_ACTION, output_vocab_size),
                        (constants.COPY_ACTION,
                         model_config.data_options.max_num_tokens)]
  index = 0
  action_type_ranges = []
  for action_type, length in action_type_tuples:
    end_index = index + length
    action_type_ranges.append(
        ActionTypeRange(
            action_type=action_type, start_index=index, end_index=end_index))
    index = end_index
  return action_type_ranges


def _get_action_type(extended_indices, output_vocab_size, model_config):
  """Returns action_type tensor."""
  action_type = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    index_in_range = tf.logical_and(
        tf.greater_equal(extended_indices, action_type_range.start_index),
        tf.less(extended_indices, action_type_range.end_index))
    action_type += (
        tf.to_int64(index_in_range) * tf.constant(
            action_type_range.action_type, dtype=tf.int64))
  return action_type


def _get_action_id(extended_indices, action_types, output_vocab_size,
                   model_config):
  """Returns action_id tensor."""
  # This initial value will be broadcast to the length of decode_steps.
  action_ids = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    is_type = tf.equal(
        tf.constant(action_type_range.action_type, dtype=tf.int64),
        action_types)
    # For each timestep, exactly one of the action_type_ranges will be added,
    # so this sum will populate each entry on exactly one iteration.
    action_ids += (
        tf.to_int64(is_type) *
        (extended_indices - action_type_range.start_index))
  return action_ids


def get_decode_steps(extended_indices, output_vocab_size, model_config):
  """Convert Tensor of indices in extended vocabulary to DecodeStep."""
  extended_indices = tf.to_int64(extended_indices)
  action_types = _get_action_type(extended_indices, output_vocab_size,
                                  model_config)
  action_ids = _get_action_id(extended_indices, action_types, output_vocab_size,
                              model_config)
  return DecodeSteps(action_types=action_types, action_ids=action_ids)


def get_extended_indices(decode_steps, output_vocab_size, model_config):
  """Convert DecodeSteps into a tensor of extended action ids."""
  # This initial value will be broadcast to the length of decode_steps.
  extended_action_indices = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    is_type = tf.equal(
        tf.constant(action_type_range.action_type, dtype=tf.int64),
        decode_steps.action_types)
    # For each timestep, exactly one of the action_type_ranges will be added,
    # so this sum will populate each entry on exactly one iteration.
    extended_action_indices += (
        tf.to_int64(is_type) *
        (decode_steps.action_ids + action_type_range.start_index))
  return extended_action_indices


def concat_logits(generate_logits, copy_logits):
  return tf.concat([generate_logits, copy_logits], axis=2)


def compare_decode_steps(decode_steps_a, decode_steps_b):
  """Returns tensor of bools indicated whether decode steps are equal."""
  return tf.reduce_all(
      tf.stack([
          tf.equal(decode_steps_a.action_types, decode_steps_b.action_types),
          tf.equal(decode_steps_a.action_ids, decode_steps_b.action_ids),
      ],
               axis=0),
      axis=0)


def compare_generating_steps(target_decode_steps, predicted_decode_steps):
  """Compare generating steps only but ignoring target copying steps.

  Args:
    target_decode_steps: Target DecodeSteps, Each tensor is expected to be shape
      [batch_size, output_length].
    predicted_decode_steps: Predicted DecodeSteps, Each tensor is expected to be
      shape [batch_size, output_length].

  Returns:
    A tensor of bools indicating whether generating steps are equal.
    Copy Steps will have value True.
  """
  # Set all copying steps to True, Since we only care about generating steps.
  return tf.logical_or(
      tf.not_equal(target_decode_steps.action_types, constants.GENERATE_ACTION),
      tf.logical_and(
          tf.equal(target_decode_steps.action_types,
                   predicted_decode_steps.action_types),
          tf.equal(target_decode_steps.action_ids,
                   predicted_decode_steps.action_ids)))


def assert_shapes_match(decode_steps_a, decode_steps_b):
  decode_steps_a.action_types.get_shape().assert_is_compatible_with(
      decode_steps_b.action_types.get_shape())
  decode_steps_a.action_ids.get_shape().assert_is_compatible_with(
      decode_steps_b.action_ids.get_shape())


def decode_steps_from_labels(labels,
                             trim_start_symbol=False,
                             trim_end_symbol=False):
  """Returns DecodeSteps given labels dict."""
  action_types = labels["target_action_types"]
  action_ids = labels["target_action_ids"]
  if trim_end_symbol:
    action_types = action_types[:, :-1]
    action_ids = action_ids[:, :-1]
  if trim_start_symbol:
    action_types = action_types[:, 1:]
    action_ids = action_ids[:, 1:]
  return DecodeSteps(action_types=action_types, action_ids=action_ids)


def decode_steps_from_predictions(predictions):
  """Returns DecodeSteps given predictions dict."""
  return DecodeSteps(
      action_types=predictions[constants.PREDICTED_ACTION_TYPES],
      action_ids=predictions[constants.PREDICTED_ACTION_IDS])


def get_labels(decode_steps):
  """Returns labels dict given DecodeSteps."""
  return dict(
      target_action_types=decode_steps.action_types,
      target_action_ids=decode_steps.action_ids)


def get_predictions(decode_steps):
  """Returns predictions dict given DecodeSteps."""
  return dict(
      predicted_action_types=decode_steps.action_types,
      predicted_action_ids=decode_steps.action_ids)

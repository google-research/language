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
"""Defines input pipeline for processing TFRecords."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from language.xsp.model import constants
from language.xsp.model import input_utils
from language.xsp.model import sequence_example_decoder

from six.moves import zip
import tensorflow as tf

# Keys used for the feature and label output dicts.
FEATURE_KEYS = [
    constants.SOURCE_WORDPIECES_KEY, constants.SOURCE_LEN_KEY,
    constants.LANGUAGE_KEY, constants.REGION_KEY, constants.TAG_KEY,
    constants.COPIABLE_INPUT_KEY
]

LABEL_KEYS = [
    constants.OUTPUT_TYPE_KEY, constants.WEIGHT_KEY,
    constants.TARGET_ACTION_TYPES_KEY, constants.TARGET_ACTION_IDS_KEY,
    constants.TARGET_LEN_KEY
]

# Delete string tensors on TPU because we cannot pass strings to TPU.
STRING_FEATURE_KEYS = [
    constants.LANGUAGE_KEY, constants.REGION_KEY, constants.TAG_KEY
]


def _get_target_action_types(keys_to_tensors):
  tensor = keys_to_tensors[constants.TARGET_ACTION_TYPES_KEY]
  # Modify tensor for start and end symbols.
  tensor = tf.concat([[1], tensor, [1]], 0)
  return tensor


def _get_target_action_ids(keys_to_tensors):
  tensor = keys_to_tensors[constants.TARGET_ACTION_IDS_KEY]
  # Modify tensor for start and end symbols.
  tensor = tf.concat([[constants.TARGET_START_SYMBOL_ID], tensor,
                      [constants.TARGET_END_SYMBOL_ID]], 0)
  return tensor


def _get_target_len(keys_to_tensors):
  size = tf.size(keys_to_tensors[constants.TARGET_ACTION_TYPES_KEY])
  # Add 2 for start and end symbols.
  return tf.add(size, 2)


def get_context_features():
  keys_to_context_features = {
      constants.LANGUAGE_KEY: tf.FixedLenFeature([], dtype=tf.string),
      constants.REGION_KEY: tf.FixedLenFeature([], dtype=tf.string),
      constants.OUTPUT_TYPE_KEY: tf.FixedLenFeature([], dtype=tf.int64),
      constants.WEIGHT_KEY: tf.FixedLenFeature([], dtype=tf.float32),
      constants.TAG_KEY: tf.VarLenFeature(dtype=tf.string),
  }

  return keys_to_context_features


def get_sequence_features(use_segment_ids, use_foreign_key_features,
                          string_alignment_features):
  """Gets sequence features (i.e., for input/output sequence to the model)."""
  keys_to_sequence_features = {
      constants.SOURCE_WORDPIECES_KEY:
          tf.FixedLenSequenceFeature([], dtype=tf.int64),
      constants.TARGET_ACTION_TYPES_KEY:
          tf.FixedLenSequenceFeature([], dtype=tf.int64),
      constants.TARGET_ACTION_IDS_KEY:
          tf.FixedLenSequenceFeature([], dtype=tf.int64),
      constants.COPIABLE_INPUT_KEY:
          tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }

  if use_segment_ids:
    keys_to_sequence_features[
        constants.SEGMENT_ID_KEY] = tf.FixedLenSequenceFeature([],
                                                               dtype=tf.int64)

  if use_foreign_key_features:
    keys_to_sequence_features[
        constants.FOREIGN_KEY_KEY] = tf.FixedLenSequenceFeature([],
                                                                dtype=tf.int64)

  if string_alignment_features:
    keys_to_sequence_features[
        constants.ALIGNED_KEY] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

  return keys_to_sequence_features


def _get_sequence_decoder(use_segment_ids, use_foreign_key_features,
                          string_alignment_features):
  """Returns a TFSequenceExampleDecoder for decoding input format."""
  keys_to_context_features = get_context_features()
  keys_to_sequence_features = get_sequence_features(use_segment_ids,
                                                    use_foreign_key_features,
                                                    string_alignment_features)

  items_to_handlers = {}

  # Context features.
  items_to_handlers[
      constants.OUTPUT_TYPE_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.OUTPUT_TYPE_KEY)
  items_to_handlers[
      constants.WEIGHT_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.WEIGHT_KEY)
  items_to_handlers[
      constants.LANGUAGE_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.LANGUAGE_KEY)
  items_to_handlers[
      constants.REGION_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.REGION_KEY)
  items_to_handlers[
      constants.TAG_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.TAG_KEY, default_value='')

  # Sequence features.
  items_to_handlers[
      constants
      .SOURCE_WORDPIECES_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.SOURCE_WORDPIECES_KEY)
  items_to_handlers[
      constants.COPIABLE_INPUT_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
          constants.COPIABLE_INPUT_KEY)

  items_to_handlers[
      constants.
      TARGET_ACTION_TYPES_KEY] = tf.contrib.slim.tfexample_decoder.ItemHandlerCallback(
          keys=[constants.TARGET_ACTION_TYPES_KEY],
          func=_get_target_action_types)
  items_to_handlers[
      constants.
      TARGET_ACTION_IDS_KEY] = tf.contrib.slim.tfexample_decoder.ItemHandlerCallback(
          keys=[constants.TARGET_ACTION_IDS_KEY], func=_get_target_action_ids)

  # Get lengths of sequences. Easiest to do this now prior to padding.
  items_to_handlers[
      constants
      .SOURCE_LEN_KEY] = tf.contrib.slim.tfexample_decoder.ItemHandlerCallback(
          keys=[constants.SOURCE_WORDPIECES_KEY],
          func=input_utils.get_source_len_fn(constants.SOURCE_WORDPIECES_KEY))
  items_to_handlers[
      constants
      .TARGET_LEN_KEY] = tf.contrib.slim.tfexample_decoder.ItemHandlerCallback(
          keys=[constants.TARGET_ACTION_TYPES_KEY], func=_get_target_len)

  # Extra features.
  if use_segment_ids:
    items_to_handlers[
        constants.SEGMENT_ID_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
            constants.SEGMENT_ID_KEY)

  if use_foreign_key_features:
    items_to_handlers[
        constants.FOREIGN_KEY_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
            constants.FOREIGN_KEY_KEY)

  if string_alignment_features:
    items_to_handlers[
        constants.ALIGNED_KEY] = tf.contrib.slim.tfexample_decoder.Tensor(
            constants.ALIGNED_KEY)

  decoder = sequence_example_decoder.TFSequenceExampleDecoder(
      keys_to_context_features, keys_to_sequence_features, items_to_handlers)

  return decoder


def _remove_keys(keys, keys_to_remove):
  """Remove given keys from list of keys.

  Args:
    keys: List of keys to subtract from.
    keys_to_remove: List of keys to remove.

  Returns:
    A list of keys after subtraction.
  """
  return list(set(keys) - set(keys_to_remove))


def _get_static_padded_shapes(model_config):
  """Return static padded shapes for TPU training."""

  max_input_symbols = model_config.data_options.max_num_tokens
  # Add 2 because of start and end symbol.
  max_decode_length = model_config.data_options.max_decode_length + 2

  padded_shapes = {
      constants.SOURCE_WORDPIECES_KEY: [max_input_symbols],
      constants.COPIABLE_INPUT_KEY: [max_input_symbols],
      constants.SOURCE_LEN_KEY: [],
      constants.TARGET_ACTION_TYPES_KEY: [max_decode_length],
      constants.TARGET_ACTION_IDS_KEY: [max_decode_length],
      constants.TARGET_LEN_KEY: [],
      constants.WEIGHT_KEY: [],
      constants.OUTPUT_TYPE_KEY: []
  }

  if model_config.model_parameters.use_foreign_key_features:
    padded_shapes[constants.FOREIGN_KEY_KEY] = [max_input_symbols]

  if model_config.model_parameters.use_segment_ids:
    padded_shapes[constants.SEGMENT_ID_KEY] = [max_input_symbols]

  if model_config.model_parameters.use_alignment_features:
    padded_shapes[constants.ALIGNED_KEY] = [max_input_symbols]

  return padded_shapes


def get_features_and_labels(data_sources,
                            batch_size,
                            model_config,
                            use_tpu=False,
                            shuffle=False,
                            num_epochs=None,
                            scope='input_fn'):
  """Get dict of features and labels."""
  feature_keys_to_remove = STRING_FEATURE_KEYS if use_tpu else []
  feature_keys = _remove_keys(FEATURE_KEYS, feature_keys_to_remove)

  if model_config.model_parameters.use_segment_ids:
    feature_keys.append(constants.SEGMENT_ID_KEY)
  if model_config.model_parameters.use_foreign_key_features:
    feature_keys.append(constants.FOREIGN_KEY_KEY)
  if model_config.model_parameters.use_alignment_features:
    feature_keys.append(constants.ALIGNED_KEY)

  static_padded_shapes = None
  if use_tpu:
    static_padded_shapes = _get_static_padded_shapes(model_config)

  sequence_decoder = _get_sequence_decoder(
      use_segment_ids=model_config.model_parameters.use_segment_ids,
      use_foreign_key_features=model_config.model_parameters
      .use_foreign_key_features,
      string_alignment_features=model_config.model_parameters
      .use_alignment_features)

  return input_utils.decode_features_and_labels(
      sequence_decoder,
      feature_keys,
      LABEL_KEYS,
      data_sources,
      batch_size,
      static_padded_shapes,
      drop_remainder=use_tpu,
      shuffle=shuffle,
      num_epochs=num_epochs,
      scope=scope)


def create_placeholder_inputs(use_segment_ids, use_foreign_key_features,
                              use_alignment_features):
  """Generate model input with a placeholder for serialized SequenceExample."""
  feature_keys = FEATURE_KEYS

  if use_foreign_key_features:
    feature_keys.append(constants.FOREIGN_KEY_KEY)
  if use_segment_ids:
    feature_keys.append(constants.SEGMENT_ID_KEY)
  if use_alignment_features:
    feature_keys.append(constants.ALIGNED_KEY)

  serialized_example_placeholder = tf.placeholder(
      dtype=tf.string, shape=[], name='input_example_tensor')
  decoder = _get_sequence_decoder(
      use_segment_ids=use_segment_ids,
      use_foreign_key_features=use_foreign_key_features,
      string_alignment_features=use_alignment_features)

  decode_items = feature_keys + LABEL_KEYS
  decoded = decoder.decode(serialized_example_placeholder, decode_items)
  # Expand dimensions to create 'batch' of size 1.
  decoded_dict = {
      k: tf.expand_dims(tensor, 0) for k, tensor in zip(decode_items, decoded)
  }
  features = {k: decoded_dict[k] for k in feature_keys}
  labels = {k: decoded_dict[k] for k in LABEL_KEYS}
  return (serialized_example_placeholder, features, labels)


def create_serving_input_fn(use_segment_ids, use_foreign_key_features,
                            use_alignment_features):
  """Sets up input function that accepts serialized tf.SequenceExample."""

  def input_fn():
    placeholder, features, _ = create_placeholder_inputs(
        use_segment_ids, use_foreign_key_features, use_alignment_features)
    inputs = {constants.SERIALIZED_EXAMPLE: placeholder}
    return tf.estimator.export.ServingInputReceiver(features, inputs)

  return input_fn


def _get_batch_size(model_config, params, use_tpu):
  # params['batch_size'] is per shard batch_size calculated by
  # model_config.training_options.batch_size / num_shards.
  if use_tpu:
    return params['batch_size']
  else:
    return model_config.training_options.batch_size


def create_training_input_fn(model_config, directory, filepaths, use_tpu):
  """Creates an input function that can be used with tf.learn estimators."""

  def input_fn(params):
    return get_features_and_labels(
        [os.path.join(directory, filepath) for filepath in filepaths],
        _get_batch_size(model_config, params, use_tpu),
        model_config,
        use_tpu,
        shuffle=True,
        num_epochs=None,
        scope='train_input_fn')

  return input_fn


def create_eval_input_fn(model_config, directory, filepaths, use_tpu):
  """Creates an input function that can be used with tf.learn estimators."""

  def input_fn(params):
    return get_features_and_labels(
        [os.path.join(directory, filepath) for filepath in filepaths],
        _get_batch_size(model_config, params, use_tpu),
        model_config,
        use_tpu,
        shuffle=True,
        num_epochs=1,
        scope='eval_input_fn')

  return input_fn

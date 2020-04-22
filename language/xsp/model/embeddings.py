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
"""Utilities for generating input embeddings and output embedding table."""
import math

from language.xsp.model import bert_utils
from language.xsp.model import common_layers
from language.xsp.model import constants
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.gfile as gfile

# Added to avoid division by zero.
EPSILON = 0.00000001


def _get_vocab_symbols(filename):
  """Returns a list of symbols in a vocabularly file."""
  vocab = []
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))
  with gfile.GFile(filename) as fp:
    for line in fp:
      vocab.append(line.rstrip("\n"))
  return vocab


def _get_vocab_size(filename):
  return len(_get_vocab_symbols(filename))


def get_output_vocab_size(output_vocab_filepath):
  return (_get_vocab_size(output_vocab_filepath) +
          constants.NUM_RESERVED_OUTPUT_SYMBOLS)


def _default_initializer(model_params):
  return tf.random_normal_initializer(
      0.0, math.pow(model_params.source_embedding_dims, -0.5))


def _ignore_pad(embeddings_table, ids, use_one_hot_embeddings=False):
  """Use mean of symbol embeddings as overall embedding but ignore PAD."""
  source_embeddings = common_layers.embedding_lookup(embeddings_table, ids,
                                                     use_one_hot_embeddings)
  # Set weights to ignore padding.
  embedded_weights = tf.to_float(tf.not_equal(ids, constants.PAD_SYMBOL_ID))
  embedded_weights = tf.expand_dims(embedded_weights, -1)
  return source_embeddings * embedded_weights


def _bert_embeddings(wordpiece_embedding_size, bert_config, features,
                     is_training, use_one_hot_embeddings, scope,
                     use_segment_ids):
  """Get embeddings from BERT."""
  token_type_ids = None
  if use_segment_ids:
    token_type_ids = features[constants.SEGMENT_ID_KEY]

  max_seq_len = tf.shape(features[constants.SOURCE_WORDPIECES_KEY])[1]
  input_mask = bert_utils.get_input_mask(max_seq_len,
                                         features[constants.SOURCE_LEN_KEY])
  input_ids = features[constants.SOURCE_WORDPIECES_KEY]
  source_embeddings = bert_utils.get_bert_embeddings(
      input_ids,
      bert_config,
      input_mask,
      token_type_ids=token_type_ids,
      is_training=is_training,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope)
  source_embeddings = common_layers.linear_transform(source_embeddings,
                                                     wordpiece_embedding_size,
                                                     "bert_transform")

  # Set weights to ignore padding.
  embedded_weights = tf.to_float(
      tf.not_equal(input_ids, constants.PAD_SYMBOL_ID))
  embedded_weights = tf.expand_dims(embedded_weights, -1)
  return source_embeddings * embedded_weights


def get_input_embeddings(model_config, bert_config, features, is_training,
                         use_one_hot_embeddings):
  """Returns tensor representing inputs for the given batch."""
  with tf.variable_scope("bert") as scope:
    wordpiece_embeddings = _bert_embeddings(
        model_config.model_parameters.source_embedding_dims, bert_config,
        features, is_training, use_one_hot_embeddings, scope,
        model_config.model_parameters.use_segment_ids)

  # Apply extra features, if present.
  if model_config.model_parameters.use_foreign_key_features:
    with tf.variable_scope("foreign_key_embeddings") as scope:
      # Embed them
      initializer = _default_initializer(model_config.model_parameters)
      key_embeddings_dim = model_config.model_parameters.source_embedding_dims
      foreign_key_embeddings = tf.get_variable(
          name="foreign_keys",
          shape=[3, key_embeddings_dim],
          initializer=initializer)

      # Looks up the embedding. Adds 1 first, because 0 indicates padding.
      wordpiece_embeddings = wordpiece_embeddings + _ignore_pad(
          foreign_key_embeddings, features[constants.FOREIGN_KEY_KEY] + 1,
          use_one_hot_embeddings)
  if model_config.model_parameters.use_alignment_features:
    with tf.variable_scope("alignment_features") as scope:
      initializer = _default_initializer(model_config.model_parameters)
      key_embeddings_dim = model_config.model_parameters.source_embedding_dims
      foreign_key_embeddings = tf.get_variable(
          name="alignment_embeddings",
          shape=[3, key_embeddings_dim],
          initializer=initializer)

      features = features[constants.ALIGNED_KEY] + 1

      # Looks up the embedding. Adds 1 first, because 0 indicates padding.
      wordpiece_embeddings = wordpiece_embeddings + _ignore_pad(
          foreign_key_embeddings, features, use_one_hot_embeddings)

  return wordpiece_embeddings


def get_output_vocab_embeddings_table(model_config, output_vocab_filepath):
  """Returns the embedding table used for the target sequence."""
  num_symbols = get_output_vocab_size(output_vocab_filepath)
  with tf.variable_scope("target_embeddings"):
    return tf.get_variable(
        name="symbols",
        shape=[
            num_symbols, model_config.model_parameters.target_embedding_dims
        ],
        initializer=_default_initializer(model_config.model_parameters))


def get_clean_output_mask(output_vocab_filepath, clean_output_vocab_path):
  """Get the masks for generating only clean actions."""
  regular_output_symbols = _get_vocab_symbols(output_vocab_filepath)
  clean_output_symbols = _get_vocab_symbols(clean_output_vocab_path)

  output_mask = []

  for _ in range(constants.NUM_RESERVED_OUTPUT_SYMBOLS):
    output_mask.append(1)

  for symbol in regular_output_symbols:
    if symbol in clean_output_symbols:
      output_mask.append(1)
    else:
      output_mask.append(0)

  tf.logging.info("Using clean output vocab ")
  tf.logging.info("   original vocab size: " + str(len(regular_output_symbols)))
  tf.logging.info("   after cleaning: " + str(sum(output_mask)))
  return output_mask

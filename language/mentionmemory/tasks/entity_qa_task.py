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
"""Contains base implementation for entity-answer question answering tasks."""



import jax.numpy as jnp
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.utils import default_values
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class EntityQATask(mention_encoder_task.MentionEncoderTask):
  """Abstract class for all entity-answer question answering tasks."""

  @staticmethod
  def get_name_to_features(config):
    """Return feature dict for decoding purposes. See BaseTask."""
    name_to_features = (
        mention_encoder_task.MentionEncoderTask.get_name_to_features(config))

    if config.apply_answer_mask:
      name_to_features['dense_answer_mask'] = tf.io.FixedLenFeature(
          config.model_config.encoder_config.max_length, tf.int64)

    return name_to_features

  @staticmethod
  def make_preprocess_fn(
      config
  ):
    """Produces function to preprocess samples.

    See BaseTask.

    During preprocessing we mask entity mentions as well as non-mention tokens.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses samples to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    max_length = config.model_config.encoder_config.max_length
    mask_token_id = getattr(config, 'mask_token_id', default_values.MASK_TOKEN)
    apply_answer_mask = config.apply_answer_mask

    def preprocess_fn(example):
      """Performs preprocessing for individual sample."""

      int_type = example['text_ids'].dtype
      dense_is_masked = tf.equal(example['text_ids'], mask_token_id)
      dense_is_masked = tf.cast(dense_is_masked, dtype=int_type)
      example['dense_is_masked'] = dense_is_masked
      example['position_ids'] = tf.range(max_length, dtype=int_type)
      example['segment_ids'] = tf.zeros(max_length, dtype=int_type)

      if apply_answer_mask:
        example['dense_mention_ids'] = (
            example['dense_mention_ids'] * example['dense_answer_mask'])
      return example

    return preprocess_fn

  @staticmethod
  def dummy_input(config):
    """Produces model-specific dummy input batch. See BaseTask."""

    model_config = config.model_config
    encoder_config = model_config.encoder_config
    bsz = config.per_device_batch_size
    text_shape = (bsz, encoder_config.max_length)
    mention_position_shape = (config.max_mentions * bsz,)
    mention_target_shape = (config.max_mention_targets * bsz,)
    int_type = jnp.int32
    float_type = encoder_config.dtype

    position_ids = np.arange(encoder_config.max_length)
    position_ids = np.tile(position_ids, (bsz, 1))

    mention_global_positions = np.arange(config.max_mentions * bsz)
    mention_batch_positions, mention_start_positions = np.divmod(
        mention_global_positions, encoder_config.max_length)
    mention_end_positions = mention_start_positions

    mention_target_indices = np.arange(config.max_mention_targets * bsz)
    mention_target_batch_positions = mention_batch_positions[
        mention_target_indices]
    mention_target_start_positions = mention_start_positions[
        mention_target_indices]
    mention_target_end_positions = mention_end_positions[mention_target_indices]

    dummy_input = {
        'text_ids':
            jnp.ones(text_shape, int_type),
        'text_mask':
            jnp.ones(text_shape, int_type),
        'position_ids':
            jnp.asarray(position_ids, int_type),
        'segment_ids':
            jnp.zeros(text_shape, int_type),
        'mention_batch_positions':
            jnp.asarray(mention_batch_positions, int_type),
        'mention_start_positions':
            jnp.asarray(mention_start_positions, int_type),
        'mention_end_positions':
            jnp.asarray(mention_end_positions, int_type),
        'mention_mask':
            jnp.ones(mention_position_shape, int_type),
        'mention_is_masked':
            jnp.ones(mention_position_shape, int_type),
        'mention_target_indices':
            jnp.asarray(mention_target_indices, int_type),
        'mention_target_weights':
            jnp.ones(mention_target_shape, float_type),
        'mention_target_ids':
            jnp.ones(mention_target_shape, int_type),
        'mention_target_batch_positions':
            jnp.asarray(mention_target_batch_positions, int_type),
        'mention_target_start_positions':
            jnp.asarray(mention_target_start_positions, int_type),
        'mention_target_end_positions':
            jnp.asarray(mention_target_end_positions, int_type),
    }
    return dummy_input

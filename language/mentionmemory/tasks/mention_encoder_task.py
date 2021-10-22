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
"""Contains task with base methods for pre-training a mention encoder."""



import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import base_task
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class MentionEncoderTask(base_task.BaseTask):
  """Task with base methods for pre-training a mention encoder."""

  encoder_name = 'example_name'

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
    model_config = config.model_config
    encoder_config = model_config.encoder_config
    max_mlm_targets = config.max_mlm_targets
    max_length = encoder_config.max_length
    vocab_size = encoder_config.vocab_size

    mask_rate = config.mask_rate
    mention_mask_rate = config.mention_mask_rate
    mask_token_id = getattr(config, 'mask_token_id', 103)

    def preprocess_fn(example):
      """Performs preprocessing for individual sample."""
      new_example = {}
      new_example['text_mask'] = example['text_mask']
      new_example['dense_span_starts'] = example['dense_span_starts']
      new_example['dense_span_ends'] = example['dense_span_ends']
      new_example['dense_mention_ids'] = example['dense_mention_ids']
      new_example['dense_mention_mask'] = example['dense_mention_mask']

      if max_mlm_targets > 0:
        # Perform masking
        masked_dict = mention_preprocess_utils.mask_mentions_and_tokens_tf(
            text_ids=example['text_ids'],
            text_mask=example['text_mask'],
            dense_span_starts=example['dense_span_starts'],
            dense_span_ends=example['dense_span_ends'],
            non_mention_mask_rate=mask_rate,
            mention_mask_rate=mention_mask_rate,
            max_mlm_targets=max_mlm_targets,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            random_replacement_prob=0.1,
            identity_replacement_prob=0.1,
        )

        new_example['text_ids'] = masked_dict['masked_text_ids']
        new_example['mlm_target_positions'] = masked_dict[
            'mlm_target_positions']
        new_example['mlm_target_ids'] = masked_dict['mlm_target_ids']
        new_example['mlm_target_weights'] = masked_dict['mlm_target_weights']
        new_example['mlm_target_is_mention'] = masked_dict[
            'mlm_target_is_mention']
        new_example['dense_is_masked'] = masked_dict['dense_is_masked']
      else:
        # Don't perform masking if 0 `max_mlm_targets` were requested.
        new_example['text_ids'] = example['text_ids']
        new_example['dense_is_masked'] = np.zeros(max_length, dtype=np.int32)

      new_example['position_ids'] = tf.range(max_length)
      new_example['segment_ids'] = tf.zeros(
          max_length, dtype=example['text_ids'].dtype)
      return new_example

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config
  ):
    """Produces function to preprocess batches.

    See BaseTask.

    In the collater we subsample and flatten mentions across the batch, and
    sample mention targets for mention entity prediction or coreference tasks.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    max_batch_mentions = config.max_mentions * config.per_device_batch_size
    max_batch_mention_targets = (
        config.max_mention_targets * config.per_device_batch_size)

    def collater_fn(batch):
      new_features = mention_preprocess_utils.process_batchwise_mention_targets(
          dense_span_starts=batch['dense_span_starts'],
          dense_span_ends=batch['dense_span_ends'],
          dense_mention_ids=batch['dense_mention_ids'],
          dense_linked_mention_mask=batch['dense_mention_mask'],
          dense_is_masked=batch.get('dense_is_masked'),
          max_mentions=max_batch_mentions,
          max_mention_targets=max_batch_mention_targets,
      )
      new_batch = {key: value for key, value in batch.items()}
      new_batch.update(new_features)

      if config.get('max_length_with_entity_tokens') is not None:
        batch_with_entity_tokens = mention_preprocess_utils.add_entity_tokens(
            text_ids=new_batch['text_ids'],
            text_mask=new_batch['text_mask'],
            mention_mask=new_batch['mention_mask'],
            mention_batch_positions=new_batch['mention_batch_positions'],
            mention_start_positions=new_batch['mention_start_positions'],
            mention_end_positions=new_batch['mention_end_positions'],
            mlm_target_positions=new_batch.get('mlm_target_positions'),
            mlm_target_weights=new_batch.get('mlm_target_weights'),
            new_length=config.max_length_with_entity_tokens,
        )
        # Update `text_ids`, `text_mask`, `mention_mask`, `mention_*_positions`
        new_batch.update(batch_with_entity_tokens)

        # Need to update mention target features because some of the positions
        # and `mention_mask` have been updated.
        mention_target_features = mention_preprocess_utils.prepare_mention_target_features(
            new_batch['mention_batch_positions'],
            new_batch['mention_start_positions'],
            new_batch['mention_end_positions'], new_batch['mention_mask'],
            new_batch['mention_target_weights'],
            new_batch['mention_target_indices'])
        new_batch.update(mention_target_features)

        # Need to update `position_ids` and `segment_ids` because the max_length
        # has changed.
        new_batch['position_ids'] = tf.tile(
            tf.expand_dims(tf.range(config.max_length_with_entity_tokens), 0),
            [config.per_device_batch_size, 1])
        new_batch['position_ids'] = tf.cast(
            new_batch['position_ids'], dtype=new_batch['text_ids'].dtype)
        new_batch['segment_ids'] = tf.zeros(
            (config.per_device_batch_size,
             config.max_length_with_entity_tokens),
            dtype=new_batch['text_ids'].dtype)

      return new_batch

    return collater_fn

  @staticmethod
  def get_name_to_features(config):
    """Return feature dict for decoding purposes. See BaseTask."""

    encoder_config = config.model_config.encoder_config
    max_length = encoder_config.max_length
    name_to_features = {
        'text_ids': tf.io.FixedLenFeature(max_length, tf.int64),
        'text_mask': tf.io.FixedLenFeature(max_length, tf.int64),
        'dense_span_starts': tf.io.FixedLenFeature(max_length, tf.int64),
        'dense_span_ends': tf.io.FixedLenFeature(max_length, tf.int64),
        'dense_mention_mask': tf.io.FixedLenFeature(max_length, tf.int64),
        'dense_mention_ids': tf.io.FixedLenFeature(max_length, tf.int64),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config):
    """Produces model-specific dummy input batch. See BaseTask."""

    model_config = config.model_config
    encoder_config = model_config.encoder_config
    bsz = config.per_device_batch_size
    text_shape = (bsz, encoder_config.max_length)
    mlm_target_shape = (bsz, config.max_mlm_targets)
    mention_position_shape = (config.max_mentions * bsz,)
    mention_target_shape = (config.max_mention_targets * bsz,)
    int_type = jnp.int32
    float_type = encoder_config.dtype

    position_ids = np.arange(encoder_config.max_length)
    position_ids = np.tile(position_ids, (bsz, 1))

    mlm_target_positions = np.arange(config.max_mlm_targets)
    mlm_target_positions = np.tile(mlm_target_positions, (bsz, 1))

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
        'mlm_target_positions':
            jnp.asarray(mlm_target_positions, int_type),
        'mlm_target_ids':
            jnp.ones(mlm_target_shape, int_type),
        'mlm_target_weights':
            jnp.ones(mlm_target_shape, float_type),
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

  @classmethod
  def load_weights(cls, config):
    """Load model weights from file.

    We assume that MentionEncoderTasks specify an encoder name as a class
    attribute, which we use to load encoder weights.

    Args:
      config: experiment config.

    Returns:
      Dictionary of model weights.
    """

    encoder_class = encoder_registry.get_registered_encoder(cls.encoder_name)
    encoder_variables = encoder_class.load_weights(config)
    model_variables = {}
    for group_key in encoder_variables:
      model_variables[group_key] = {'encoder': encoder_variables[group_key]}
    return model_variables

  @classmethod
  def make_output_postprocess_fn(
      cls,
      config  # pylint: disable=unused-argument
  ):
    """Postprocess task samples (input and output). See BaseTask."""

    base_postprocess_fn = base_task.BaseTask.make_output_postprocess_fn(config)

    encoder_class = encoder_registry.get_registered_encoder(cls.encoder_name)
    encoder_postprocess_fn = encoder_class.make_output_postprocess_fn(config)

    def postprocess_fn(batch,
                       auxiliary_output):
      """Function that prepares model's input and output for serialization."""

      new_auxiliary_output = {}
      new_auxiliary_output.update(auxiliary_output)
      encoder_specific_features = encoder_postprocess_fn(
          batch, new_auxiliary_output)
      new_auxiliary_output.update(encoder_specific_features)
      return base_postprocess_fn(batch, new_auxiliary_output)

    return postprocess_fn

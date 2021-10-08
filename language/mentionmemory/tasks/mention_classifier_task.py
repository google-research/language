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
"""Contains base implementation for all mention-level classification tasks."""
from typing import Any, Callable, Dict, Text

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import downstream_encoder_task
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class MentionClassifierModel(nn.Module):
  """Encoder wrapper with classification layer for mention classification.

  This model takes mention-annotated text as input and performs n-ary
  classification on top of target mention encodings. The model assumes that
  there is only ONE target mention per sample.

  Attributes:
    num_classes: Number of classification labels.
    encoder_name: Name of encoder model to use to encode passage.
    encoder_config: Encoder hyperparameters.
    dtype: Precision of computation.
  """

  num_classes: int
  encoder_name: str
  encoder_config: ml_collections.FrozenConfigDict
  dtype: Dtype
  mention_encodings_feature: str = 'target_mention_encodings'

  def setup(self):
    self.encoder = encoder_registry.get_registered_encoder(
        self.encoder_name)(**self.encoder_config)
    self.linear_classifier = nn.Dense(self.num_classes, dtype=self.dtype)

  def __call__(self, batch: Dict[str, Array], deterministic: bool):
    _, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)
    mention_encodings = loss_helpers[self.mention_encodings_feature]

    classifier_logits = self.linear_classifier(mention_encodings)
    loss_helpers['classifier_logits'] = classifier_logits

    return loss_helpers, logging_helpers


class MentionClassifierTask(downstream_encoder_task.DownstreamEncoderTask):
  """Abstract class for all mention-level classification tasks."""
  model_class = MentionClassifierModel

  @staticmethod
  def make_collater_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[Text, tf.Tensor]], Dict[Text, tf.Tensor]]:
    """Produces function to preprocess batches for mention classification task.

    This function samples and flattens mentions from input data.

    Args:
      config: task configuration.

    Returns:
      collater function.
    """
    max_mentions_per_sample = config.max_mentions_per_sample
    encoder_config = config.model_config.encoder_config
    bsz = config.per_device_batch_size
    max_batch_mentions = config.max_mentions * bsz
    n_candidate_mentions = max_mentions_per_sample * bsz

    def collater_fn(batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
      """Collater function for mention classification task. See BaseTask."""

      new_batch = {}

      # Sample mentions uniformly across batch
      mention_mask = tf.reshape(batch['mention_mask'], [n_candidate_mentions])
      sample_scores = tf.random.uniform(shape=[n_candidate_mentions]) * tf.cast(
          mention_mask, tf.float32)

      mention_target_indices = tf.reshape(batch['mention_target_indices'],
                                          [bsz])

      # We want to make sure that the target mentions always have a priority
      # when we sample `max_batch_mentions` out of all available mentions.
      # Additionally, we want these target mentions to be in the same order as
      # their samples. In other words, we want the first sampled mention to be
      # target mention from the first sample, the second sampled mention to be
      # tagret mention from the second sample, etc.

      # Positions of target mentions in the flat array
      mention_target_indices_flat = (
          tf.cast(
              tf.range(bsz) * max_mentions_per_sample,
              mention_target_indices.dtype) + mention_target_indices)
      # These extra score makes sure that target mentions have a priority and
      # will be sampled in the correct order.
      mention_target_extra_score_flat = tf.cast(
          tf.reverse(tf.range(bsz) + 1, axis=[0]), tf.float32)
      # The model assumes that there is only ONE target mention per sample.
      # Moreover,we want to select them according to the order of samples:
      # target mention from sample 0, target mention from sample 1, ..., etc.
      sample_scores = tf.tensor_scatter_nd_add(
          sample_scores, tf.expand_dims(mention_target_indices_flat, 1),
          mention_target_extra_score_flat)

      sampled_indices = tf.math.top_k(
          sample_scores, max_batch_mentions, sorted=True).indices

      # Double-check target mentions were selected correctly.
      assert_op = tf.assert_equal(
          sampled_indices[:bsz],
          tf.cast(mention_target_indices_flat, sampled_indices.dtype))

      with tf.control_dependencies([assert_op]):
        mention_mask = tf.gather(mention_mask, sampled_indices)
      dtype = batch['mention_start_positions'].dtype
      mention_start_positions = tf.gather(
          tf.reshape(batch['mention_start_positions'], [n_candidate_mentions]),
          sampled_indices)
      mention_end_positions = tf.gather(
          tf.reshape(batch['mention_end_positions'], [n_candidate_mentions]),
          sampled_indices)

      mention_batch_positions = tf.gather(
          tf.repeat(tf.range(bsz, dtype=dtype), max_mentions_per_sample),
          sampled_indices)

      new_batch['text_ids'] = batch['text_ids']
      new_batch['text_mask'] = batch['text_mask']
      new_batch['classifier_target'] = tf.reshape(
          batch['target'], [bsz, config.max_num_labels_per_sample])
      new_batch['classifier_target_mask'] = tf.reshape(
          batch['target_mask'], [bsz, config.max_num_labels_per_sample])

      new_batch['mention_mask'] = mention_mask
      new_batch['mention_start_positions'] = mention_start_positions
      new_batch['mention_end_positions'] = mention_end_positions
      new_batch['mention_batch_positions'] = mention_batch_positions
      new_batch['mention_target_indices'] = tf.range(bsz, dtype=dtype)

      if config.get('max_length_with_entity_tokens') is not None:
        batch_with_entity_tokens = mention_preprocess_utils.add_entity_tokens(
            text_ids=new_batch['text_ids'],
            text_mask=new_batch['text_mask'],
            mention_mask=new_batch['mention_mask'],
            mention_batch_positions=new_batch['mention_batch_positions'],
            mention_start_positions=new_batch['mention_start_positions'],
            mention_end_positions=new_batch['mention_end_positions'],
            new_length=config.max_length_with_entity_tokens,
        )
        # Update `text_ids`, `text_mask`, `mention_mask`, `mention_*_positions`
        new_batch.update(batch_with_entity_tokens)
        # Update `max_length`
        max_length = config.max_length_with_entity_tokens
      else:
        max_length = encoder_config.max_length

      new_batch['mention_target_batch_positions'] = tf.gather(
          new_batch['mention_batch_positions'],
          new_batch['mention_target_indices'])
      new_batch['mention_target_start_positions'] = tf.gather(
          new_batch['mention_start_positions'],
          new_batch['mention_target_indices'])
      new_batch['mention_target_end_positions'] = tf.gather(
          new_batch['mention_end_positions'],
          new_batch['mention_target_indices'])
      new_batch['mention_target_weights'] = tf.ones(bsz)

      # Fake IDs -- some encoders (ReadTwice) need them
      new_batch['mention_target_ids'] = tf.zeros(bsz)

      new_batch['segment_ids'] = tf.zeros_like(new_batch['text_ids'])

      position_ids = tf.expand_dims(tf.range(max_length, dtype=dtype), axis=0)
      new_batch['position_ids'] = tf.tile(position_ids, (bsz, 1))

      return new_batch

    return collater_fn

  @staticmethod
  def get_name_to_features(
      config: ml_collections.ConfigDict) -> Dict[Text, Any]:
    """Return feature dict for decoding purposes. See BaseTask for details."""
    encoder_config = config.model_config.encoder_config
    max_length = encoder_config.max_length

    name_to_features = {
        'text_ids':
            tf.io.FixedLenFeature(max_length, tf.int64),
        'text_mask':
            tf.io.FixedLenFeature(max_length, tf.int64),
        'target':
            tf.io.FixedLenFeature(config.max_num_labels_per_sample, tf.int64),
        'target_mask':
            tf.io.FixedLenFeature(config.max_num_labels_per_sample, tf.int64),
        'mention_start_positions':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'mention_end_positions':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'mention_mask':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'mention_target_indices':
            tf.io.FixedLenFeature(1, tf.int64),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[Text, Any]:
    """Produces model-specific dummy input batch. See BaseTask for details."""

    encoder_config = config.model_config.encoder_config
    bsz = config.per_device_batch_size
    if config.get('max_length_with_entity_tokens') is not None:
      max_length = config.max_length_with_entity_tokens
    else:
      max_length = encoder_config.max_length

    text_shape = (bsz, max_length)
    mention_shape = (config.max_mentions)
    int_type = jnp.int32

    position_ids = np.arange(max_length)
    position_ids = np.tile(position_ids, (bsz, 1))

    dummy_input = {
        'text_ids':
            jnp.ones(text_shape, int_type),
        'text_mask':
            jnp.ones(text_shape, int_type),
        'position_ids':
            jnp.asarray(position_ids, int_type),
        'segment_ids':
            jnp.zeros(text_shape, int_type),
        'classifier_target':
            jnp.ones((bsz, config.max_num_labels_per_sample), int_type),
        'classifier_target_mask':
            jnp.ones((bsz, config.max_num_labels_per_sample), int_type),
        'mention_start_positions':
            jnp.zeros(mention_shape, int_type),
        'mention_end_positions':
            jnp.zeros(mention_shape, int_type),
        'mention_mask':
            jnp.ones(mention_shape, int_type),
        'mention_batch_positions':
            jnp.ones(mention_shape, int_type),
        'mention_target_indices':
            jnp.arange(bsz, dtype=int_type),
        'mention_target_batch_positions':
            jnp.arange(bsz, dtype=int_type),
        'mention_target_start_positions':
            jnp.zeros(bsz, int_type),
        'mention_target_end_positions':
            jnp.zeros(bsz, int_type),
        'mention_target_weights':
            jnp.zeros(bsz, int_type),
        'mention_target_ids':
            jnp.zeros(bsz, int_type),
    }

    return dummy_input

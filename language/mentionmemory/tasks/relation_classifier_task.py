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
"""Contains base implementation for relation classification tasks."""
from typing import Any, Callable, Dict, Optional, Text, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import mlp
from language.mentionmemory.tasks import downstream_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, Dtype, MetricGroups  # pylint: disable=g-multiple-import
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class RelationClassifierModel(nn.Module):
  """Encoder wrapper with classification layer for relation classification.

  This model takes mention-annotated text with special mentions per sample
  denoted as "subject" and "object". The model generates mention encodings for
  these two mentions, concatenates them and applies (potentially, multi-layer)
  n-ary classification layers.

  Attributes:
    num_classes: number of classification labels.
    num_layers: number of classification MLP layers on top of mention encodings.
    input_dim: input dimensionality of classification MLP layers. This must be
      equal to 2 * mention encodings size.
    hidden_dim: hidden dimensionality of classification MLP layers.
    dropout_rate: dropout rate of classification MLP layers.
    encoder_name: name of encoder model to use to encode passage.
    encoder_config: encoder hyperparameters.
    dtype: precision of computation.
    mention_encodings_feature: feature name for encodings of target mentions.
  """

  num_classes: int
  num_layers: int
  input_dim: int
  hidden_dim: int
  dropout_rate: float
  encoder_name: str
  encoder_config: ml_collections.FrozenConfigDict
  dtype: Dtype

  mention_encodings_feature: str = 'target_mention_encodings'
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

  def setup(self):
    self.encoder = encoder_registry.get_registered_encoder(
        self.encoder_name)(**self.encoder_config)

    self.classification_mlp_layers = [
        mlp.MLPBlock(  # pylint: disable=g-complex-comprehension
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
        ) for _ in range(self.num_layers)
    ]

    self.linear_classifier = nn.Dense(self.num_classes, dtype=self.dtype)

  def __call__(self, batch: Dict[str, Array], deterministic: bool):
    _, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)
    mention_encodings = loss_helpers[self.mention_encodings_feature]

    subject_mention_encodings = jut.matmul_slice(
        mention_encodings, batch['mention_subject_indices'])

    object_mention_encodings = jut.matmul_slice(mention_encodings,
                                                batch['mention_object_indices'])

    relation_encodings = jnp.concatenate(
        [subject_mention_encodings, object_mention_encodings], -1)

    for mlp_layer in self.classification_mlp_layers:
      relation_encodings = mlp_layer(relation_encodings, deterministic)

    classifier_logits = self.linear_classifier(relation_encodings)
    loss_helpers['classifier_logits'] = classifier_logits

    return loss_helpers, logging_helpers


@task_registry.register_task('relation_classifier')
class RelationClassifierTask(downstream_encoder_task.DownstreamEncoderTask):
  """Class for relation classification task."""
  model_class = RelationClassifierModel

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates loss function for Relation Classifier training.

    TODO(urikz): Write detailed description.

    Args:
      config: task configuration.

    Returns:
      Loss function.
    """

    ignore_label = config.ignore_label

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[Text, Any],
        model_vars: Dict[Text, Any],
        batch: Dict[Text, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[Text, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Loss function used by Relation Classifier task. See BaseTask."""

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, _ = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      weights = jnp.ones_like(batch['classifier_target'])

      loss, denom = metric_utils.compute_weighted_cross_entropy(
          loss_helpers['classifier_logits'], batch['classifier_target'],
          weights)

      acc, _ = metric_utils.compute_weighted_accuracy(
          loss_helpers['classifier_logits'], batch['classifier_target'],
          weights)

      predictions = jnp.argmax(loss_helpers['classifier_logits'], axis=-1)

      tp, fp, fn = metric_utils.compute_tp_fp_fn_weighted(
          predictions, batch['classifier_target'], weights, ignore_label)

      metrics = {
          'agg': {
              'loss': loss,
              'denominator': denom,
              'acc': acc,
          },
          'micro_precision': {
              'value': tp,
              'denominator': tp + fp,
          },
          'micro_recall': {
              'value': tp,
              'denominator': tp + fn,
          }
      }

      auxiliary_output = {'predictions': predictions}
      auxiliary_output.update(cls.get_auxiliary_output(loss_helpers))

      return loss, metrics, auxiliary_output

    return loss_fn

  @staticmethod
  def make_collater_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[Text, tf.Tensor]], Dict[Text, tf.Tensor]]:
    """Produces function to preprocess batches for relation classification task.

    This function samples and flattens mentions from input data.

    Args:
      config: task configuration.

    Returns:
      collater function.
    """
    encoder_config = config.model_config.encoder_config
    bsz = config.per_device_batch_size
    max_batch_mentions = config.max_mentions * bsz
    n_candidate_mentions = config.max_mentions_per_sample * bsz

    if config.max_mentions < 2:
      raise ValueError('Need at least two mentions per sample in order to '
                       'include object and subject mentions.')

    def collater_fn(batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
      """Collater function for relation classification task. See BaseTask."""

      def flatten_bsz(tensor):
        return tf.reshape(tensor, [bsz])

      new_batch = {
          'text_ids': batch['text_ids'],
          'text_mask': batch['text_mask'],
          'classifier_target': flatten_bsz(batch['target']),
      }

      # Sample mentions across batch

      # We want to make sure that the subject / object mentions always have
      # priority when we sample `max_batch_mentions` out of all available
      # mentions. Additionally, we want these subject / object  mentions to be
      # in the same order as their samples. In other words, we want the first
      # sampled mention to be object mention from the first sample, the second
      # sampled mention to be subject mention from the first sample, the third
      # sampled mention to be object mention from the second sample, etc.

      subj_index = flatten_bsz(batch['subject_mention_indices'])
      obj_index = flatten_bsz(batch['object_mention_indices'])

      # Adjust subject / object mention positions in individual samples to their
      # positions in flattened mentions.
      shift = tf.range(
          bsz, dtype=obj_index.dtype) * config.max_mentions_per_sample
      mention_target_indices = tf.reshape(
          tf.stack([subj_index + shift, obj_index + shift], axis=1), [-1])

      # Sample the rest of the mentions uniformly across batch
      scores = tf.random.uniform(shape=tf.shape(batch['mention_mask']))
      scores = scores * tf.cast(batch['mention_mask'], tf.float32)

      # We want to adjust scores for target mentions so they don't get sampled
      # for the second time. We achive this by making their scores negative.
      def set_negative_scores(scores, indices):
        indices_2d = tf.stack([tf.range(bsz, dtype=indices.dtype), indices],
                              axis=1)
        return tf.tensor_scatter_nd_update(scores, indices_2d,
                                           tf.fill(tf.shape(indices), -1.0))

      # Note that since we're using 2D scores (not yet flattened for simplicity)
      # we use unadjusted `subj_index` and `obj_index`.
      scores = set_negative_scores(scores, subj_index)
      scores = set_negative_scores(scores, obj_index)

      # There are `2 * bsz` target mentions which were already chosen
      num_to_sample = tf.maximum(max_batch_mentions - 2 * bsz, 0)
      sampled_scores, sampled_indices = tf.math.top_k(
          tf.reshape(scores, [-1]), num_to_sample, sorted=True)

      # Note that negative scores indicate that we have double-sampled some of
      # the target mentions (we set their scores to negative right above).
      # In this case, we remove them.
      num_not_double_sampled = tf.reduce_sum(
          tf.cast(tf.not_equal(sampled_scores, -1), tf.int32))
      sampled_indices = sampled_indices[:num_not_double_sampled]

      # Combine target mentions (subject / object) with sampled mentions
      mention_target_indices = tf.cast(mention_target_indices,
                                       sampled_indices.dtype)
      sampled_indices = tf.concat([mention_target_indices, sampled_indices],
                                  axis=0)

      sampled_indices = mention_preprocess_utils.dynamic_padding_1d(
          sampled_indices, max_batch_mentions)

      dtype = batch['mention_start_positions'].dtype
      mention_mask = tf.reshape(batch['mention_mask'], [n_candidate_mentions])
      new_batch['mention_mask'] = tf.gather(mention_mask, sampled_indices)
      new_batch['mention_start_positions'] = tf.gather(
          tf.reshape(batch['mention_start_positions'], [n_candidate_mentions]),
          sampled_indices)
      new_batch['mention_end_positions'] = tf.gather(
          tf.reshape(batch['mention_end_positions'], [n_candidate_mentions]),
          sampled_indices)
      new_batch['mention_batch_positions'] = tf.gather(
          tf.repeat(tf.range(bsz, dtype=dtype), config.max_mentions_per_sample),
          sampled_indices)

      new_batch['mention_target_indices'] = tf.range(2 * bsz, dtype=dtype)
      new_batch['mention_subject_indices'] = tf.range(bsz, dtype=dtype) * 2
      new_batch['mention_object_indices'] = tf.range(bsz, dtype=dtype) * 2 + 1

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
      new_batch['mention_target_weights'] = tf.ones(2 * bsz)

      # Fake IDs -- some encoders (ReadTwice) need them
      new_batch['mention_target_ids'] = tf.zeros(2 * bsz)

      new_batch['segment_ids'] = tf.zeros_like(new_batch['text_ids'])

      position_ids = tf.expand_dims(tf.range(max_length), axis=0)
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
            tf.io.FixedLenFeature(1, tf.int64),
        'mention_start_positions':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'mention_end_positions':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'mention_mask':
            tf.io.FixedLenFeature(config.max_mentions_per_sample, tf.int64),
        'subject_mention_indices':
            tf.io.FixedLenFeature(1, tf.int64),
        'object_mention_indices':
            tf.io.FixedLenFeature(1, tf.int64),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[Text, Any]:
    """Produces model-specific dummy input batch. See BaseTask for details."""

    if config.get('max_length_with_entity_tokens') is not None:
      max_length = config.max_length_with_entity_tokens
    else:
      max_length = config.model_config.encoder_config.max_length

    bsz = config.per_device_batch_size
    text_shape = (bsz, max_length)
    mention_shape = (config.max_mentions)
    mention_target_shape = (2 * bsz)
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
            jnp.ones((bsz,), int_type),
        'mention_start_positions':
            jnp.zeros(mention_shape, int_type),
        'mention_end_positions':
            jnp.zeros(mention_shape, int_type),
        'mention_mask':
            jnp.ones(mention_shape, int_type),
        'mention_batch_positions':
            jnp.ones(mention_shape, int_type),
        'mention_target_indices':
            jnp.arange(mention_target_shape, dtype=int_type),
        'mention_target_weights':
            jnp.ones(mention_target_shape, dtype=int_type),
        'mention_object_indices':
            jnp.arange(bsz, dtype=int_type),
        'mention_subject_indices':
            jnp.arange(bsz, dtype=int_type),
        'mention_target_batch_positions':
            jnp.arange(mention_target_shape, dtype=int_type),
        'mention_target_start_positions':
            jnp.zeros(mention_target_shape, int_type),
        'mention_target_end_positions':
            jnp.zeros(mention_target_shape, int_type),
        'mention_target_ids':
            jnp.zeros(mention_target_shape, int_type),
    }

    return dummy_input

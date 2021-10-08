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
"""Simple model for testing purposes."""
from typing import Any, Callable, Dict, Optional, Text, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import downstream_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, Dtype, MetricGroups  # pylint: disable=g-multiple-import
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class TextClassifierModel(nn.Module):
  """Encoder wrapper with classification layer to perform text classification.

  This model takes mention-annotated text as input and performs n-ary
  classification. First, a user-specified encoder processes the passage to
  generate a sequence of token representations. The model treats the first token
  representation as a passage representation. The passage representation is
  passed through a linear projection with output size equal to the
  classification vocabulary size to produce a score for each class. The loss is
  given by softmax cross-entropy applied to the scores.

  Attributes:
    vocab_size: Number of classification labels.
    encoder_name: Name of encoder model to use to encode passage.
    encoder_config: Encoder hyperparameters.
    dtype: Precision of computation.
    apply_mlp: If true, apply mlp before linear classifier.
  """

  vocab_size: int
  encoder_name: str
  encoder_config: ml_collections.FrozenConfigDict
  dtype: Dtype
  apply_mlp: bool = False

  def setup(self):
    self.encoder = encoder_registry.get_registered_encoder(
        self.encoder_name)(**self.encoder_config)

    if self.apply_mlp:
      self.mlp = nn.Dense(self.encoder_config.hidden_size, self.dtype)
      self.dropout = nn.Dropout(self.encoder_config.dropout_rate)
    self.linear_classifier = nn.Dense(self.vocab_size, dtype=self.dtype)

  def __call__(self, batch: Dict[str, Array], deterministic: bool):
    encoding, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)
    cls_encoding = encoding[:, 0, ...]

    if self.apply_mlp:
      cls_encoding = self.mlp(cls_encoding)
      cls_encoding = nn.tanh(cls_encoding)
      cls_encoding = self.dropout(cls_encoding, deterministic=deterministic)
    classifier_logits = self.linear_classifier(cls_encoding)
    loss_helpers['classifier_logits'] = classifier_logits

    return loss_helpers, logging_helpers


@task_registry.register_task('text_classifier')
class TextClassifier(downstream_encoder_task.DownstreamEncoderTask):
  """Text classification task."""
  model_class = TextClassifierModel

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates task loss function."""

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[Text, Any],
        model_vars: Dict[Text, Any],
        batch: Dict[Text, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[Text, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Model-specific loss function.

      See BaseTask.

      Classification loss is standard cross-entropy.

      Args:
        model_config: contains model config hyperparameters.
        model_params: contains model parameters.
        model_vars: contains model variables (not optimized).
        batch: model input.
        deterministic: whether dropout etc should be applied.
        dropout_rng: seed for dropout randomness.

      Returns:
        Loss and metrics.
      """
      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, _ = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      if 'sample_weights' in batch:
        weights = batch['sample_weights']
      else:
        weights = jnp.ones_like(batch['classifier_target'])

      loss, denom = metric_utils.compute_weighted_cross_entropy(
          loss_helpers['classifier_logits'], batch['classifier_target'],
          weights)

      acc, _ = metric_utils.compute_weighted_accuracy(
          loss_helpers['classifier_logits'], batch['classifier_target'],
          weights)

      metrics = {
          'agg': {
              'loss': loss,
              'denominator': denom,
              'acc': acc,
          }
      }

      predictions = jnp.argmax(loss_helpers['classifier_logits'], axis=-1)

      auxiliary_output = {'predictions': predictions}
      auxiliary_output.update(cls.get_auxiliary_output(loss_helpers))

      return loss, metrics, auxiliary_output

    return loss_fn

  @staticmethod
  def make_preprocess_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[Text, tf.Tensor]], Dict[Text, tf.Tensor]]:
    """Produces function to preprocess samples. See BaseTask."""

    def preprocess_fn(example: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
      return example

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[Text, tf.Tensor]], Dict[Text, tf.Tensor]]:
    """Produces function to preprocess batches.

    See BaseTask.

    Our encoders take a flat tensor of mentions as input for the whole batch of
    passages to save memory. Here we aggregate mentions from different examples,
    and subsample if there are too many total mentions. In particular, we sample
    a random score for each mention, zeroing out padded mentions, then select
    the k highest-scoring. Similarly, we flatten mention features (passage start
    and end positions and padding mask), keeping track of original passage id
    through batch position feature.

    The encoders process the mention features in flattened form. Mention batch
    positions and start/end positions are then used to incorporate the results
    into the unflattened token representation through a scatter-type operation,
    taking the batch positions as the index into the first axis and the
    start/end positions as index into the second axis.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    max_sample_mentions = config.max_sample_mentions
    encoder_config = config.model_config.encoder_config
    bsz = config.per_device_batch_size
    max_batch_mentions = config.max_mentions * bsz
    n_candidate_mentions = max_sample_mentions * bsz

    def collater_fn(batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:

      new_batch = {}

      # Sample mentions uniformly across batch
      mention_mask = tf.reshape(batch['mention_mask'], [n_candidate_mentions])
      sample_scores = tf.random.uniform(shape=[n_candidate_mentions]) * tf.cast(
          mention_mask, tf.float32)
      sampled_indices = tf.math.top_k(
          sample_scores, max_batch_mentions, sorted=False).indices

      dtype = batch['mention_start_positions'].dtype
      mention_mask = tf.gather(mention_mask, sampled_indices)
      mention_start_positions = tf.gather(
          tf.reshape(batch['mention_start_positions'], [n_candidate_mentions]),
          sampled_indices)
      mention_end_positions = tf.gather(
          tf.reshape(batch['mention_end_positions'], [n_candidate_mentions]),
          sampled_indices)

      mention_batch_positions = tf.gather(
          tf.repeat(tf.range(bsz, dtype=dtype), max_sample_mentions),
          sampled_indices)

      new_batch['text_ids'] = batch['text_ids']
      new_batch['text_mask'] = batch['text_mask']
      new_batch['classifier_target'] = tf.reshape(batch['target'], [bsz])

      new_batch['mention_mask'] = mention_mask
      new_batch['mention_start_positions'] = mention_start_positions
      new_batch['mention_end_positions'] = mention_end_positions
      new_batch['mention_batch_positions'] = mention_batch_positions

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

      new_batch['segment_ids'] = tf.zeros_like(new_batch['text_ids'])

      position_ids = tf.expand_dims(tf.range(max_length), axis=0)
      new_batch['position_ids'] = tf.tile(position_ids, (bsz, 1))

      return new_batch

    return collater_fn

  @staticmethod
  def get_name_to_features(
      config: ml_collections.ConfigDict) -> Dict[Text, Any]:
    """Return feature dict for decoding purposes. See BaseTask."""

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
            tf.io.FixedLenFeature(config.max_sample_mentions, tf.int64),
        'mention_end_positions':
            tf.io.FixedLenFeature(config.max_sample_mentions, tf.int64),
        'mention_mask':
            tf.io.FixedLenFeature(config.max_sample_mentions, tf.int64),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[Text, Any]:
    """Produces model-specific dummy input batch. See BaseTask."""

    if config.get('max_length_with_entity_tokens') is not None:
      max_length = config.max_length_with_entity_tokens
    else:
      max_length = config.model_config.encoder_config.max_length

    bsz = config.per_device_batch_size
    text_shape = (bsz, max_length)
    mention_shape = (config.max_mentions)
    int_type = jnp.int32

    position_ids = np.arange(max_length)
    position_ids = np.tile(position_ids, (bsz, 1))

    dummy_input = {
        'text_ids': jnp.ones(text_shape, int_type),
        'text_mask': jnp.ones(text_shape, int_type),
        'position_ids': jnp.asarray(position_ids, int_type),
        'segment_ids': jnp.zeros(text_shape, int_type),
        'classifier_target': jnp.ones(bsz, int_type),
        'mention_start_positions': jnp.zeros(mention_shape, int_type),
        'mention_end_positions': jnp.zeros(mention_shape, int_type),
        'mention_mask': jnp.ones(mention_shape, int_type),
        'mention_batch_positions': jnp.ones(mention_shape, int_type),
    }

    return dummy_input

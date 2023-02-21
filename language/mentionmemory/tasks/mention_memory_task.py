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
"""Contains mention memory model implementation."""

from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import mention_memory_encoder
from language.mentionmemory.modules import mention_losses
from language.mentionmemory.modules import mlm_layer
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import tensorflow.compat.v2 as tf


class MentionMemoryModel(nn.Module):
  """Mention Memory pre-training model.

  Attributes:
    encoder_config: Mention Memory encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = mention_memory_encoder.MentionMemoryEncoder(
        **self.encoder_config)
    self.mlm_layer = mlm_layer.MLMLayer(
        vocab_size=self.encoder.vocab_size,
        hidden_size=self.encoder.hidden_size,
        dtype=self.encoder.dtype,
        layer_norm_epsilon=self.encoder.layer_norm_epsilon,
        embedding_init=self.encoder.kernel_init,
        bias_init=self.encoder.bias_init,
    )

  def __call__(
      self, batch: Dict[str, Array],
      deterministic: bool) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    encoded_input, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    loss_helpers['mlm_logits'] = self.mlm_layer(
        encoded_input=encoded_input,
        mlm_target_positions=batch['mlm_target_positions'],
        shared_embedding=loss_helpers['word_embeddings'])

    return loss_helpers, logging_helpers


@task_registry.register_task('mention_memory')
class MentionMemoryTask(mention_encoder_task.MentionEncoderTask):
  """Pre-training task for mention memory encoder."""

  model_class = MentionMemoryModel
  encoder_name = 'mention_memory'

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates task loss function.

    See BaseTask.

    The Mention Memory encoder is pre-trained with a combination of 1) MLM loss,
    2) same-entity retrieval loss encouraging retrieval of mentions of the same
    entity as the passage mention, 3) entity coreference loss encouraging
    mentions of the same entity to have similar representations, and 4) Matching
    the Blanks-style loss encouraging mentions of the same entity which co-occur
    with mentions of the same second entity to have similar representations.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """

    mlm_weight = config.mlm_weight
    el_im_weight = config.el_im_weight
    el_second_im_weight = config.get('el_second_im_weight', 0)
    coref_res_weight = config.get('coref_res_weight', 0)
    coref_res_mode = config.get('coref_res_mode', 'dot')
    mtb_im_weight = config.get('mtb_im_weight', 0)
    mtb_final_weight = config.get('mtb_final_weight', 0)
    mtb_score_mode = config.get('mtb_score_mode', 'dot')
    same_passage_weight = config.get('same_passage_weight', 0)
    same_entity_set_retrieval_weight = config.get(
        'same_entity_set_retrieval_weight', 0)
    el_final_weight = config.get('el_final_weight', 0)

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[str, Any],
        model_vars: Dict[str, Any],
        batch: Dict[str, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[str, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Model-specific loss function. See BaseTask."""

      batch_size = batch['text_ids'].shape[0]
      mention_target_ids = batch['mention_target_ids']
      mention_target_ids *= batch['mention_target_weights']

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, logging_helpers = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      mlm_logits = loss_helpers['mlm_logits']
      mlm_target_is_mention = batch['mlm_target_is_mention']
      mlm_target_is_not_mention = 1 - batch['mlm_target_is_mention']
      mention_target_is_masked = batch['mention_target_is_masked']
      mention_target_is_not_masked = 1 - mention_target_is_masked
      mlm_loss, mlm_denom = metric_utils.compute_weighted_cross_entropy(
          mlm_logits, batch['mlm_target_ids'], batch['mlm_target_weights'])
      correct_mask = jnp.equal(
          jnp.argmax(mlm_logits, axis=-1),
          batch['mlm_target_ids']) * batch['mlm_target_weights']
      mlm_acc = correct_mask.sum()
      mlm_mention_acc = (correct_mask * mlm_target_is_mention).sum()
      mlm_mention_denom = (batch['mlm_target_weights'] *
                           mlm_target_is_mention).sum()
      mlm_non_mention_acc = (correct_mask * mlm_target_is_not_mention).sum()
      mlm_non_mention_denom = (batch['mlm_target_weights'] *
                               mlm_target_is_not_mention).sum()
      loss = mlm_weight * mlm_loss / mlm_denom

      metrics = {
          'mlm': {
              'loss': mlm_loss,
              'acc': mlm_acc,
              'denominator': mlm_denom,
          },
          'mlm_mention': {
              'acc': mlm_mention_acc,
              'denominator': mlm_mention_denom,
          },
          'mlm_non_mention': {
              'acc': mlm_non_mention_acc,
              'denominator': mlm_non_mention_denom,
          },
      }

      def process_el_im_loss(loss, weight, prefix=''):
        memory_attention_weights = loss_helpers[prefix +
                                                'memory_attention_weights']
        memory_entity_ids = loss_helpers[prefix + 'top_entity_ids']

        target_mentions_memory_attention_weights = jut.matmul_slice(
            memory_attention_weights, batch['mention_target_indices'])

        intermediate_entity_ids = jut.matmul_slice(
            memory_entity_ids, batch['mention_target_indices'])

        el_loss_intermediate, same_entity_avg_prob, el_im_denom = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
            target_mentions_memory_attention_weights, intermediate_entity_ids,
            mention_target_ids, batch['mention_target_weights'])

        if weight > 0:
          loss += weight * el_loss_intermediate / el_im_denom
        metrics[prefix + 'el_intermediate'] = {
            'loss': el_loss_intermediate,
            'same_entity_avg_prob': same_entity_avg_prob,
            'denominator': el_im_denom,
        }
        return loss

      loss = process_el_im_loss(loss, el_im_weight)
      if 'second_memory_attention_weights' in loss_helpers:
        loss = process_el_im_loss(loss, el_second_im_weight, 'second_')

      if coref_res_weight > 0:
        (coref_res_loss,
         coref_res_metrics) = mention_losses.coreference_resolution_loss(
             loss_helpers['target_mention_encodings'],
             batch['mention_target_batch_positions'], mention_target_ids,
             batch_size, coref_res_mode, mention_target_is_masked)
        coref_res_denom = coref_res_metrics['coref_resolution']['denominator']
        loss += coref_res_weight * coref_res_loss / coref_res_denom
        metrics.update(coref_res_metrics)

      if mtb_im_weight > 0:
        (mtb_im_loss, mtb_im_metrics) = mention_losses.mtb_loss(
            loss_helpers['intermediate_target_mention_encodings'],
            batch['mention_target_batch_positions'], mention_target_ids,
            batch_size, mtb_score_mode, mention_target_is_masked, 'im_')
        mtb_im_denom = mtb_im_metrics['im_mtb']['denominator']
        loss += mtb_im_weight * mtb_im_loss / mtb_im_denom
        metrics.update(mtb_im_metrics)

      if mtb_final_weight > 0:
        (mtb_final_loss, mtb_final_metrics) = mention_losses.mtb_loss(
            loss_helpers['target_mention_encodings'],
            batch['mention_target_batch_positions'], mention_target_ids,
            batch_size, mtb_score_mode, mention_target_is_masked, 'final_')
        mtb_final_denom = mtb_final_metrics['final_mtb']['denominator']
        loss += mtb_final_weight * mtb_final_loss / mtb_final_denom
        metrics.update(mtb_final_metrics)

      if same_passage_weight > 0:
        same_passage_mask = loss_helpers['memory_attention_disallowed_mask']
        (same_passage_loss, same_passage_metrics, _
        ) = metric_utils.compute_cross_entropy_loss_with_positives_and_negatives_masks(
            loss_helpers['memory_attention_scores_with_disallowed'],
            same_passage_mask, jnp.logical_not(same_passage_mask),
            batch['mention_mask'])
        same_passage_denom = same_passage_metrics['denominator']
        loss += same_passage_weight * same_passage_loss / same_passage_denom
        metrics['same_passage'] = same_passage_metrics

      if same_entity_set_retrieval_weight > 0:
        if config.get('same_entity_set_target_threshold') is None:
          raise ValueError(
              '`same_entitites_retrieval_threshold` must be specified '
              'if `same_entity_set_retrieval_weight` is provided')

        (same_entity_set_retrieval_loss, same_entity_set_retrieval_avg_prob,
         same_entity_set_retrieval_denom
        ) = mention_losses.same_entity_set_retrieval_loss(
            mention_target_batch_positions=batch[
                'mention_target_batch_positions'],
            mention_target_ids=mention_target_ids,
            mention_target_weights=batch['mention_target_weights'],
            mention_batch_positions=batch['mention_batch_positions'],
            mention_mask=batch['mention_mask'],
            memory_text_entities=loss_helpers['memory_top_text_entities'],
            memory_attention_weights=loss_helpers['memory_attention_weights'],
            memory_mask=1 - loss_helpers['memory_attention_disallowed_mask'],
            batch_size=batch_size,
            same_entity_set_target_threshold=config
            .same_entity_set_target_threshold)

        loss += (
            same_entity_set_retrieval_weight * same_entity_set_retrieval_loss /
            same_entity_set_retrieval_denom)

        metrics['same_entity_set_retrieval'] = {
            'loss': same_entity_set_retrieval_loss,
            'avg_prob': same_entity_set_retrieval_avg_prob,
            'denominator': same_entity_set_retrieval_denom,
        }

      if el_final_weight > 0:
        final_attention_weights = loss_helpers['final_memory_attention_weights']
        final_memory_entity_ids = loss_helpers['final_top_entity_ids']

        (el_loss_final, same_entity_avg_prob_final, el_loss_denom
        ) = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
            final_attention_weights, final_memory_entity_ids,
            mention_target_ids, batch['mention_target_weights'])

        (_, same_entity_avg_prob_final_masked, el_loss_denom_masked
        ) = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
            final_attention_weights, final_memory_entity_ids,
            mention_target_ids,
            batch['mention_target_weights'] * mention_target_is_masked)

        (_, same_entity_avg_prob_final_not_masked, el_loss_denom_not_masked
        ) = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
            final_attention_weights, final_memory_entity_ids,
            mention_target_ids,
            batch['mention_target_weights'] * mention_target_is_not_masked)

        metrics['el_final'] = {
            'loss': el_loss_final,
            'same_entity_avg_prob': same_entity_avg_prob_final,
            'denominator': el_loss_denom,
        }
        metrics['el_final_masked'] = {
            'same_entity_avg_prob': same_entity_avg_prob_final_masked,
            'denominator': el_loss_denom_masked,
        }
        metrics['el_final_not_masked'] = {
            'same_entity_avg_prob': same_entity_avg_prob_final_not_masked,
            'denominator': el_loss_denom_not_masked,
        }
        loss += el_final_weight * el_loss_final / (
            el_loss_denom + default_values.SMALL_NUMBER)

      metrics['agg'] = {
          'loss': loss,
          'denominator': 1.0,
      }

      if 'n_disallowed' in logging_helpers:
        metrics['disallowed'] = {
            'per_mention': logging_helpers['n_disallowed'],
            'denominator': batch['mention_mask'].sum(),
        }

      if 'second_n_disallowed' in logging_helpers:
        metrics['second_n_disallowed'] = {
            'per_mention': logging_helpers['second_n_disallowed'],
            'denominator': batch['mention_mask'].sum(),
        }

      auxiliary_output = {
          'top_entity_ids': loss_helpers['top_entity_ids'],
          'top_memory_ids': loss_helpers['top_memory_ids'],
      }

      if 'second_top_entity_ids' in loss_helpers:
        auxiliary_output['second_top_entity_ids'] = loss_helpers[
            'second_top_entity_ids']
        auxiliary_output['second_top_memory_ids'] = loss_helpers[
            'second_top_memory_ids']

      return loss, metrics, auxiliary_output  # pytype: disable=bad-return-type  # jax-ndarray

    return loss_fn

  @staticmethod
  def make_preprocess_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    """Produces function to preprocess samples.

    See BaseTask.

    Here we add a text identifier hash to the standard MentionEncoderTask
    preprocessing pipeline.

    Args:
      config: ConfigDict. Contains experiment hyperparameters.

    Returns:
      Function that preprocesses samples to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    max_length = config.model_config.encoder_config.max_length

    mention_preprocessing_fn = mention_encoder_task.MentionEncoderTask.make_preprocess_fn(config)  # pylint: disable=line-too-long

    def preprocess_fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Performs preprocessing for individual sample."""
      new_example = mention_preprocessing_fn(example)

      # Compute hash of text for text identifiers
      new_example['text_identifiers'] = mention_preprocess_utils.text_hash_tf(
          example['text_ids'], max_length)
      return new_example

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    """Produces function to preprocess batches.

    See BaseTask.

    Batches text identifiers after standard mention preprocessing.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    mention_collater_fn = mention_encoder_task.MentionEncoderTask.make_collater_fn(config)  # pylint: disable=line-too-long

    def collater_fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      new_batch = mention_collater_fn(batch)
      new_batch['text_identifiers'] = tf.gather(
          new_batch['text_identifiers'], new_batch['mention_batch_positions'])
      return new_batch

    return collater_fn

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Produces model-specific dummy input batch. See BaseTask."""

    dummy_input = mention_encoder_task.MentionEncoderTask.dummy_input(config)
    mention_position_shape = (config.max_mentions *
                              config.per_device_batch_size,)
    int_type = jnp.int32
    dummy_input['text_identifiers'] = jnp.ones(mention_position_shape, int_type)

    return dummy_input

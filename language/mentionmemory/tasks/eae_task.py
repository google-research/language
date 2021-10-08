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
"""Contains Entities as Experts pre-training task."""

from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import eae_encoder
from language.mentionmemory.modules import mention_losses
from language.mentionmemory.modules import mlm_layer
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections


class EaEModel(nn.Module):
  """Entities as Experts (EaE) pre-training model.

  Attributes:
    encoder_config: EaE encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = eae_encoder.EaEEncoder(**self.encoder_config)
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


@task_registry.register_task('eae')
class EaETask(mention_encoder_task.MentionEncoderTask):
  """Task for pre-training Entities as Experts (EaE) encoder."""

  model_class = EaEModel
  encoder_name = 'eae'

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates task loss function.

    See BaseTask.

    EaE is pre-trained with a combination of 1) MLM loss, 2) entity-linking loss
    comparing mention encodings to entity embeddings at the retrieval and final
    layers, and 3) Matching the Blanks-style loss encouraging mentions of the
    same entity which co-occur with mentions of the same second entity to have
    similar representations.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """
    mlm_weight = config.mlm_weight
    el_im_weight = config.el_im_weight
    el_final_weight = config.el_final_weight
    el_score_mode = config.get('el_score_mode', 'dot')
    mtb_im_weight = config.get('mtb_im_weight', 0)
    mtb_final_weight = config.get('mtb_final_weight', 0)
    mtb_score_mode = config.get('mtb_score_mode', 'dot')

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[str, Any],
        model_vars: Dict[str, Any],  # pylint: disable=unused-argument
        batch: Dict[str, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[str, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Task-specific loss function. See BaseTask."""

      batch_size = batch['text_ids'].shape[0]
      loss_helpers, logging_helpers = cls.build_model(model_config).apply(  # pylint: disable=unused-variable
          {'params': model_params},
          batch,
          deterministic=deterministic,
          rngs=dropout_rng)
      mention_target_is_masked = batch['mention_target_is_masked']
      mention_target_is_not_masked = 1 - batch['mention_target_is_masked']
      mention_target_ids = batch['mention_target_ids']
      mention_target_ids = mention_target_ids * batch['mention_target_weights']

      mlm_logits = loss_helpers['mlm_logits']

      mlm_loss, mlm_denom = metric_utils.compute_weighted_cross_entropy(
          mlm_logits, batch['mlm_target_ids'], batch['mlm_target_weights'])

      mlm_correct_mask = jnp.equal(
          jnp.argmax(mlm_logits, axis=-1),
          batch['mlm_target_ids']) * batch['mlm_target_weights']
      mlm_acc = mlm_correct_mask.sum()
      mlm_mention_acc = (mlm_correct_mask *
                         batch['mlm_target_is_mention']).sum()
      mlm_mention_denom = (batch['mlm_target_weights'] *
                           batch['mlm_target_is_mention']).sum()
      mlm_non_mention_acc = (mlm_correct_mask *
                             (1 - batch['mlm_target_is_mention'])).sum()
      mlm_non_mention_denom = (batch['mlm_target_weights'] *
                               (1 - batch['mlm_target_is_mention'])).sum()

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

      if 'intermediate_mention_encodings' in loss_helpers:
        intermediate_target_mention_encodings = jut.matmul_slice(
            loss_helpers['intermediate_mention_encodings'],
            batch['mention_target_indices'])
      else:
        intermediate_target_mention_encodings = loss_helpers[
            'im_target_mention_encodings']

      if model_config.encoder_config.get('no_entity_attention', False):
        (el_im_loss, el_im_metrics,
         (el_im_acc_per_mention,
          el_im_weight_per_mention)) = mention_losses.entity_linking_loss(
              intermediate_target_mention_encodings,
              loss_helpers['entity_embeddings'], mention_target_ids,
              batch['mention_target_weights'], el_score_mode)
        el_im_denom = el_im_metrics['denominator']
        metrics['el_intermediate'] = el_im_metrics
        metrics['el_intermediate_masked'] = {
            'acc':
                jnp.dot(el_im_acc_per_mention,
                        el_im_weight_per_mention * mention_target_is_masked),
            'denominator':
                jnp.dot(el_im_weight_per_mention, mention_target_is_not_masked),
        }
        metrics['el_intermediate_non_masked'] = {
            'acc':
                jnp.dot(el_im_acc_per_mention,
                        el_im_weight_per_mention * mention_target_is_masked),
            'denominator':
                jnp.dot(el_im_weight_per_mention, mention_target_is_not_masked),
        }
      else:
        intermediate_entity_attention = loss_helpers[
            'intermediate_entity_attention']

        # Construct targets and ids for intermediate entity linking loss
        intermediate_target_ids = jnp.zeros_like(batch['mention_mask'])
        intermediate_target_ids = intermediate_target_ids.at[
            batch['mention_target_indices']].add(
                mention_target_ids * batch['mention_target_weights'])

        intermediate_target_weights = jnp.zeros_like(
            batch['mention_mask'], dtype=intermediate_entity_attention.dtype)
        intermediate_target_weights = intermediate_target_weights.at[
            batch['mention_target_indices']].add(
                batch['mention_target_weights'])

        mention_is_masked = jnp.zeros_like(batch['mention_mask'])
        mention_is_masked = mention_is_masked.at[
            batch['mention_target_indices']].add(
                mention_target_is_masked * batch['mention_target_weights'])

        el_im_loss, el_im_denom = metric_utils.compute_weighted_cross_entropy(
            intermediate_entity_attention,
            intermediate_target_ids,
            intermediate_target_weights,
            inputs_are_prob=True)

        el_im_correct_mask = jnp.equal(
            jnp.argmax(intermediate_entity_attention, axis=-1),
            intermediate_target_ids) * intermediate_target_weights
        el_im_acc = el_im_correct_mask.sum()

        el_im_acc, _ = metric_utils.compute_weighted_accuracy(
            intermediate_entity_attention, intermediate_target_ids,
            intermediate_target_weights)

        intermediate_entity_cos_sim = loss_helpers[
            'intermediate_entity_cos_sim'][batch['mention_target_indices'],
                                           mention_target_ids]

        metrics['el_intermediate'] = {
            'loss':
                el_im_loss,
            'acc':
                el_im_acc,
            'cos_sim':
                jnp.dot(intermediate_entity_cos_sim,
                        batch['mention_target_weights']),
            'denominator':
                el_im_denom,
        }
        metrics['el_intermediate_masked'] = {
            'acc':
                jnp.dot(el_im_correct_mask, mention_is_masked),
            'denominator':
                jnp.dot(batch['mention_target_weights'],
                        batch['mention_target_is_masked']),
        }
        metrics['el_intermediate_non_masked'] = {
            'acc':
                jnp.dot(el_im_correct_mask, (1 - mention_is_masked)),
            'denominator':
                jnp.dot(batch['mention_target_weights'],
                        (1 - batch['mention_target_is_masked'])),
        }

        im_final_mention_encodings_cos_sim = jut.cosine_similarity(
            intermediate_target_mention_encodings,
            loss_helpers['target_mention_encodings'])
        metrics['im_final_mention_encodings'] = {
            'cos_sim':
                jnp.dot(im_final_mention_encodings_cos_sim,
                        batch['mention_target_weights']),
            'denominator':
                batch['mention_target_weights'].sum(),
        }

      (el_final_loss, el_final_metrics,
       (el_final_acc_per_mention,
        el_final_weight_per_mention)) = mention_losses.entity_linking_loss(
            loss_helpers['target_mention_encodings'],
            loss_helpers['entity_embeddings'], mention_target_ids,
            batch['mention_target_weights'], el_score_mode)
      el_final_denom = el_final_metrics['denominator']
      metrics['el_final'] = el_final_metrics
      metrics['el_final_masked'] = {
          'acc':
              jnp.dot(el_final_acc_per_mention,
                      el_final_weight_per_mention * mention_target_is_masked),
          'denominator':
              jnp.dot(el_final_weight_per_mention, mention_target_is_masked),
      }
      metrics['el_final_non_masked'] = {
          'acc':
              jnp.dot(
                  el_final_acc_per_mention,
                  el_final_weight_per_mention * mention_target_is_not_masked),
          'denominator':
              jnp.dot(el_final_weight_per_mention,
                      mention_target_is_not_masked),
      }

      loss = mlm_weight * mlm_loss / mlm_denom
      loss += el_im_weight * el_im_loss / el_im_denom
      loss += el_final_weight * el_final_loss / el_final_denom

      if mtb_im_weight > 0:
        (mtb_im_loss, mtb_im_metrics) = mention_losses.mtb_loss(
            intermediate_target_mention_encodings,
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

      metrics['agg'] = {
          'loss': loss,
          'denominator': 1.0,
      }
      return loss, metrics, {}

    return loss_fn

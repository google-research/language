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
"""Contains ReadTwice task."""



import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import readtwice_encoder
from language.mentionmemory.modules import mention_losses
from language.mentionmemory.modules import mlm_layer
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections


class ReadTwiceModel(nn.Module):
  """Implementation of ReadTwice Model.

  Attributes:
    encoder_config: ReadTwice encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = readtwice_encoder.ReadTwiceEncoder(**self.encoder_config)

    self.mlm_layer = mlm_layer.MLMLayer(
        vocab_size=self.encoder.vocab_size,
        hidden_size=self.encoder.hidden_size,
        dtype=self.encoder.dtype,
        layer_norm_epsilon=self.encoder.layer_norm_epsilon,
        embedding_init=self.encoder.kernel_init,
        bias_init=self.encoder.bias_init,
    )

  def __call__(self, batch, deterministic):
    encoding, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    loss_helpers['mlm_logits_first'] = self.mlm_layer(
        encoding, batch['mlm_target_positions'],
        loss_helpers['word_embeddings'])
    loss_helpers['mlm_logits_second'] = self.mlm_layer(
        encoding, batch['mlm_target_positions'],
        loss_helpers['word_embeddings'])

    return loss_helpers, logging_helpers


@task_registry.register_task('read_twice')
class ReadTwiceTask(mention_encoder_task.MentionEncoderTask):
  """Pre-training task for ReadTwice encoder."""

  model_class = ReadTwiceModel
  encoder_name = 'read_twice'

  @classmethod
  def make_loss_fn(
      cls, config
  ):
    """Creates task loss function.

    See BaseTask.

    The ReadTwice encoder is pre-trained with a combination of 1) MLM loss,
    2) same-entity retrieval loss encouraging retrieval of mentions of the same
    entity as the passage mention, 3) entity coreference loss encouraging
    mentions of the same entity to have similar representations, and 4) Matching
    the Blanks-style loss encouraging mentions of the same entity which co-occur
    with mentions of the same second entity to have similar representations.
    Entity coreference and MTB losses can be applied to memory keys and values,
    as well as the final mention encodings.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """

    mlm_weight = config.mlm_weight
    mlm_first_weight = config.mlm_first_weight
    el_im_weight = config.el_im_weight
    second_el_im_weight = config.get('second_el_im_weight', 0)
    no_retrieval = config.model_config.encoder_config.get('no_retrieval', False)

    mention_type_dict = {
        'key': 'memory_keys',
        'value': 'memory_values',
        'final': 'target_mention_encodings',
    }

    coref_weights = {}
    for mention_type in mention_type_dict:
      weight = config.get('coref_' + mention_type + '_weight')
      assert weight is not None
      coref_weights[mention_type] = weight

    mtb_weights = {}
    for mention_type in mention_type_dict:
      weight = config.get('mtb_' + mention_type + '_weight')
      assert weight is not None
      mtb_weights[mention_type] = weight

    coref_res_mode = config.get('coref_res_mode', 'dot')
    mtb_score_mode = config.get('mtb_score_mode', 'dot')

    def loss_fn(
        model_config,
        model_params,
        model_vars,
        batch,
        deterministic,
        dropout_rng = None,
    ):
      """Model-specific loss function. See BaseTask."""

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, logging_helpers = cls.build_model(model_config).apply(  # pylint: disable=unused-variable
          variable_dict,
          batch,
          deterministic=deterministic,
          rngs=dropout_rng)

      mlm_logits = loss_helpers['mlm_logits_second']
      mlm_target_is_mention = batch['mlm_target_is_mention']
      mlm_target_is_not_mention = 1 - batch['mlm_target_is_mention']
      mention_target_is_masked = batch['mention_target_is_masked']

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

      loss = mlm_weight * mlm_loss / (mlm_denom + default_values.SMALL_NUMBER)
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

      if mlm_first_weight > 0:
        mlm_logits_first = loss_helpers['mlm_logits_first']
        mlm_loss_first, _ = metric_utils.compute_weighted_cross_entropy(
            mlm_logits_first, batch['mlm_target_ids'],
            batch['mlm_target_weights'])
        correct_mask_first = jnp.equal(
            jnp.argmax(mlm_logits_first, axis=-1),
            batch['mlm_target_ids']) * batch['mlm_target_weights']
        mlm_acc_first = correct_mask_first.sum()
        mlm_mention_acc_first = (correct_mask_first *
                                 mlm_target_is_mention).sum()
        mlm_non_mention_acc_first = (correct_mask_first *
                                     mlm_target_is_not_mention).sum()
        metrics.update({
            'mlm_first': {
                'loss': mlm_loss_first,
                'acc': mlm_acc_first,
                'denominator': mlm_denom,
            },
            'mlm_mention_first': {
                'acc': mlm_mention_acc_first,
                'denominator': mlm_mention_denom,
            },
            'mlm_non_mention_first': {
                'acc': mlm_non_mention_acc_first,
                'denominator': mlm_non_mention_denom,
            },
        })
        loss += mlm_first_weight * mlm_loss_first / (
            mlm_denom + default_values.SMALL_NUMBER)

      def apply_same_entity_retrieval_loss(loss,
                                           metrics,
                                           loss_weight,
                                           prefix=''):
        if prefix + 'top_entity_ids' not in loss_helpers:
          if loss_weight > 0:
            raise KeyError('%s not found in loss helpers' % prefix +
                           'top_entity_ids')
          else:
            return loss, metrics
        memory_entity_ids = loss_helpers[prefix + 'top_entity_ids']
        memory_attention_weights = loss_helpers[prefix +
                                                'memory_attention_weights']
        intermediate_entity_probs = jut.matmul_slice(
            memory_attention_weights, batch['mention_target_indices'])

        intermediate_entity_ids = jut.matmul_slice(
            memory_entity_ids, batch['mention_target_indices'])

        el_loss_intermediate, same_entity_avg_prob, el_im_denom = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
            intermediate_entity_probs, intermediate_entity_ids,
            batch['mention_target_ids'], batch['mention_target_weights'])
        metrics[prefix + 'el_intermediate'] = {
            'loss': el_loss_intermediate,
            'same_entity_avg_prob': same_entity_avg_prob,
            'denominator': el_im_denom,
        }
        if loss_weight > 0:
          loss += loss_weight * el_loss_intermediate / (
              el_im_denom + default_values.SMALL_NUMBER)

        return loss, metrics

      if not no_retrieval:
        loss, metrics = apply_same_entity_retrieval_loss(
            loss, metrics, el_im_weight)
        loss, metrics = apply_same_entity_retrieval_loss(
            loss, metrics, second_el_im_weight, 'second_')

      batch_size = batch['text_ids'].shape[0]
      mention_target_ids = batch['mention_target_ids']
      mention_target_ids *= batch['mention_target_weights']

      for mention_type, mention_type_key in mention_type_dict.items():
        if coref_weights[mention_type] > 0:
          (coref_res_loss,
           coref_res_metrics) = mention_losses.coreference_resolution_loss(
               mention_encodings=loss_helpers[mention_type_key],
               mention_batch_positions=batch['mention_target_batch_positions'],
               mention_target_ids=mention_target_ids,
               batch_size=batch_size,
               mode=coref_res_mode,
               mention_target_is_masked=mention_target_is_masked,
               metrics_prefix=mention_type + '_',
           )
          coref_res_denom = coref_res_metrics[
              mention_type + '_coref_resolution']['denominator']
          loss += coref_weights[mention_type] * coref_res_loss / (
              coref_res_denom + default_values.SMALL_NUMBER)
          metrics.update(coref_res_metrics)

        if mtb_weights[mention_type] > 0:
          (mtb_loss, mtb_metrics) = mention_losses.mtb_loss(
              mention_encodings=loss_helpers[mention_type_key],
              mention_batch_positions=batch['mention_target_batch_positions'],
              mention_target_ids=mention_target_ids,
              batch_size=batch_size,
              mode=mtb_score_mode,
              mention_target_is_masked=mention_target_is_masked,
              metrics_prefix=mention_type + '_',
          )
          mtb_denom = mtb_metrics[mention_type + '_mtb']['denominator']
          loss += mtb_weights[mention_type] * mtb_loss / (
              mtb_denom + default_values.SMALL_NUMBER)
          metrics.update(mtb_metrics)

      metrics['agg'] = {
          'loss': loss,
          'denominator': 1.0,
      }

      return loss, metrics, {}

    return loss_fn

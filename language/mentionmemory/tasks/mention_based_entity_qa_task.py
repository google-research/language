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
"""Contains entity-answer question answering tasks via mention memory."""



import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import mention_memory_encoder
from language.mentionmemory.tasks import entity_qa_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections


def get_predictions_max(attention_weights, memory_entity_ids,
                        weights):
  """Predict entity ID based on a memory with the largest attention weight."""
  # `stop_gradient` is a safety check so the model doesn't keep activations
  # around, which could be expensive.
  attention_weights = jax.lax.stop_gradient(attention_weights)
  memory_entity_ids = jax.lax.stop_gradient(memory_entity_ids)
  weights = jax.lax.stop_gradient(weights)
  memory_with_largest_attn = jnp.argmax(attention_weights, axis=1)
  predictions = jnp.take_along_axis(
      memory_entity_ids, jnp.expand_dims(memory_with_largest_attn, 1), axis=1)
  predictions = jnp.squeeze(predictions, axis=1)
  predictions = predictions * weights
  return predictions


def get_predictions_sum(attention_weights, memory_entity_ids,
                        weights, entity_vocab_size):
  """Get entity with the largest sum of attention weights over its mentions."""
  attention_weights = jax.lax.stop_gradient(attention_weights)
  memory_entity_ids = jax.lax.stop_gradient(memory_entity_ids)
  weights = jax.lax.stop_gradient(weights)

  n_mentions = attention_weights.shape[0]
  attention_weights_per_entity = jnp.zeros((n_mentions, entity_vocab_size),
                                           dtype=attention_weights.dtype)
  attention_weights_per_entity = jax.ops.index_add(
      attention_weights_per_entity,
      (jnp.expand_dims(jnp.arange(n_mentions), 1), memory_entity_ids),
      attention_weights)
  predictions = jnp.argmax(attention_weights_per_entity, axis=1)
  predictions = predictions * weights
  return predictions


class MentionMemoryQAModel(nn.Module):
  """Mention Memory model for entity-seeking question-answering task.

  Attributes:
    encoder_config: Mention Memory encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = mention_memory_encoder.MentionMemoryEncoder(
        **self.encoder_config)

  def __call__(
      self, batch,
      deterministic):
    _, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    return loss_helpers, logging_helpers


@task_registry.register_task('mention_based_entity_qa')
class MentionBasedEntityQATask(entity_qa_task.EntityQATask):
  """Class for all entity seeking question answering tasks using EaE model."""

  model_class = MentionMemoryQAModel
  encoder_name = 'mention_memory'

  @classmethod
  def make_loss_fn(
      cls, config
  ):
    """Creates task loss function.

    See BaseTask.

    Model is trained using same-entity retrieval loss encouraging retrieval of
    mentions of the same entity as the passage mention. See MentionMemoryTask
    for details.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """

    def loss_fn(
        model_config,
        model_params,
        model_vars,  # pylint: disable=unused-argument
        batch,
        deterministic,
        dropout_rng = None,
    ):
      """Task-specific loss function. See BaseTask."""

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, logging_helpers = cls.build_model(model_config).apply(  # pylint: disable=unused-variable
          variable_dict,
          batch,
          deterministic=deterministic,
          rngs=dropout_rng)

      qa_attention_weights = loss_helpers['final_memory_attention_weights']
      qa_memory_entity_ids = loss_helpers['final_top_entity_ids']
      qa_entity_ids = batch['mention_target_ids']
      qa_weights = batch['mention_target_weights']

      (loss, correct_entity_avg_prob, loss_denom
      ) = metric_utils.compute_loss_and_prob_from_probs_with_duplicates(
          qa_attention_weights, qa_memory_entity_ids, qa_entity_ids, qa_weights)

      predictions_max = get_predictions_max(qa_attention_weights,
                                            qa_memory_entity_ids, qa_weights)
      acc_max = (predictions_max == qa_entity_ids) * qa_weights

      predictions_sum = get_predictions_sum(qa_attention_weights,
                                            qa_memory_entity_ids, qa_weights,
                                            config.entity_vocab_size)
      acc_sum = (predictions_sum == qa_entity_ids) * qa_weights

      metrics = {
          'agg': {
              'loss': loss,
              'correct_entity_avg_prob': correct_entity_avg_prob,
              'acc_max': acc_max,
              'acc_sum': acc_sum,
              'denominator': loss_denom,
          }
      }

      auxiliary_output = {
          'predictions_max':
              predictions_max,
          'predictions_sum':
              predictions_sum,
          # Save final retrievals for debugging purposes.
          'memory_attention_weights':
              loss_helpers['final_memory_attention_weights'],
          'top_entity_ids':
              loss_helpers['final_top_entity_ids'],
          'top_memory_ids':
              loss_helpers['final_top_memory_ids'],
      }

      return loss, metrics, auxiliary_output

    return loss_fn

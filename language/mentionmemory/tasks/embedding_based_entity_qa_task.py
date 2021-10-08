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
"""Contains entity-answer question answering tasks via entity embeddings."""

from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
from language.mentionmemory.encoders import eae_encoder
from language.mentionmemory.modules import mention_losses
from language.mentionmemory.tasks import entity_qa_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections


class EaEQAModel(nn.Module):
  """Entities as Experts (EaE) model for entity-answer question-answering task.

  Attributes:
    encoder_config: EaE encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = eae_encoder.EaEEncoder(**self.encoder_config)

  def __call__(
      self, batch: Dict[str, Array],
      deterministic: bool) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    _, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    return loss_helpers, logging_helpers


@task_registry.register_task('embedding_based_entity_qa')
class EmbeddingBasedEntityQATask(entity_qa_task.EntityQATask):
  """Class for all entity-answer question answering tasks using EaE model."""

  model_class = EaEQAModel
  encoder_name = 'eae'

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates task loss function.

    See BaseTask.

    Model is trained using entity linking loss.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """
    el_score_mode = config.get('el_score_mode', 'dot')

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[str, Any],
        model_vars: Dict[str, Any],  # pylint: disable=unused-argument
        batch: Dict[str, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[str, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Task-specific loss function. See BaseTask."""

      loss_helpers, logging_helpers = cls.build_model(model_config).apply(  # pylint: disable=unused-variable
          {'params': model_params},
          batch,
          deterministic=deterministic,
          rngs=dropout_rng)

      mention_target_ids = batch['mention_target_ids']
      mention_target_ids = mention_target_ids * batch['mention_target_weights']

      (loss, el_final_metrics, _) = mention_losses.entity_linking_loss(
          loss_helpers['target_mention_encodings'],
          loss_helpers['entity_embeddings'], mention_target_ids,
          batch['mention_target_weights'], el_score_mode)

      metrics = {'agg': el_final_metrics}

      return loss, metrics, {}

    return loss_fn

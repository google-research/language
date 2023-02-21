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
"""Ultra Fine Entity Typing mention classification task."""

from typing import Any, Callable, Dict, Optional, Text, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.tasks import mention_classifier_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections

NUM_CLASSES = 10331
COARSE_CLASSES_START = 0
COARSE_CLASSES_END = 9
FINE_CLASSES_START = COARSE_CLASSES_END
FINE_CLASSES_END = 130
ULTRA_FINE_CLASSES_START = FINE_CLASSES_END
ULTRA_FINE_CLASSES_END = NUM_CLASSES

_SMALL_NUMBER = 1e-10


def get_weight_per_group(labels: Array, group_start: int,
                         group_end: int) -> Array:
  """Computes which samples have at least one labels within a group."""
  label_per_group_exists = labels[:, group_start:group_end].sum(1) > 0
  label_per_group_exists = label_per_group_exists.astype(jnp.float32)
  return label_per_group_exists


def get_loss_per_group(loss_per_label: Array, weight_per_group: Array,
                       group_start: int, group_end: int) -> Array:
  """Computes loss per sample within a group of labels."""
  loss_per_group = loss_per_label[:, group_start:group_end].sum(1)
  loss_per_group *= weight_per_group
  return loss_per_group


def get_predictions(logit_per_label: Array) -> Array:
  """Prediction according to https://www.aclweb.org/anthology/P18-1009.pdf."""
  num_labels = logit_per_label.shape[1]
  # Independent (per-label) predictions
  predictions = (logit_per_label > 0).astype(jnp.int32)
  # A single most confident prediction
  single_best_prediction = jnp.argmax(logit_per_label, axis=-1)
  single_best_prediction = jax.nn.one_hot(
      single_best_prediction, num_labels, dtype=jnp.int32)
  predictions_exists = predictions.sum(axis=1, keepdims=True) > 0
  predictions_exists = predictions_exists.astype(jnp.int32)
  final_predictions = (
      predictions_exists * predictions +
      (1 - predictions_exists) * single_best_prediction)
  return final_predictions


def get_mrr(labels: Array, logits: Array) -> Array:
  """Mean reciprocal rank in https://www.aclweb.org/anthology/P18-1009.pdf."""
  labels_exists = labels.sum(axis=-1) > 0
  labels_exists = labels_exists.astype(jnp.float32)
  order = jnp.argsort(-logits, axis=-1)
  ranks = jnp.argsort(order, axis=-1)
  mrr_per_sample = 1.0 / (ranks + 1)
  mrr_per_sample = (labels * mrr_per_sample).sum(-1) / (
      labels.sum(axis=-1) + 1e-5)
  return {  # pytype: disable=bad-return-type  # jax-ndarray
      'value': jnp.dot(mrr_per_sample, labels_exists),
      'denominator': labels_exists.sum(),
  }


def get_positives_negatives(metric_name: str, labels: Array, predictions: Array,
                            group_start: int, group_end: int) -> MetricGroups:
  """Computes metrics over precision and recall for a specific groups."""
  tp = jnp.logical_and(labels[:, group_start:group_end] == 1,
                       predictions[:, group_start:group_end] == 1).sum(-1)
  fp = jnp.logical_and(labels[:, group_start:group_end] == 0,
                       predictions[:, group_start:group_end] == 1).sum(-1)
  fn = jnp.logical_and(labels[:, group_start:group_end] == 1,
                       predictions[:, group_start:group_end] == 0).sum(-1)
  precision = tp / (tp + fp + _SMALL_NUMBER)
  precision_weight = (tp + fp) > 0
  recall = tp / (tp + fn + _SMALL_NUMBER)
  recall_weight = (tp + fn) > 0
  return {
      metric_name + '_precision': {
          'value': jnp.dot(precision, precision_weight),
          'denominator': precision_weight.sum(),
      },
      metric_name + '_recall': {
          'value': jnp.dot(recall, recall_weight),
          'denominator': recall_weight.sum(),
      }
  }


def get_prediction_recall_metrics(labels: Array,
                                  predictions: Array) -> MetricGroups:
  """Computes metrics over precision and recall over different groups."""
  metrics = {}
  metrics.update(
      get_positives_negatives('total', labels, predictions, 0, NUM_CLASSES))
  metrics.update(
      get_positives_negatives('coarse_grained', labels, predictions,
                              COARSE_CLASSES_START, COARSE_CLASSES_END))
  metrics.update(
      get_positives_negatives('fine_grained', labels, predictions,
                              FINE_CLASSES_START, FINE_CLASSES_END))
  metrics.update(
      get_positives_negatives('ultra_fine_grained', labels, predictions,
                              ULTRA_FINE_CLASSES_START, ULTRA_FINE_CLASSES_END))
  return metrics


def get_eval_metrics(labels: Array, logits: Array) -> MetricGroups:
  predictions = get_predictions(logits)
  metrics = get_prediction_recall_metrics(labels, predictions)
  metrics['agg_mrr'] = get_mrr(labels, logits)
  return metrics  # pytype: disable=bad-return-type  # jax-ndarray


@task_registry.register_task('ultra_fine_entity_typing')
class UltraFineEntityTypingTask(mention_classifier_task.MentionClassifierTask):
  """Ultra Fine Entity Typing benchmark.

  TODO(urikz): Write detailed description.
  See https://www.aclweb.org/anthology/P18-1009.pdf for details.
  """

  @classmethod
  def build_model(cls,
                  model_config: ml_collections.FrozenConfigDict) -> nn.Module:
    """Builds model by instantiating flax module associated with task."""
    return cls.model_class(num_classes=NUM_CLASSES, **model_config)

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates loss function for Ultra Fine Entity Typing training.

    TODO(urikz): Write detailed description.
    See https://www.aclweb.org/anthology/P18-1009.pdf for details how loss
    is being computed.

    Args:
      config: task configuration.

    Returns:
      Loss function.
    """

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[Text, Any],
        model_vars: Dict[Text, Any],
        batch: Dict[Text, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[Text, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Loss function used by Ultra Fine Entity Typing task. See BaseTask."""

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, _ = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      classifier_logits = loss_helpers['classifier_logits'].astype(jnp.float32)
      log_prob = jax.nn.log_sigmoid(classifier_logits)
      # log(1 - sigmoid(x)) = log_sigmoid(-x)
      # We use the latter since it is more numerically stable and denote it
      # as `log_comp_prob` (log of probability of the complimentary event).
      log_comp_prob = jax.nn.log_sigmoid(-classifier_logits)

      # batch['classifier_target'] has shape [batch_size, max_labels_per_sample]
      # and contain all labels in a sparse format. The code below converts
      # this to a dense format.
      classifier_labels = jax.nn.one_hot(
          batch['classifier_target'], NUM_CLASSES, dtype=jnp.float32)
      classifier_labels *= jnp.expand_dims(batch['classifier_target_mask'], -1)
      # Labels in a dense format with a shape [batch_size, NUM_CLASSES]
      classifier_labels = classifier_labels.sum(axis=1)
      loss_per_label = -log_prob * classifier_labels - log_comp_prob * (
          1.0 - classifier_labels)

      coarse_grained_weight = get_weight_per_group(classifier_labels,
                                                   COARSE_CLASSES_START,
                                                   COARSE_CLASSES_END)
      fine_grained_weight = get_weight_per_group(classifier_labels,
                                                 FINE_CLASSES_START,
                                                 FINE_CLASSES_END)
      ultra_fine_grained_weight = get_weight_per_group(
          classifier_labels, ULTRA_FINE_CLASSES_START, ULTRA_FINE_CLASSES_END)

      coarse_grained_loss = get_loss_per_group(loss_per_label,
                                               coarse_grained_weight,
                                               COARSE_CLASSES_START,
                                               COARSE_CLASSES_END)
      fine_grained_loss = get_loss_per_group(loss_per_label,
                                             fine_grained_weight,
                                             FINE_CLASSES_START,
                                             FINE_CLASSES_END)
      ultra_fine_grained_loss = get_loss_per_group(loss_per_label,
                                                   ultra_fine_grained_weight,
                                                   ULTRA_FINE_CLASSES_START,
                                                   ULTRA_FINE_CLASSES_END)
      loss_per_sample = (
          coarse_grained_loss + fine_grained_loss + ultra_fine_grained_loss)
      loss = loss_per_sample.sum()

      metrics = {
          'agg': {
              'loss': loss,
              'denominator': loss_per_sample.shape[0],
          },
          'coarse_grained': {
              'loss': coarse_grained_loss.sum(),
              'denominator': coarse_grained_weight.sum(),
          },
          'fine_grained': {
              'loss': fine_grained_loss.sum(),
              'denominator': fine_grained_weight.sum(),
          },
          'ultra_fine_grained': {
              'loss': ultra_fine_grained_loss.sum(),
              'denominator': ultra_fine_grained_weight.sum(),
          },
      }
      metrics.update(get_eval_metrics(classifier_labels, classifier_logits))
      return loss, metrics, {}  # pytype: disable=bad-return-type  # jax-ndarray

    return loss_fn

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
"""Utilities for processing loss and metrics."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from language.mentionmemory.utils.custom_types import Array

_SMALL_NUMBER = 1e-10
_BIG_NUMBER = 1e10


def compute_weighted_cross_entropy(
    scores: Array,
    targets: Array,
    weights: Array,
    inputs_are_prob: Optional[bool] = False,
) -> Tuple[float, float]:
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   scores: [batch, length, num_classes] float array.
   targets: [batch, length] categorical target integer array.
   weights: [batch, length].
   inputs_are_prob: true if inputs are probabilities rather than logits.

  Returns:
    Tuple of scalar loss and batch denominator.
  """
  scores = scores.astype(jnp.float32)
  targets = targets.astype(jnp.float32)
  weights = weights.astype(jnp.float32)
  if scores.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s scores and %s targets' %
                     (str(scores.shape), str(targets.shape)))
  vocab_size = scores.shape[-1]
  soft_targets = jax.nn.one_hot(targets, vocab_size)

  if inputs_are_prob:
    loss = -jnp.sum(soft_targets * jnp.log(scores + _SMALL_NUMBER), axis=-1)
  else:
    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(scores), axis=-1)

  loss = loss * weights
  normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor  # pytype: disable=bad-return-type  # jax-ndarray


def compute_weighted_accuracy(scores: Array, targets: Array,
                              weights: Array) -> Tuple[float, float]:
  """Compute weighted accuracy for log probs and targets.

  Args:
   scores: [batch, length, num_classes] float array.
   targets: [batch, length] categorical targets int array.
   weights: [batch, length].

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if scores.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s scores and %s targets' %
                     (str(scores.shape), str(targets.shape)))
  acc = jnp.equal(jnp.argmax(scores, axis=-1), targets)
  acc = acc * weights
  normalizing_factor = weights.sum()

  return acc.sum(), normalizing_factor  # pytype: disable=bad-return-type  # jax-ndarray


def compute_tp_fp_fn_weighted(
    predictions: Array, labels: Array, weights: Array,
    ignore_class: Optional[int]) -> Tuple[float, float, float]:
  """Compute true positives, false positives and false negatives.

  Args:
   predictions: [batch, length] categorical predictions int array.
   labels: [batch, length] categorical labels int array.
   weights: [batch, length].
   ignore_class: which class to ignore in the computations

  Returns:
    Tuple with numbers of true positive, false positive and false negative
      predictions.
  """
  true_positives = (predictions == labels)
  false_positives = jnp.logical_not(true_positives)
  false_negatives = false_positives

  if ignore_class is not None:
    dont_ignore_predictions = (predictions != ignore_class)
    dont_ignore_labels = (labels != ignore_class)
    true_positives = jnp.logical_and(true_positives, dont_ignore_predictions)
    # Exactly the same as
    # true_positives = jnp.logical_and(true_positives, dont_ignore_labels)
    # since for true positives `dont_ignore_predictions` = `dont_ignore_labels`.
    false_positives = jnp.logical_and(false_positives, dont_ignore_predictions)
    false_negatives = jnp.logical_and(false_negatives, dont_ignore_labels)

  def get_weighted_sum(values):
    values = values.astype(weights.dtype)
    return jnp.dot(values, weights)

  n_true_positive = get_weighted_sum(true_positives)
  n_false_positive = get_weighted_sum(false_positives)
  n_false_negative = get_weighted_sum(false_negatives)

  return n_true_positive, n_false_positive, n_false_negative


def compute_loss_and_prob_from_probs_with_duplicates(
    probs: Array,
    classes: Array,
    targets: Array,
    weights: Array,
) -> Tuple[float, float, float]:
  """Compute weighted loss and avg correct probability given probs and targets.

  Args:
   probs: [batch, length, num_items] float array.
   classes: [batch, length, num_items] class for each item.
   targets: [batch, length] categorical target int array.
   weights:  [batch, length].

  Returns:
    Tuple of scalar loss, avg correct probability and normalizing factor.
  """
  probs = probs.astype(jnp.float32)
  weights = weights.astype(jnp.float32)

  correct_mask = (classes == jnp.expand_dims(targets, axis=-1))
  correct_mask = correct_mask.astype(jnp.float32)

  correct_probs = (correct_mask * probs).sum(axis=-1)
  avg_probs = correct_probs * weights
  loss = -jnp.log(correct_probs + _SMALL_NUMBER)
  loss = loss * weights

  return loss.sum(), avg_probs.sum(), weights.sum()  # pytype: disable=bad-return-type  # jax-ndarray


def compute_cross_entropy_loss_with_positives_and_negatives_masks(
    scores: Array,
    positives: Array,
    negatives: Array,
    weights: Optional[Array] = None,
) -> Tuple[float, Dict[str, float], Tuple[Array, Array]]:
  """Compute (weighted) cross-entropy loss and accuracy-related metrics.

  The function computes cross entropy loss when there are potentially multiple
  positive classes per sample, multiple negative classes and others are neutral.
  In this case, loss per sample is average of cross entropy losses computed
  by considering each positive class and all negative classes.
  Neutral classes are ignored.

  Arguments `positives` and `negatives` are boolean matrices that specify
  which class is considered positive or negative per every sample.
  `positive[i, j]` is True <=> class j is considered positive for the sample i
  `negative[i, j]` is True <=> class j is considered negative for the sample i

  The loss is computed in 3 stages:

  (1) For every sample i and positive class j we compute cross-entropy loss
  using j as a positive class and all negative classes for i as negatives.

  (2) For every sample i the total loss is average of losses per each of its
  positive classes.

  (3) Total loss is a sum of losses per each sample. The loss only includes
  samples, which have at least one positive and one negative classes. Users
  can limit this even further by providing a custom `weights`.

  Args:
   scores: [batch_size, num_classes] scores or logits.
   positives: [batch_size, num_classes] 0-1 mask for which classes are positive.
   negatives: [batch_size, num_classes] 0-1 mask for which classes are negative.
   weights: [batch_size] 0-1 masks indicating whether the loss should be
     computed for the corresponding item in the batch.

  Returns:
    A tuple of scalar loss, a dictionary with metrics, per sample information
    (a tuple of average positive probability per sample and weight per sample).
  """
  at_least_one_positive_and_negative = jnp.logical_and(
      positives.sum(-1) > 0,
      negatives.sum(-1) > 0)
  if weights is None:
    weights = at_least_one_positive_and_negative
  else:
    weights = jnp.logical_and(weights, at_least_one_positive_and_negative)

  scores = scores.astype(jnp.float32)
  positives = positives.astype(jnp.float32)
  negatives = negatives.astype(jnp.float32)
  weights = weights.astype(jnp.float32)

  # For simplicity, we ignore the first batch dimension in the equations below
  # and assume that the loss is computed for a single sample.
  # Let p_1, ..., p_N be scores of positive classes
  # and n_1, ..., n_M be scores of negative classes.
  # In this case the loss is
  # sum_{i=1..N} -log softmax([p_i, n_1, ..., n_M])_1.
  # It's too computationally expensive to compute it naively.
  # We implement the loss in the following way

  # (1) compute S, the negatives part of softmax denominator. In other words,
  # exp(S) = sum_{j=1..M} exp(n_j)
  negative_scores = scores * negatives - _BIG_NUMBER * (1.0 - negatives)

  negative_scores_log_sum_exp = jax.nn.logsumexp(
      negative_scores, axis=-1, keepdims=True)

  # (2) now the loss per positive class i is just
  # -log (exp(p_i) / (exp(p_i) + exp(S)) = -log(1 / (1 + exp(-(p_i - S))))
  # = -log sigmoid(p_i - S)
  scores_minus_negatives = scores - negative_scores_log_sum_exp
  positives_weight = (positives.sum(axis=-1) + _SMALL_NUMBER)
  per_positive_loss = -jax.nn.log_sigmoid(scores_minus_negatives)

  # (3) compute average loss over all positive classes
  loss_per_sample = (per_positive_loss * positives).sum(axis=-1)
  loss_per_sample /= positives_weight
  loss_per_sample *= weights

  # (4) compute sum of losses over all positive samples
  loss = loss_per_sample.sum()

  # Now we need to compute the average accuracy.
  # First, compute the max score of negative classes per sample.
  # A positive class needs to have a higher score in order to get predicted.
  max_negative_scores = negative_scores.max(axis=-1, keepdims=True)

  # Second, a prediction for pair of a sample and its positive class
  # is correct if the score of the positive class is larger than
  # scores of all corresponding negative classes. In other words, the score
  # of the positive class needs to be larger than `max_negative_scores`.
  correct_prediction = (scores > max_negative_scores).astype(jnp.float32)

  # Take average over all positive classes per sample
  correct_prediction = (correct_prediction * positives).sum(axis=-1)
  correct_prediction /= positives_weight

  # Mask out samples with 0 weight
  correct_prediction = correct_prediction * weights

  metrics = {
      'loss': loss,
      'acc': correct_prediction.sum(),
      'denominator': weights.sum(),
  }
  return loss, metrics, (correct_prediction, weights)  # pytype: disable=bad-return-type  # jax-ndarray


def update_value_dtype(value: Any) -> Any:
  """Convert value to more precise type."""
  if isinstance(value, jnp.ndarray):
    return value.astype(jnp.float32)
  return value


def update_metrics_dtype(
    metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
  """Convert metrics to more precise types."""
  return jax.tree_map(update_value_dtype, metrics)


def process_metrics(
    metrics: Dict[str, Dict[str, Array]],
    prefix: Optional[str] = None,
) -> Dict[str, Union[float, int]]:
  """Flatten 2-level dictionary of metrics and divide values by denominator."""
  processed_metrics = {}
  for group_key, group_value in metrics.items():
    denom = group_value.pop('denominator')
    for metric_key, metric_value in group_value.items():
      flattened_key = group_key + '_' + metric_key
      if prefix is not None:
        flattened_key = prefix + '/' + flattened_key
      processed_metrics[flattened_key] = metric_value / denom
    denom_key = group_key + '_denom'
    if prefix is not None:
      denom_key = prefix + '/' + denom_key
    processed_metrics[denom_key] = denom

  return processed_metrics  # pytype: disable=bad-return-type  # jax-ndarray

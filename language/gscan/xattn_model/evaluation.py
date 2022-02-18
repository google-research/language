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
"""Evaluation utils."""

import jax.numpy as jnp
import numpy as np


def exact_match(logits, targets, weights=None):
  """Sequence exact match averaged over batch size."""
  if weights is None:
    weights = jnp.ones(targets.shape)
  predicted_targets = logits.argmax(-1)
  match_targets = (targets == predicted_targets).astype(jnp.float32) * weights
  match_sum_per_example = match_targets.sum(1)
  expected_sum_per_example = weights.sum(1)
  example_weights = expected_sum_per_example > 0
  em = (match_sum_per_example == expected_sum_per_example) * example_weights
  em = em.sum() * 100
  normalizing_factor = example_weights.sum()
  return em, normalizing_factor


def accuracy(logits, targets, weights=None):
  """Sequence accuracy averaged over sequence length."""
  if logits.ndim != targets.ndim + 1:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  acc = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    acc = acc * weights * 100
    normalizing_factor = weights.sum()
  return acc.sum(), normalizing_factor


def sequence_accuracy(prediction, target):
  """Compute a sequence accuracy.

  This follows the original evaluation in https://github.com/LauraRuis/
    gscan_seq2seq/blob/master/seq2seq/helpers.py.

  Args:
    prediction: a list of predicted tokens.
    target: a list of target tokens.

  Returns:
    The sequence accuracy.
  """
  correct = 0
  total = 0
  prediction = prediction.copy()
  target = target.copy()
  if len(prediction) < len(target):
    difference = len(target) - len(prediction)
    prediction.extend([0] * difference)
  if len(target) < len(prediction):
    difference = len(prediction) - len(target)
    target.extend([-1] * difference)
  for i, target_int in enumerate(target):
    if i >= len(prediction):
      break
    prediction_int = prediction[i]
    if prediction_int == target_int:
      correct += 1
    total += 1
  if not total:
    return 0.
  return (correct / total) * 100

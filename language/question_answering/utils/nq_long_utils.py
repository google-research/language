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
"""Utilities for NQ long model scoring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def truncate_contexts(context_token_ids, num_contexts, context_len,
                      max_contexts, max_context_len):
  """Truncate context based on max_contexts and max_context_words.

  Drops more than `max_contexts` contexts entirely, and truncates
  each remaining context to contain at most `max_context_words` words.

  Args:
    context_token_ids: <int32> [batch_size, padded_num_contexts,
      padded_context_len].
    num_contexts: <int32> [batch_size] for number of contexts in each batch.
    context_len: <int32> [batch_size, padded_num_contexts] for length of each
      context.
    max_contexts: <int32> for maximum number of contexts per example.
    max_context_len: <int32> for maximum length of each context.

  Returns:
    Truncated tensors for context_token_ids, context_len and document_len
    based on max_contexts and max_context_words.
  """
  # Compute effective max contexts and max context words.
  padded_num_contexts = tf.shape(context_token_ids)[1]
  padded_context_len = tf.shape(context_token_ids)[2]

  max_contexts = tf.to_int32(tf.minimum(padded_num_contexts, max_contexts))
  max_context_len = tf.to_int32(tf.minimum(padded_context_len, max_context_len))

  # Truncate context_token_ids, context_len, and document len
  context_token_ids = context_token_ids[:, :max_contexts, :max_context_len]
  context_len = context_len[:, :max_contexts]
  context_len = tf.minimum(context_len, max_context_len)
  num_contexts = tf.minimum(num_contexts, max_contexts)

  return context_token_ids, num_contexts, context_len


def truncate_labels(context_labels, max_contexts):
  """Truncate labels based on max_contexts.

  Limit non-null labels to only be one of the first 'max_context' contexts.
  Otherwise mask it to -1 (the NULL label).

  Args:
    context_labels: <int32> [batch_size] with gold labels.
    max_contexts: <int32> for max_contexts.

  Returns:
    truncated context_labels: <int32>[batch_size]
  """
  # Null contexts are assigned label -1.
  null_labels = tf.ones_like(context_labels, dtype=tf.int32) * -1
  prune_gold_context = tf.greater_equal(context_labels, max_contexts)
  context_labels = tf.where(prune_gold_context, null_labels, context_labels)
  return context_labels


def compute_null_weights(labels, null_weight):
  has_real_context = tf.to_float(tf.reduce_any(tf.greater(labels, 0), 1))
  weights = has_real_context + (1.0 - has_real_context) * null_weight
  return weights


def compute_thresholded_labels(labels, null_threshold=4):
  """Computes thresholded labels.

  Args:
    labels: <int32> [batch_size, num_annotators]
    null_threshold: If number of null annotations is greater than or equal to
      this threshold, all annotations are set to null for this example.

  Returns:
    thresholded_labels: <int32> [batch_size, num_annotators]
  """
  null_labels = tf.equal(labels, 0)

  # <int32> [batch_size]
  null_count = tf.reduce_sum(tf.to_int32(null_labels), 1)
  threshold_mask = tf.less(null_count, null_threshold)

  # <bool> [batch_size, num_annotators]
  threshold_mask = tf.tile(
      tf.expand_dims(threshold_mask, -1), [1, tf.shape(labels)[1]])

  # <bool> [batch_size, num_annotators]
  thresholded_labels = tf.where(
      threshold_mask, x=labels, y=tf.zeros_like(labels))
  return thresholded_labels


def compute_match_stats(predictions, labels):
  """Compute statistics that are used to compute evaluation metrics.

  Args:
    predictions: <int32> [batch_size]
    labels: <int32> [batch_size, num_annotators]

  Returns:
    numerator: <float> [batch_size]
    recall_weights: <float> [batch_size]
    precision_weights: <float> [batch_size]
  """
  # <int32> [batch_size, num_labels]
  thresholded_labels = compute_thresholded_labels(labels)
  non_null_mask = tf.greater(thresholded_labels, 0)

  # <bool> [batch_size, num_labels]
  exact_match = tf.equal(tf.expand_dims(predictions, -1), thresholded_labels)
  non_null_match = tf.logical_and(exact_match, non_null_mask)

  # <float> [batch_size]
  non_null_match = tf.to_float(tf.reduce_any(non_null_match, axis=1))
  non_null_gold = tf.to_float(tf.reduce_any(non_null_mask, 1))
  non_null_predictions = tf.to_float(tf.greater(predictions, 0))

  return non_null_match, non_null_gold, non_null_predictions


def f1_metric(precision, precision_op, recall, recall_op):
  """Computes F1 based on precision and recall.

  Args:
    precision: <float> [batch_size]
    precision_op: Update op for precision.
    recall: <float> [batch_size]
    recall_op: Update op for recall.

  Returns:
    tensor and update op for F1.
  """
  f1_op = tf.group(precision_op, recall_op)
  numerator = 2 * tf.multiply(precision, recall)
  denominator = tf.add(precision, recall)
  f1 = tf.divide(numerator, denominator)

  # <float> [batch_size]
  zero_vec = tf.zeros_like(f1)
  is_valid = tf.greater(denominator, zero_vec)
  f1 = tf.where(is_valid, x=f1, y=zero_vec)

  return f1, f1_op

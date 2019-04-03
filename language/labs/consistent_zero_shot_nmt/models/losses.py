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
"""A collection of losses for NMT training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow as tf


__all__ = [
    "BaseLoss",
    "CrossEntropyLoss",
    "DistanceLoss",
    "CrossAttentionDistanceLoss",
]


@six.add_metaclass(abc.ABCMeta)
class BaseLoss(object):
  """Base class for loss functions."""

  def __init__(self, name="BaseLoss"):
    """Creates a new loss.

    Args:
      name: String used as the most outer scope name of the loss graph.
    """
    self.name = name

  def __call__(self, *args, **kwargs):
    """Builds the loss and returns <float32> [] loss tensor."""
    with tf.name_scope(self.name):
      loss = self._build(*args, **kwargs)
    return loss

  @abc.abstractmethod
  def _build(self, *args, **kwargs):
    """Must be implemented by a subclass."""
    raise NotImplementedError("Abstract Method")


class CrossEntropyLoss(BaseLoss):
  """Cross entropy loss computed on sequential inputs."""

  def __init__(self, name="CrossEntropyLoss", sparse=True):
    """Creates a new CrossEntropyLoss.

    Args:
      name: String used as the most outer scope name of the loss graph.
      sparse: Boolean indicating whether to use sparse softmax cross-entropy.
    """
    super(CrossEntropyLoss, self).__init__(name=name)
    self._sparse = sparse

  def _build(self, logits, targets, target_lens, normalize_by_length=False):
    """Builds the cross entropy loss.

    Args:
      logits: <float32> [batch_size, seq_len, vocab_size] for predicted logits.
      targets: <int32> [batch_size, seq_len] if `sparse=True` or
        <float32> [batch_size, seq_len, vocab_size] otherwise.
      target_lens: <int32> [batch_size] for the target sequence lengths.
      normalize_by_length: Boolean indicating whether to normalize the loss by
        the sequence length (i.e., shorter sequences are penalized more).

    Returns:
      loss: <float32> [batch_size] for the loss.
    """
    # Build weights.
    weights = tf.sequence_mask(target_lens, dtype=logits.dtype)

    # Build loss.
    if self._sparse:
      loss = self._build_sparse(logits, targets, weights, normalize_by_length)
    else:
      loss = self._build_dense(logits, targets, weights, normalize_by_length)

    return loss

  @staticmethod
  def _build_sparse(logits, targets, weights, normalize_by_length):
    """Build cross entropy loss assuming the targets are indices."""
    # Compute cross entropy.
    # <float32> [batch_size, max_length].
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)

    # Compute the loss.
    loss = tf.reduce_sum(cross_ent * weights, axis=1)
    if normalize_by_length:
      loss /= tf.reduce_sum(weights, axis=1)

    return loss

  @staticmethod
  def _build_dense(logits, targets, weights, normalize_by_length):
    """Build cross entropy loss assuming the targets are probabilities."""
    # Compute cross entropy.
    # <float32> [batch_size, max_length].
    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=targets, logits=logits)

    # Compute the loss.
    loss = tf.reduce_sum(cross_ent * weights, axis=1)
    if normalize_by_length:
      loss /= tf.reduce_sum(weights, axis=1)

    return loss


class DistanceLoss(BaseLoss):
  """Loss proportional to the distance between two vectors."""

  def __init__(self, dist_fn, name="DistanceLoss"):
    """Creates a new DistanceLoss.

    Args:
      dist_fn: Function that maps a pair of tensors to distances.
      name: String used as the most outer scope name of the loss graph.
    """
    super(DistanceLoss, self).__init__(name=name)
    self._dist_fn = dist_fn

  def _build(self, labels, predictions):
    """Builds a distance-based loss.

    Args:
      labels: <float32> [batch_size, dim].
      predictions: <float32> [batch_size, dim].

    Returns:
      distances: <float32> [batch_size] for the loss.
    """
    distances = self._dist_fn(labels, predictions)
    return distances


class CrossAttentionDistanceLoss(DistanceLoss):
  """Loss proportional to the mean entropy of the attention distributions.

  Computed using elements of the two sequences as keys and values, respectively.
  """

  def __init__(self, dist_fn, name="CrossAttentionDistanceLoss"):
    super(CrossAttentionDistanceLoss, self).__init__(dist_fn=dist_fn, name=name)

  def _build(self, a, b):
    # Normalize inputs.
    a_normed = tf.nn.l2_normalize(a, axis=-1)
    b_normed = tf.nn.l2_normalize(b, axis=-1)
    # <float32> [batch_size, seq_len_a, seq_len_b].
    cosine_similarity = tf.matmul(a_normed, b_normed, transpose_b=True)
    pairwise_distances = 0.5 * (1. - cosine_similarity)
    # Compute log attention distributions.
    # <float32> [batch_size, seq_len_a, seq_len_b].
    att_a_b = tf.nn.softmax(pairwise_distances, axis=2)
    # <float32> [batch_size, seq_len_b, seq_len_a].
    att_b_a = tf.transpose(tf.nn.softmax(pairwise_distances, axis=1), [0, 2, 1])
    # Compute cross-attention contexts.
    # <float32> [batch_size, seq_len_a, size].
    ctx_a_b = tf.matmul(att_a_b, b)
    # <float32> [batch_size, seq_len_b, size].
    ctx_b_a = tf.matmul(att_b_a, a)
    # Compute entropy loss.
    loss = tf.reduce_mean(
        self._dist_fn(a, ctx_a_b, reduce_axis=[1, 2]) +
        self._dist_fn(b, ctx_b_a, reduce_axis=[1, 2]))
    # loss = - tf.reduce_mean(
    #     tf.reduce_sum(log_att_input1 * tf.exp(log_att_input1), axis=[1, 2]) +
    #     tf.reduce_sum(log_att_input2 * tf.exp(log_att_input2), axis=[1, 2]))
    return loss


def l2_distance(x, y, normalize=False, reduce_axis=-1):
  if normalize:
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
  sq_diff = tf.squared_difference(x, y)
  return tf.reduce_sum(sq_diff, axis=reduce_axis)


def cosine_distance(x, y, normalize=True, reduce_axis=-1):
  if normalize:
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
  cos_diff = 0.5 * (1 - tf.multiply(x, y))
  return tf.reduce_sum(cos_diff, axis=reduce_axis)

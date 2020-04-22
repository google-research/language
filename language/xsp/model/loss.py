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
"""Loss functions."""
# TODO(alanesuhr): Rename things "positive_loss" etc. to just "loss".

import collections
import tensorflow.compat.v1 as tf

# Add an epsilon value to prevent 1 - probs from ever being 0.
EPSILON = 1e-4
# Represents losses for a batch of examples.
# losses: A tensor of shape [B, T] that contains the loss per example, per time
# step.
# total_steps: A scalar, total number of steps that we cacluate the loss for.
Loss = collections.namedtuple("Loss", ["losses", "total_steps"])


def _target_len_mask(targets, sequence_length):
  # Mask out losses that are beyond the sequence length for each examples.
  max_seq_len = tf.shape(targets)[1]
  return tf.sequence_mask(
      tf.to_int32(sequence_length), max_seq_len, dtype=tf.float32)


def _positive_example_loss_mask(is_pos, target_len_mask):
  # Expand dimension of tf.to_float(is_pos) so that it is broadcast for every
  # time step.
  return tf.expand_dims(tf.to_float(is_pos), 1) * target_len_mask


def _example_losses(logits, targets):
  """Compute the loss for examples."""
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)


def sequence_loss(logits, targets, sequence_length, weights=None):
  """Calculates the cross-entropy loss for examples.

  Args:
    logits: A float32 Tensor of logits of shape [B, T, vocab_size], where T is
      the maximum sequence length and B is the batch size.
    targets: An int32 Tensor of target classes of shape [B, T], with values in
      the range from 0 to vocab_size - 1.
    sequence_length: An int32 tensor of shape [B] corresponding
      to the length of each input
    weights: Optional Tensor of weights for each example in the batch. Should be
      shape [B].

  Returns:
    A Loss tuple.
  """
  with tf.name_scope("sequence_loss"):
    target_len_mask = _target_len_mask(targets, sequence_length)

    losses = _example_losses(logits, targets)

    if weights is not None:
      # Expand dimension of weights so that it is broadcast for every time step.
      weights = tf.expand_dims(weights, 1)
      losses *= weights

    total_steps = tf.reduce_sum(target_len_mask)

  with tf.name_scope("sequence_loss_components"):
    tf.summary.scalar("avg_loss",
                      tf.reduce_sum(losses) / total_steps)

  return Loss(losses=losses, total_steps=total_steps)

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
"""Miscellaneous utility functions for models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

_EPSILON = 1e-8  # for numerical stability


def layer_norm(layer_inputs, hidden_size):
  """Implements layer norm from [Ba et al. 2016] Layer Normalization.

  See eqn. 4 in (https://arxiv.org/pdf/1607.06450.pdf).

  Args:
    layer_inputs (tensor): The inputs to the layer.
      shape <float32>[batch_size, hidden_size]
    hidden_size (int): Dimensionality of the hidden layer.

  Returns:
    normalized (tensor): layer_inputs, normalized over all the hidden units in
      the layer.
      shape <float32>[batch_size, hidden_size]
  """

  mean, var = tf.nn.moments(layer_inputs, [1], keep_dims=True)
  with tf.variable_scope("layernorm", reuse=tf.AUTO_REUSE):
    gain = tf.get_variable(
        "gain", shape=[hidden_size], initializer=tf.constant_initializer(1))
    bias = tf.get_variable(
        "bias", shape=[hidden_size], initializer=tf.constant_initializer(0))

  normalized = gain * (layer_inputs - mean) / tf.sqrt(var + _EPSILON) + bias
  return normalized


def hamming_loss(preds, targets, sign=False):
  """Implements hamming loss.

  Args:
    preds: Tensor of predicted values.
    targets: Tensor of target values.
    sign (bool): Set to True if targets={-1, 1} to take the sign of preds
    before calculating loss.

  Returns:
    A tf.metrics tuple containing the proportion of incorrect predictions and an
    update op for the metric.
  """
  if sign:
    preds = tf.sign(preds)
  equal = tf.equal(preds, tf.cast(targets, preds.dtype))
  proportion_correct, update_op = tf.metrics.mean(tf.cast(equal, tf.float32))
  return 1 - proportion_correct, update_op


def variable_summaries(var, scope=""):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(scope):
    with tf.name_scope("summaries"):
      mean = tf.reduce_mean(var)
      tf.summary.scalar("mean", mean)
      with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar("stddev", stddev)
      tf.summary.scalar("max", tf.reduce_max(var))
      tf.summary.scalar("min", tf.reduce_min(var))
      tf.summary.histogram("histogram", var)


def write_flags_to_file(flags, filename):
  """Write hyperparameter settings in flags to a file."""
  task_flags = {
      "num_examples": flags.num_examples,
      "num_sets_per_sequence": flags.num_sets_per_sequence,
      "num_patterns_store": flags.num_patterns_store,
      "pattern_size": flags.pattern_size,
      "selective_task": flags.selective_task,
      "task_name": flags.task_name,
  }

  model_flags = {
      "model_name": flags.model_name,
      "hidden_size": flags.hidden_size,
      "learning_rate": flags.learning_rate,
      "lr_decay_step": flags.lr_decay_step,
      "max_gradient_norm": flags.max_gradient_norm,
      "batch_size": flags.batch_size,
      "num_epochs": flags.num_epochs,
      "fast_steps": flags.fast_steps,
      "fast_decay_rate": flags.fast_decay_rate,
      "fast_lr": flags.fast_lr,
      "use_oja": flags.use_oja,
      "update_mem_with_prev_timestep": flags.update_mem_with_prev_timestep,
      "learn_fast_lr": flags.learn_fast_lr,
      "learn_plasticity_coeffs": flags.learn_plasticity_coeffs,
      "fast_lr_learning_rule": flags.fast_lr_learning_rule,
      "fast_lr_activation": flags.fast_lr_activation
  }

  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  with open(filename, "w") as f:
    f.write("=== TASK PARAMS ===\n")
    f.write(json.dumps(task_flags))
    f.write("=== MODEL PARAMS ===\n")
    f.write(json.dumps(model_flags))

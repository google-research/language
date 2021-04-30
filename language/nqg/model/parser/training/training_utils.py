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
"""Utilities to define model training loop."""

from language.nqg.model.parser.training import forest_utils

import tensorflow as tf


def get_training_step(optimizer, model, verbose=False):
  """Get training step function."""
  forest_score_function = forest_utils.get_forest_score_function(
      verbose=verbose)

  def training_step(inputs):
    """Executes a step of training."""
    with tf.GradientTape() as tape:
      loss = tf.constant(0.0, dtype=tf.float32)
      application_scores_batch = model(inputs["wordpiece_ids"],
                                       inputs["num_wordpieces"],
                                       inputs["application_span_begin"],
                                       inputs["application_span_end"],
                                       inputs["application_rule_idx"])

      nu_num_nodes_batch = tf.squeeze(inputs["nu_num_nodes"], 1)
      de_num_nodes_batch = tf.squeeze(inputs["de_num_nodes"], 1)

      with tf.name_scope("forest_score"):
        # TODO(petershaw): Consider a batched implementation of
        # forest_score_function to avoid iteration over examples in the batch.
        for idx in tf.range(model.batch_size):
          application_scores = application_scores_batch[idx]

          nu_node_type = inputs["nu_node_type"][idx]
          nu_node_1_idx = inputs["nu_node_1_idx"][idx]
          nu_node_2_idx = inputs["nu_node_2_idx"][idx]
          nu_application_idx = inputs["nu_application_idx"][idx]
          nu_num_nodes = nu_num_nodes_batch[idx]

          # Log score for numerator (sum over derivations of target).
          nu_score = forest_score_function(application_scores, nu_num_nodes,
                                           nu_node_type, nu_node_1_idx,
                                           nu_node_2_idx, nu_application_idx)

          de_node_type = inputs["de_node_type"][idx]
          de_node_1_idx = inputs["de_node_1_idx"][idx]
          de_node_2_idx = inputs["de_node_2_idx"][idx]
          de_application_idx = inputs["de_application_idx"][idx]
          de_num_nodes = de_num_nodes_batch[idx]

          # Log score for denominator (partition function).
          de_score = forest_score_function(application_scores, de_num_nodes,
                                           de_node_type, de_node_1_idx,
                                           de_node_2_idx, de_application_idx)

          # -log(numerator/denominator) = log(denominator) - log(numerator)
          example_loss = de_score - nu_score
          loss += example_loss
      loss /= tf.cast(model.batch_size, dtype=tf.float32)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

  return training_step


def get_train_for_n_steps_fn(strategy, optimizer, model):
  """Return train_for_n_steps_fn."""

  training_step = get_training_step(optimizer, model)

  @tf.function
  def train_for_n_steps_fn(iterator, steps):
    mean_loss = tf.constant(0.0, dtype=tf.float32)
    for _ in tf.range(steps):
      inputs = next(iterator)
      loss = strategy.run(training_step, args=(inputs,))
      mean_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
    mean_loss /= tf.cast(steps, dtype=tf.float32)
    return mean_loss

  return train_for_n_steps_fn

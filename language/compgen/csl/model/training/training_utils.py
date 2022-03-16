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

from language.compgen.csl.model.training import forest_utils
import tensorflow as tf


def get_batch_emb_idxs(emb_idx_list, max_num_batch_emb, config):
  """Get unique embedding idxs within the batch."""
  batch_size = tf.shape(emb_idx_list)[0]
  flatten_shape = (batch_size * config["max_num_nts"] *
                   config["max_num_numerator_nodes"],)
  # <int>[num_unquie_emb_idxs], <int>[batch_size * num_nts * num_nodes]
  batch_unique_emb_idxs, batch_emb_idx_list = tf.unique(
      tf.reshape(emb_idx_list, flatten_shape))
  num_unique_emb_idxs = tf.shape(batch_unique_emb_idxs)[0]
  paddings = tf.zeros((max_num_batch_emb - num_unique_emb_idxs), dtype=tf.int32)
  # <int>[max_num_batch_emb_idxs]
  batch_emb_idxs = tf.concat((batch_unique_emb_idxs, paddings), axis=0)
  batch_emb_idxs = tf.where(batch_emb_idxs == -1, 0, batch_emb_idxs)
  batch_emb_idx_list = tf.where(
      tf.gather(batch_unique_emb_idxs, batch_emb_idx_list) == -1, -1,
      batch_emb_idx_list)
  # <int>[batch_size, num_nts, num_nodes]
  batch_emb_idx_list = tf.reshape(batch_emb_idx_list, tf.shape(emb_idx_list))
  return batch_emb_idxs, batch_emb_idx_list


def get_training_step(config, optimizer, model, verbose=False):
  """Get training step function."""
  forest_score_function = forest_utils.get_forest_score_function(
      config, verbose=verbose)
  max_num_batch_emb = config.get("max_num_batch_embs", None)
  approximate_denominator = config.get("approximate_denominator", False)

  def training_step(inputs):
    """Executes a step of training."""
    with tf.GradientTape() as tape:
      loss = tf.constant(0.0, dtype=tf.float32)

      batch_rhs_emb_idxs = None
      batch_lhs_emb_idxs = None
      batch_rhs_emb_idx_list = inputs["rhs_emb_idx_list"]
      batch_lhs_emb_idx_list = inputs["lhs_emb_idx_list"]

      if max_num_batch_emb is not None:
        (batch_rhs_emb_idxs,
         batch_rhs_emb_idx_list) = get_batch_emb_idxs(batch_rhs_emb_idx_list,
                                                      max_num_batch_emb, config)
        (batch_lhs_emb_idxs,
         batch_lhs_emb_idx_list) = get_batch_emb_idxs(batch_lhs_emb_idx_list,
                                                      max_num_batch_emb, config)
      # Compute all possible application scores once per batch.
      application_scores = model.scoring_layer.get_scores(
          rhs_emb_idxs=batch_rhs_emb_idxs,
          lhs_emb_idxs=batch_lhs_emb_idxs,
          approximate_denominator=approximate_denominator)
      num_nodes_batch = tf.squeeze(inputs["num_nodes"], 1)

      with tf.name_scope("forest_score"):
        # TODO(petershaw): Consider a batched implementation of
        # forest_score_function to avoid iteration over examples in the batch.
        for idx in tf.range(model.batch_size):
          node_type_list = inputs["node_type_list"][idx]
          node_idx_list = inputs["node_idx_list"][idx]
          rhs_emb_idx_list = batch_rhs_emb_idx_list[idx]
          lhs_emb_idx_list = batch_lhs_emb_idx_list[idx]
          num_nodes = num_nodes_batch[idx]

          # Log prob for (x,y).
          log_prob = forest_score_function(application_scores, num_nodes,
                                           node_type_list, node_idx_list,
                                           rhs_emb_idx_list, lhs_emb_idx_list)
          # Negative log likelihood.
          example_loss = -log_prob
          loss += example_loss
      loss /= tf.cast(model.batch_size, dtype=tf.float32)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

  return training_step


def get_train_for_n_steps_fn(config, strategy, optimizer, model):
  """Return train_for_n_steps_fn."""

  training_step = get_training_step(config, optimizer, model)

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

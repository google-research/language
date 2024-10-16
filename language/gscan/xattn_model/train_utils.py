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
"""Training utils."""

from absl import logging
from clu import parameter_overview

import flax
import flax.linen as nn
import flax.optim
from flax.training import common_utils
import jax
import jax.example_libraries.optimizers
import jax.numpy as jnp

from language.gscan.xattn_model import evaluation

import ml_collections
import numpy as np
import tensorflow as tf


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer


def get_init_inputs(ds):
  """Initialize model inputs."""

  def map_type(tf_type):
    if tf_type == tf.int64:
      return jnp.int32
    elif tf_type == tf.float64:
      return jnp.float32

  inputs = {
      k: jnp.ones(v.shape[1:], dtype=map_type(v.dtype))
      for k, v in ds.element_spec.items()
  }
  return inputs


def create_train_state(model, config, rng, inputs):
  """Create and initialize the model.

  Args:
    model: Flax nn module.
    config: Configuration for model.
    rng: JAX PRNG Key.
    inputs: The init inputs fed into the model.

  Returns:
    The initialized TrainState with the optimizer.
  """
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, inputs)
  params = variables['params']
  parameter_overview.log_parameter_overview(params)  # pytype: disable=wrong-arg-types

  if config.optimizer == 'AdamW':

    def w_decay_fn(path):
      return all(pn not in path for pn in config.no_weight_decay)

    def wo_decay_fn(path):
      return any(pn in path for pn in config.no_weight_decay)

    optimizer_w_decay = flax.optim.Adam(
        learning_rate=config.learning_rate,
        weight_decay=config.learning_rate_weight_decay)
    optimizer_wo_decay = flax.optim.Adam(
        learning_rate=config.learning_rate, weight_decay=0)
    params_w_decay = flax.optim.ModelParamTraversal(
        lambda path, _: w_decay_fn(path))
    params_wo_decay = flax.optim.ModelParamTraversal(
        lambda path, _: wo_decay_fn(path))
    optimizer = flax.optim.MultiOptimizer(
        (params_w_decay, optimizer_w_decay),
        (params_wo_decay, optimizer_wo_decay)).create(params)
  else:
    raise NotImplementedError
  return TrainState(step=0, optimizer=optimizer)


def get_learning_rate(step,
                      *,
                      base_learning_rate,
                      num_train_steps,
                      schedule_type='step',
                      warmup_proportion=0.1,
                      warmup_type='polynomial',
                      step_boundaries=[0.5, 0.75]):
  """Get learning rate schedule."""
  logging.info(
      'get_learning_rate(step=%s, base_learning_rate=%s, warmup_proportion=%s, num_train_steps=%s',
      step, base_learning_rate, warmup_proportion, num_train_steps)
  num_warmup_steps = int(warmup_proportion * num_train_steps)
  if schedule_type == 'step':
    boundaries = [int(b * num_train_steps) for b in step_boundaries]
    values = [
        base_learning_rate * 0.1**i for i in range(len(step_boundaries) + 1)
    ]
    lr = jax.example_libraries.optimizers.piecewise_constant(
        boundaries=boundaries, values=values)(
            step)
  else:
    raise NotImplementedError
  if num_warmup_steps:
    if warmup_type == 'polynomial':
      warmup = jnp.minimum(1, step / num_warmup_steps)
    else:
      raise NotImplementedError
    lr = lr * warmup
  return lr


def flatten_config(config, parent_key = '', sep = '_'):
  """Flattens config and remove tuple/list values for logging in tboard."""
  items = {}
  for k, v in config.items():
    new_key = sep.join(filter(None, (parent_key, k)))
    if isinstance(v, ml_collections.ConfigDict):
      items.update(flatten_config(v, new_key, sep=sep))
    elif isinstance(v, (list, tuple)):
      continue
    else:
      items[new_key] = v
  return items


def weighted_cross_entropy(logits, targets, weights=None, label_smoothing=0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, targets, weights):
  """Compute loss and evaluation metrics."""
  loss, weight_sum = weighted_cross_entropy(
      logits=logits, targets=targets, weights=weights)
  exact_match, example_sum = evaluation.exact_match(
      logits=logits, targets=targets, weights=weights)
  accuracy, _ = evaluation.accuracy(
      logits=logits, targets=targets, weights=weights)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'exact_match': exact_match,
      'weight_sum': weight_sum,
      'example_sum': example_sum
  }
  metrics = jax.lax.psum(metrics, axis_name='batch')
  return metrics


def metrics_summary(metrics, prefix):
  """Gather metrics summary."""
  metrics_sums = jax.tree.map(jnp.sum, metrics)
  weight_sum = metrics_sums.pop('weight_sum')
  example_sum = metrics_sums.pop('example_sum')
  exact_match = metrics_sums.pop('exact_match')
  summary = {
      f'{prefix}_{k}': v
      for k, v in jax.tree.map(lambda x: x / weight_sum, metrics_sums).items()
  }
  summary[f'{prefix}_exact_match'] = exact_match / example_sum
  return summary

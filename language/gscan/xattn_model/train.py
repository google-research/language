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
"""Methods for training the model using JAX."""

import functools
import json
import os

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions

import flax.jax_utils as flax_utils
from flax.training import common_utils
import jax
import jax.example_libraries.optimizers
from language.gscan.xattn_model import train_utils
from language.gscan.xattn_model.dataset import input_pipeline
from language.gscan.xattn_model.model import model_utils
from language.gscan.xattn_model.model import models

import numpy as np
import tensorflow as tf


def train_step(batch, rng, state, config, learning_rate_fn, grad_clip=None):
  """Perform a single training step."""

  logging.info('train_step(batch=%s)', batch)

  step = state.step + 1
  lr = learning_rate_fn(step)
  rng = jax.random.fold_in(rng, step)
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  rngs = {'params': params_rng, 'dropout': dropout_rng}

  def loss_fn(params):
    model = models.Model(config)
    decoder_logits = model.apply({'params': params}, batch, rngs=rngs)
    loss, normalizing_factor = train_utils.weighted_cross_entropy(
        decoder_logits,
        model_utils.shift_right(batch['target_token']),
        weights=model_utils.shift_right(batch['target_txt_mask']))
    loss /= normalizing_factor
    train_metrics = {
        'logits': decoder_logits,
        'targets': model_utils.shift_right(batch['target_token']),
        'weights': model_utils.shift_right(batch['target_txt_mask'])
    }
    return loss, train_metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, train_metrics), grad = grad_fn(state.optimizer.target)
  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name='batch')
  if grad_clip:
    grad = jax.example_libraries.optimizers.clip_grads(grad, grad_clip)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(step=step, optimizer=new_optimizer)
  metrics = train_utils.compute_metrics(**train_metrics)
  metrics['learning_rate'] = lr

  return new_state, metrics


def eval_step(batch, state, config):
  """Compute the metrics for the given model in inference mode."""

  logging.info('eval_step(batch=%s)', batch)
  variables = {'params': state.optimizer.target}
  model = models.Model(config)
  decoder_logits = model.apply(variables, batch)
  mask = model_utils.shift_right(batch['target_txt_mask'])
  mask *= batch['mask'][:, None]
  metrics = train_utils.compute_metrics(
      logits=decoder_logits,
      targets=model_utils.shift_right(batch['target_token']),
      weights=mask)
  return metrics


def evaluate(p_eval_step, state, eval_ds, num_eval_steps = -1):
  """Evaluate on the given dataset."""
  logging.info('Starting evaluating.')
  eval_metrics = []
  for step, batch in enumerate(eval_ds):
    batch = jax.tree.map(np.asarray, batch)
    metrics = p_eval_step(batch=batch, state=state)
    eval_metrics.append(metrics)
    if num_eval_steps > 0 and step + 1 == num_eval_steps:
      break
  eval_metrics = common_utils.get_metrics(eval_metrics)
  summary = train_utils.metrics_summary(eval_metrics, 'eval')
  return summary


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  logging.info('Starting training at %s', workdir)
  tf.io.gfile.makedirs(workdir)
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(os.path.join(workdir, 'config.json'), 'w') as f:
      json.dump(config.to_dict(), f, indent=2)
  rng = jax.random.PRNGKey(config.seed)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  train_ds, eval_ds = input_pipeline.create_datasets(config.dataset, data_rng)
  train_iter = iter(train_ds)

  test_ds = []
  for split in config.dataset.test_splits:
    ds = input_pipeline.create_val_dataset(
        config.dataset, split, config.dataset.test_per_device_batch_size,
        config.dataset.test_pad_last_batch)
    test_ds.append(ds)

  # Learning rate schedule.
  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    num_train_steps = train_ds.cardinality().numpy()
  steps_per_epoch = num_train_steps // config.dataset.num_epochs
  logging.info('num_train_steps=%d, steps_per_epoch=%d', num_train_steps,
               steps_per_epoch)
  learning_rate_fn = functools.partial(
      train_utils.get_learning_rate,
      base_learning_rate=config.learning_rate,
      num_train_steps=num_train_steps,
      schedule_type=config.learning_rate_schedule,
      warmup_proportion=config.warmup_proportion,
      step_boundaries=config.learning_rate_step_boundaries)

  # Initialize model.
  inputs = train_utils.get_init_inputs(train_ds)
  rng, model_rng = jax.random.split(rng)
  eval_config = models.TransformerConfig(**config.model.to_dict())
  train_config = eval_config.replace(deterministic=False)
  model = models.Model(eval_config)
  state = train_utils.create_train_state(
      model, config, model_rng, inputs=inputs)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=3)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Distribute training.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          grad_clip=config.grad_clip),
      axis_name='batch',
      donate_argnums=(0,))
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config=eval_config), axis_name='batch')

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(train_utils.flatten_config(config))

  logging.info('Starting training loop at step %d.', initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(
            num_profile_steps=config.num_profile_steps, logdir=workdir)
    ]

  rng, train_rngs = jax.random.split(rng)
  train_rngs = jax.random.fold_in(train_rngs, jax.process_index())
  train_rngs = jax.random.split(train_rngs, jax.local_device_count())

  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      is_last_step = step == num_train_steps
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = jax.tree.map(np.asarray, next(train_iter))
        state, metrics = p_train_step(batch=batch, rng=train_rngs, state=state)
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      if config.log_loss_every_steps > 0 and (step % config.log_loss_every_steps
                                              == 0 or is_last_step):
        train_metrics = common_utils.get_metrics(train_metrics)
        lr = train_metrics.pop('learning_rate').mean()
        train_summary = train_utils.metrics_summary(train_metrics, 'train')
        train_summary['learning_rate'] = lr
        writer.write_scalars(step, train_summary)
        train_metrics = []

      if config.eval_every_steps > 0 and (step % config.eval_every_steps == 0 or
                                          is_last_step):
        with report_progress.timed('eval'):
          eval_summary = evaluate(p_eval_step, state, eval_ds,
                                  config.num_eval_steps)
        writer.write_scalars(step, eval_summary)

      if config.checkpoint_every_steps > 0 and (
          step % config.checkpoint_every_steps == 0 or is_last_step):
        with report_progress.timed('checkpoint'):
          ckpt.save(flax_utils.unreplicate(state))
        logging.info('Checkpoint saved to %s', checkpoint_dir)

  logging.info('Finishing training at step %d', num_train_steps)

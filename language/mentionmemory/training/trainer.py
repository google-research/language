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
"""Contains train method and other training utilities for training loop."""

import functools
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state as ts
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import import_tasks  # pylint: disable=unused-import
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import checkpoint_utils
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils import optim_utils
import ml_collections
import optax


def eval_step(
    train_state,
    model_vars,
    batch: Dict[str, Any],
    model_config: ml_collections.FrozenConfigDict,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Calculate evaluation metrics on a batch.

  Args:
    train_state: contains model params, loss fn, grad update fn.
    model_vars: model variables that are not optimized.
    batch: input to model.
    model_config: contains model hyperparameters.

  Returns:
    Dictionary of metrics and auxiliary output.
  """

  def eval_step_partial(model_params):
    return train_state.apply_fn(
        model_config,
        model_params,
        model_vars,
        batch,
        deterministic=True,
    )

  _, metrics, auxiliary_output = eval_step_partial(train_state.params)
  metrics = metric_utils.update_metrics_dtype(metrics)
  metrics = jax.lax.psum(metrics, axis_name='batch')

  return metrics, auxiliary_output


def evaluate(
    eval_step_fn,
    train_state: ts.TrainState,
    model_vars: Dict[str, Any],
    eval_data: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Sequence[Tuple[Dict[str, Any], Optional[Dict[
    str, Any]]]]]:
  """Evaluate current parameters and return a dictionary with metrics.

  Args:
    eval_step_fn: partial eval step that takes in model params and inputs only
    train_state: contains model params, loss fn, grad update fn.
    model_vars: model variables that are not optimized.
    eval_data: sequence of evaluation data.

  Returns:
    Dictionary of metrics aggregated over all evaluation steps and the info
    for the very first batch (batch itself and corresponding auxiliary output).
  """

  logging.info('Performing evaluation.')
  eval_metrics = []
  eval_auxiliary = []
  for batch in eval_data:
    batch = jax.tree_map(jnp.asarray, batch)
    metrics, auxiliary_output = eval_step_fn(
        train_state,
        model_vars,
        batch,
    )
    eval_metrics.append(metrics)
    batch_auxiliary = (jax.device_get(batch), jax.device_get(auxiliary_output))
    eval_auxiliary.append(batch_auxiliary)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_summary = metric_utils.process_metrics(eval_metrics_sums, prefix='eval')
  return eval_summary, eval_auxiliary


def train_step(
    train_state: ts.TrainState,
    model_vars: Dict[str, Any],
    batch: Dict[str, Any],
    dropout_rng: jnp.ndarray,
    model_config: ml_collections.FrozenConfigDict,
) -> Tuple[ts.TrainState, Dict[str, Any]]:
  """Perform a single training step.

  Args:
    train_state: contains model params, loss fn, grad update fn.
    model_vars: model variables that are not optimized.
    batch: input to model.
    dropout_rng: seed for dropout rng in model.
    model_config: contains model hyperparameters.

  Returns:
    Train state with updated parameters and dictionary of metrics.
  """

  dropout_rng = jax.random.fold_in(dropout_rng, train_state.step)

  def loss_fn_partial(model_params):
    loss, metrics, _ = train_state.apply_fn(
        model_config,
        model_params,
        model_vars,
        batch,
        deterministic=False,
        dropout_rng={'dropout': dropout_rng},
    )
    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn_partial, has_aux=True)
  (_, metrics), grad = grad_fn(train_state.params)
  grad = jax.lax.pmean(grad, 'batch')
  metrics = jax.lax.psum(metrics, axis_name='batch')
  metrics = metric_utils.update_metrics_dtype(metrics)
  new_train_state = train_state.apply_gradients(grads=grad)
  return new_train_state, metrics


def train(config: ml_collections.ConfigDict):
  """Run training."""

  # Establish host information
  local_device_count = jax.local_device_count()
  host_count = jax.process_count()
  host_id = jax.process_index()

  task = task_registry.get_registered_task(config.task_name)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)

  model_config = ml_collections.FrozenConfigDict(config.model_config)
  model = task.build_model(model_config)

  # Initialization needs to be pmapped because models use collective ops.
  # Create dummy input
  dummy_input = {
      key: jnp.tile(value, (local_device_count,) + (1,) * value.ndim)
      for key, value in task.dummy_input(config).items()
  }

  rng, init_rng = jax.random.split(rng)
  init_rng = jax.random.split(init_rng, local_device_count)

  logging.info('Initializing model.')
  initial_variables = jax.pmap(
      model.init, 'batch', static_broadcasted_argnums=2)(init_rng, dummy_input,
                                                         True)
  logging.info('Finished initializing model.')
  initial_variables = flax.core.unfreeze(initial_variables)

  if config.load_weights is not None:
    logging.info('Loading model weights from file')
    loaded_variables = task.load_weights(config)
    unexpected, missing = checkpoint_utils.merge_nested_dicts(
        initial_variables, loaded_variables)
    logging.info('*** Unexpected features: ***')
    for feature_name in unexpected:
      logging.info('\t%s', feature_name)
    logging.info('*** Missing features: ***')
    for feature_name in missing:
      logging.info('\t%s', feature_name)

  model_vars = {
      key: value for key, value in initial_variables.items() if key != 'params'
  }

  learning_rate_fn = optim_utils.create_learning_rate_scheduler(
      learning_rate=config.learning_rate,
      warmup=config.warmup,
      warmup_steps=config.get('warmup_steps', None),
      linear_decay=config.linear_decay,
      max_steps=config.num_train_steps,
      decay_minimum_factor=config.get('decay_minimum_factor', None),
  )

  if config.weight_decay_exclude is not None:
    decay_mask = optim_utils.create_dict_mask(initial_variables['params'],
                                              config.weight_decay_exclude)
  else:
    decay_mask = None
  tx = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=learning_rate_fn,
      weight_decay=config.weight_decay,
      b1=0.9,
      b2=0.999,
      eps=1e-6,
      mask=decay_mask)
  if config.grad_clip is not None:
    tx = optax.chain(tx, optax.clip_by_global_norm(config.grad_clip))

  ignore_k_nans = config.get('ignore_k_nans')
  if ignore_k_nans is not None:
    tx = optax.apply_if_finite(tx, max_consecutive_errors=ignore_k_nans)

  loss_fn = task.make_loss_fn(config)
  train_state = ts.TrainState.create(
      apply_fn=loss_fn,
      params=jax_utils.unreplicate(initial_variables['params']),
      tx=tx,
  )

  # We access model params only from train state.
  del initial_variables

  # Restore unreplicated train state from last checkpoint
  try:
    train_state = checkpoints.restore_checkpoint(config.model_dir, train_state)
  except ValueError:
    pass
  # Grab last step.
  start_step = int(train_state.step)

  writer = metric_writers.create_default_writer(
      config.model_dir, just_logging=jax.process_index() > 0)
  if start_step == 0:
    writer.write_hparams(config.to_dict())

  dropout_rngs = jax.random.split(rng, local_device_count)

  del rng

  # Load datasets
  logging.info('Loading dataset.')

  # Make sure we don't re-use same data if we load weights or checkpoint
  seed = config.seed + start_step
  if config.load_weights:
    seed = seed + hash(config.load_weights)

  name_to_features = task.get_name_to_features(config)
  preprocess_fn = task.make_preprocess_fn(config)
  collater_fn = task.make_collater_fn(config)

  train_data = data_utils.load_multi_dataset(
      datasets_config=config.train_data,
      name_to_features=name_to_features,
      preprocess_fn=preprocess_fn,
      collater_fn=collater_fn,
      is_training=True,
      per_device_batch_size=config.per_device_batch_size,
      local_device_count=local_device_count,
      host_count=host_count,
      host_id=host_id,
      seed=config.seed,
  )
  train_iter = iter(train_data)

  pad_eval = config.get('pad_eval', False)
  if pad_eval:
    logging.info('Eval data is padded such that none of samples are dropped.')
  else:
    logging.warn('Eval data is NOT padded -- some samples might be dropped.')

  eval_data = data_utils.load_multi_dataset(
      datasets_config=config.eval_data,
      name_to_features=name_to_features,
      preprocess_fn=preprocess_fn,
      collater_fn=collater_fn,
      is_training=False,
      per_device_batch_size=config.per_device_batch_size,
      local_device_count=local_device_count,
      host_count=host_count,
      host_id=host_id,
      seed=config.seed,
      pad_eval=pad_eval,
  )
  eval_data = list(eval_data)
  logging.info('Loaded %d samples for evaluation.', len(eval_data))

  # Setup postprocessing_fn for saving samples occasionally.
  if config.get('save_samples_every_steps') is not None:
    if config.get('save_samples_every_steps') % config.eval_every_steps != 0:
      raise ValueError(
          '`eval_every_steps` must divide `save_samples_every_steps`.')
    postprocessing_fn = task.make_output_postprocess_fn(config)

  # Training loop
  logging.info('Starting training.')

  # Replicate train state.
  train_state = jax_utils.replicate(train_state)

  # compile multidevice versions of train/eval/predict step
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model_config=model_config,
      ),
      axis_name='batch',
      donate_argnums=(0,),
  )  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          model_config=model_config,
      ),
      axis_name='batch')

  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)

  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=config.model_dir, num_profile_steps=5)
    ]
  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      # Shard data to devices and perform a training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = jax.tree_map(jnp.asarray, train_iter.get_next())
        train_state, metrics = p_train_step(
            train_state,
            model_vars,
            batch,
            dropout_rngs,
        )
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d', 5, step)
      for h in hooks:
        h(step)

        # Periodic metric handling.
      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          train_metrics = common_utils.get_metrics(train_metrics)
          metrics_sums = jax.tree_map(jnp.sum, train_metrics)
          summary = metric_utils.process_metrics(metrics_sums, prefix='train')
          summary['learning_rate'] = learning_rate_fn(step)

          writer.write_scalars(step, summary)
          train_metrics = []

          with report_progress.timed('eval'):
            eval_results, eval_auxiliary = evaluate(
                eval_step_fn=p_eval_step,
                train_state=train_state,
                model_vars=model_vars,
                eval_data=eval_data,
            )
            writer.write_scalars(step, eval_results)

            if config.get('save_samples_every_steps') is not None:
              with report_progress.timed('save_samples'):
                if config.get('save_first_batch_only', 'True'):
                  postprocessing_input = [eval_auxiliary[0]]
                eval_processed = [
                    postprocessing_fn(batch, auxiliary_output)
                    for batch, auxiliary_output in eval_auxiliary
                ]
                data_utils.save_samples_to_json(eval_processed, config, step)

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step % config.checkpoint_every_steps == 0 or is_last_step)
      if (config.save_checkpoints and save_checkpoint and
          jax.process_index() == 0):
        with report_progress.timed('checkpoint'):
          logging.info('Saving checkpoint at step %s', step)
          checkpoints.save_checkpoint(
              config.model_dir,
              jax_utils.unreplicate(train_state),
              step,
              keep=config.get('keep_checkpoints', 1),
              keep_every_n_steps=config.get('keep_checkpoint_every_steps'),
          )

      save_model = (
          config.save_every_steps and
          (step % config.save_every_steps == 0 or is_last_step) and step != 0)
      if (save_model and jax.process_index() == 0):
        with report_progress.timed('checkpoint'):
          logging.info('Saving weights at step %s', step)
          save_path = os.path.join(config.model_dir, 'weights',
                                   'step' + str(step))
          # By default, save only encoder weights
          weights = jax_utils.unreplicate(train_state).params['encoder']
          checkpoint_utils.save_weights(save_path, weights)

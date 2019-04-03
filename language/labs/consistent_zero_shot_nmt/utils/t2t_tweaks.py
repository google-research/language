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
"""Monkey-patching of certain tesor2tensor functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from language.labs.consistent_zero_shot_nmt.data_generators import translate_multilingual
import numpy as np

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer("train_data_size", -1, "Limits training data size.")


def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary,
                           batch_size, max_input_size, task_id=-1):
  """Generator to produce batches of inputs."""
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_target_tags = []
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      target_tag = inputs[:translate_multilingual.LANG_TAG_LENGTH]
      target_tag_id = translate_multilingual.get_tag_id(target_tag)
      batch_target_tags.append([target_tag_id])
      # Add +1 to drop the space character right after the tag.
      input_seq = inputs[(translate_multilingual.LANG_TAG_LENGTH + 1):]
      input_ids = vocabulary.encode(input_seq)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      final_id = text_encoder.EOS_ID if task_id < 0 else task_id
      input_ids.append(final_id)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
        "target_tags": np.array(batch_target_tags).astype(np.int32),
    }


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters

  Returns:
    A features dictionary, as expected by the decoder.
  """
  x = tf.convert_to_tensor(feature_map["inputs"])
  t = tf.convert_to_tensor(feature_map["target_tags"])

  # Add a third empty dimension.
  x = tf.to_int32(tf.expand_dims(x, axis=[2]))
  t = tf.to_int32(tf.expand_dims(t, axis=[2]))

  p_hparams = hparams.problem_hparams
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = tf.shape(x)[1] + 50
  features["inputs"] = x
  features["target_tags"] = t

  return features


# Monkey-patch batch input_fn in T2T decoding.
decoding._decode_batch_input_fn = _decode_batch_input_fn  # pylint: disable=protected-access
decoding._decode_input_tensor_to_features_dict = _decode_input_tensor_to_features_dict  # pylint: disable=protected-access,line-too-long


def create_experiment(
    run_config,
    hparams,
    model_name,
    problem_name,
    data_dir,
    train_steps,
    eval_steps,
    min_eval_frequency=2000,
    eval_throttle_seconds=600,
    schedule="train_and_evaluate",
    export=False,
    decode_hparams=None,
    use_tfdbg=False,
    use_dbgprofile=False,
    eval_early_stopping_steps=None,
    eval_early_stopping_metric=None,
    eval_early_stopping_metric_delta=None,
    eval_early_stopping_metric_minimize=True,
    eval_timeout_mins=240,
    use_tpu=False,
    use_tpu_estimator=False,
    use_xla=False,
    additional_train_hooks=None,
    additional_eval_hooks=None,
    warm_start_from=None,
    decode_from_file=None,
    decode_to_file=None,
    decode_reference=None,
    std_server_protocol=None):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", schedule)
  hparams.add_hparam("warm_start_from", warm_start_from)
  hparams.add_hparam("std_server_protocol", std_server_protocol)
  hparams.add_hparam("eval_freq_in_steps", min_eval_frequency)
  hparams.add_hparam("eval_timeout_mins", eval_timeout_mins)
  if decode_hparams is not None:
    decode_hparams.add_hparam("decode_from_file", decode_from_file)
    decode_hparams.add_hparam("decode_to_file", decode_to_file)
    decode_hparams.add_hparam("decode_reference", decode_reference)
  trainer_lib.add_problem_hparams(hparams, problem_name)

  # Estimator
  estimator = trainer_lib.create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu,
      use_tpu_estimator=use_tpu_estimator,
      use_xla=use_xla)

  # Input fns from Problem
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams,
      dataset_kwargs={"max_records": FLAGS.train_data_size})
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Export
  exporter = None
  if export:
    def compare_fn(best_eval_result, current_eval_result):
      metric = eval_early_stopping_metric or "loss"
      return current_eval_result[metric] < best_eval_result[metric]

    exporter = tf.estimator.BestExporter(
        name="best",
        serving_input_receiver_fn=lambda: problem.serving_input_fn(hparams),
        compare_fn=compare_fn,
        assets_extra=problem.export_assets)

  # Hooks
  validation_monitor_kwargs = dict(
      input_fn=eval_input_fn,
      eval_steps=eval_steps,
      every_n_steps=min_eval_frequency,
      early_stopping_rounds=eval_early_stopping_steps,
      early_stopping_metric=eval_early_stopping_metric,
      early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
  dbgprofile_kwargs = {"output_dir": run_config.model_dir}
  early_stopping_kwargs = dict(
      events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
      tag=eval_early_stopping_metric,
      num_plateau_steps=eval_early_stopping_steps,
      plateau_decrease=eval_early_stopping_metric_minimize,
      plateau_delta=eval_early_stopping_metric_delta,
      every_n_steps=min_eval_frequency)

  # Eval on TPU Pods is not supported yet
  if use_tpu and run_config.tpu_config.num_shards > 8 and "eval" in schedule:
    raise ValueError("Eval is not currently supported on a TPU Pod")

  # In-process eval (and possible early stopping)
  if schedule == "continuous_train_and_eval" and min_eval_frequency:
    tf.logging.warn("ValidationMonitor only works with "
                    "--schedule=train_and_evaluate")
  use_validation_monitor = (
      schedule == "train_and_evaluate" and min_eval_frequency)
  # Distributed early stopping
  local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
  use_early_stopping = (
      schedule not in local_schedules and eval_early_stopping_steps)
  train_hooks, eval_hooks = trainer_lib.create_hooks(
      use_tfdbg=use_tfdbg,
      use_dbgprofile=use_dbgprofile,
      dbgprofile_kwargs=dbgprofile_kwargs,
      use_validation_monitor=use_validation_monitor,
      validation_monitor_kwargs=validation_monitor_kwargs,
      use_early_stopping=use_early_stopping,
      early_stopping_kwargs=early_stopping_kwargs)

  hook_context = trainer_lib.HookContext(
      estimator=estimator, problem=problem, hparams=hparams)

  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name, hook_context)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name, hook_context)
  if additional_train_hooks:
    train_hooks += additional_train_hooks
  if additional_eval_hooks:
    eval_hooks += additional_eval_hooks

  train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      train_hooks, estimator)
  eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      eval_hooks, estimator)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      throttle_secs=eval_throttle_seconds,
      exporters=exporter)

  return trainer_lib.T2TExperiment(estimator, hparams, train_spec, eval_spec,
                                   use_validation_monitor, decode_hparams)

# Monkey-patch ceate_experiment in T2T trainer_lib.
trainer_lib.create_experiment = create_experiment

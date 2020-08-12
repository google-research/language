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
"""Experiment utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import flags

import tensorflow.compat.v1 as tf

flags.DEFINE_integer("batch_size", 16, "Batch size.")

flags.DEFINE_integer("eval_batch_size", 16,
                     "Batch size for evaluation. Only used on TPU.")

flags.DEFINE_string("model_dir", None, "Model directory")

flags.DEFINE_integer("tf_random_seed", None,
                     "Random seed for tensorflow")

flags.DEFINE_integer("num_eval_steps", None,
                     "Number of steps to take during evaluation.")

flags.DEFINE_integer("num_train_steps", None,
                     "Number of steps to take during training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "Number of steps between checkpoint saves.")

flags.DEFINE_integer("eval_throttle_secs", 600,
                     "Minimum number of seconds to wait between evaluations")

flags.DEFINE_integer("eval_start_delay_secs", 120,
                     "Number of seconds to wait before starting evaluations.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "Max number of checkpoints to keep")

flags.DEFINE_bool("use_tpu", None, "Use TPU model.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

FLAGS = flags.FLAGS


def run_experiment(model_fn,
                   train_input_fn,
                   eval_input_fn,
                   exporters=None,
                   params=None,
                   params_fname=None):
  """Run an experiment using estimators.

  This is a light wrapper around typical estimator usage to avoid boilerplate
  code. Please use the following components separately for more complex
  usages.

  Args:
    model_fn: A model function to be passed to the estimator. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args_1
    train_input_fn: An input function to be passed to the estimator that
      corresponds to the training data. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train
    eval_input_fn: An input function to be passed to the estimator that
      corresponds to the held-out eval data. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate
    exporters: (Optional) An tf.estimator.Exporter or a list of them.
    params: (Optional) A dictionary of parameters that will be accessible by the
      model_fn and input_fns. The 'batch_size' and 'use_tpu' values will be set
      automatically.
    params_fname: (Optional) If specified, `params` will be written to here
      under `FLAGS.model_dir` in JSON format.
  """
  params = params if params is not None else {}
  params.setdefault("use_tpu", FLAGS.use_tpu)

  if FLAGS.model_dir and params_fname:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    params_path = os.path.join(FLAGS.model_dir, params_fname)
    with tf.io.gfile.GFile(params_path, "w") as params_file:
      json.dump(params, params_file, indent=2, sort_keys=True)

  if params["use_tpu"]:
    if FLAGS.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      tpu_cluster_resolver = None
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        tf_random_seed=FLAGS.tf_random_seed,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.save_checkpoints_steps))
    if "batch_size" in params:
      # Let the TPUEstimator fill in the batch size.
      params.pop("batch_size")
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        params=params,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)
  else:
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        tf_random_seed=FLAGS.tf_random_seed,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    params["batch_size"] = FLAGS.batch_size
    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model_fn,
        params=params,
        model_dir=FLAGS.model_dir)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=FLAGS.num_train_steps)
  eval_spec = tf.estimator.EvalSpec(
      name="default",
      input_fn=eval_input_fn,
      exporters=exporters,
      start_delay_secs=FLAGS.eval_start_delay_secs,
      throttle_secs=FLAGS.eval_throttle_secs,
      steps=FLAGS.num_eval_steps)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.estimator.train_and_evaluate(
      estimator=estimator,
      train_spec=train_spec,
      eval_spec=eval_spec)

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

import copy
import functools
import json
import multiprocessing
import os
import tempfile
import time
import uuid

from absl import flags
from language.capwap.utils import io_utils
import tensorflow.compat.v1 as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "Mode the job is being run in.")

flags.DEFINE_string("model_dir", None, "Model directory.")

flags.DEFINE_integer("batch_size", 32, "Batch size for training set.")

flags.DEFINE_integer("eval_batch_size", 32, "Batch size for evaluation set.")

flags.DEFINE_integer("predict_batch_size", 32, "Batch size for prediction.")

flags.DEFINE_boolean("mix_batches", True,
                     "Combine multiple dataset samples in the same batch.")

flags.DEFINE_string("metric", "loss", "Metric for comparing eval checkpoints.")

flags.DEFINE_float("metric_sign", 1.0,
                   "1.0 for maximizing, -1.0 for minimizing the eval metric.")

flags.DEFINE_integer("tf_random_seed", 1234, "Random seed for tensorflow")

flags.DEFINE_integer("num_eval_steps", 100,
                     "Number of steps to take during evaluation.")

flags.DEFINE_integer("num_train_steps", 100000,
                     "Number of steps to take during training.")

flags.DEFINE_integer("num_warmup_steps", 10000,
                     "Number of linear learning rate warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "Number of steps between checkpoint saves.")

flags.DEFINE_string("session_master", None, "Tensorflow master address.")

flags.DEFINE_boolean("use_tpu", False, "Use TPU model.")

flags.DEFINE_string("tpu_job_name", "tpu_worker", "Name of tpu worker.")

flags.DEFINE_integer("tpu_iterations_per_loop", None,
                     "TPU batch iterations per loop.")

flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "Max number of checkpoints to keep")

flags.DEFINE_string("warm_start_path", None,
                    "Pre-trained checkpoint to load to start fine-tuning.")

flags.DEFINE_integer("num_input_threads", tf.data.experimental.AUTOTUNE,
                     "Parallel calls for data loading.")

flags.DEFINE_integer("prefetch_batches", 5,
                     "Number of batches to prefetch during data loading.")

flags.DEFINE_float("learning_rate", 2e-5, "Optimizer learning rate.")

flags.DEFINE_string("tempdir", None, "Directory for temporary files.")


def get_tempdir():
  """Return tempdir."""
  dirname = os.path.join(tempfile.gettempdir(), "capwap")
  if FLAGS.tempdir:
    dirname = FLAGS.tempdir
  elif os.getenv("CAPWAP_TMP"):
    dirname = os.getenv("CAPWAP_TMP")


  tf.io.gfile.makedirs(dirname)
  return dirname


def get_tempfile():
  """Return tempfile."""
  dirname = get_tempdir()
  filename = uuid.uuid4().hex
  return os.path.join(dirname, filename)


def get_model_dir():
  """Returns model directory specified in either FLAGS or TF_CONFIG."""
  model_dir = None
  if FLAGS.model_dir:
    model_dir = FLAGS.model_dir
  elif os.getenv("TF_CONFIG"):
    tf_config = json.loads(os.getenv("TF_CONFIG"))
    model_dir = tf_config.get("model_dir")
  if not model_dir:
    tmp_dir = os.path.join(get_tempdir(), "adhoc_models")
    model_dir = os.path.join(tmp_dir, uuid.uuid4().hex)

  return model_dir


def save_params(params):
  """Serialize params to model directory."""
  model_dir = get_model_dir()
  filename = os.path.join(model_dir, "params.json")
  tf.logging.info("Saving params to %s", filename)
  tf.io.gfile.makedirs(model_dir)
  with tf.io.gfile.GFile(filename, "w") as f:
    json.dump(params, f, cls=io_utils.NumpyEncoder, indent=2, sort_keys=True)


def get_estimator(model_fn, params):
  """Gets an instance of the TPU estimator.

  Args:
    model_fn: tf Estimator model function.
    params: Parameter dictionary for estimator.

  Returns:
    tf.estimator.tpu.TPUEstimator: tf.estimator.tpu.TPUEstimator instance.
  """
  # Validate settings for TPU inner loop.
  assert (FLAGS.tpu_iterations_per_loop or FLAGS.save_checkpoints_steps)
  tpu_iterations_per_loop = FLAGS.tpu_iterations_per_loop
  if not tpu_iterations_per_loop:
    tpu_iterations_per_loop = FLAGS.save_checkpoints_steps
  if FLAGS.save_checkpoints_steps % tpu_iterations_per_loop != 0:
    raise ValueError(
        "TPU iterations per loop should evenly divide checkpointing.")

  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=tpu_iterations_per_loop,
      tpu_job_name=FLAGS.tpu_job_name)

  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.session_master,
      model_dir=params.get("model_dir") or get_model_dir(),
      tf_random_seed=FLAGS.tf_random_seed,
      tpu_config=tpu_config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)

  # Ugly removal from params. These need to be dynamically set by TPUEstimator.
  params = copy.copy(params)
  batch_size = params.pop("batch_size")
  eval_batch_size = params.pop("eval_batch_size")
  predict_batch_size = params.pop("predict_batch_size")
  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=params["use_tpu"],
      config=run_config,
      params=params,
      train_batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size=predict_batch_size)

  return estimator


def checkpoints_iterator(model_dir):
  """Iterate all checkpoints in model directory in order they are created."""
  visited = set()
  while True:
    tf.logging.info("Waiting for new checkpoints at %s", model_dir)
    while True:
      ckpt_state = tf.train.get_checkpoint_state(model_dir)
      if not ckpt_state:
        time.sleep(1)
        continue
      ckpt_paths = ckpt_state.all_model_checkpoint_paths
      new_ckpts = [p for p in ckpt_paths if p not in visited]
      if new_ckpts:
        break
    new_ckpts = sorted(
        new_ckpts, key=lambda x: int(os.path.basename(x).split("-")[1]))
    tf.logging.info("Found new checkpoint at %s", new_ckpts[0])
    yield new_ckpts[0]
    visited.add(new_ckpts[0])


def evaluate_checkpoints(
    estimator,
    caption_input_fn=None,
    caption_eval_fn=None,
    vqa_input_fn=None,
    vqa_eval_fn=None,
    max_checkpoint_number=None,
):
  """Continuously loops while checking for checkpoints to evaluate.

  Args:
    estimator: Instance of Estimator.
    caption_input_fn: Estimator input function for supervised captioning.
    caption_eval_fn: Evaluation callback on captioning output.
    vqa_input_fn: Estimator input function for visual qa.
    vqa_eval_fn: Evaluation callback on visual qa output.
    max_checkpoint_number: Stop after this checkpoint is evaluated.
  """

  def _compare_metrics(x, y):
    """Compare metrics of result x to result y."""
    metric_x = FLAGS.metric_sign * x.get(FLAGS.metric, 0)
    metric_y = FLAGS.metric_sign * y.get(FLAGS.metric, 0)
    tf.logging.info("Metric: %s\tNew: %2.4f\tOld: %2.4f", FLAGS.metric,
                    metric_x, metric_y)
    return metric_x > metric_y

  # Store best
  best_results = None
  best_path = os.path.join(estimator.model_dir, "model.ckpt-best")

  # Init async threads.
  threads = multiprocessing.pool.ThreadPool(2)

  # Reload best from disk.
  if tf.io.gfile.exists(best_path + ".eval"):
    with tf.io.gfile.GFile(best_path + ".eval", "r") as f:
      best_results = json.load(f)

  # Summary writer
  summaries = tf.summary.FileWriter(os.path.join(estimator.model_dir, "eval"))

  # Track evaluated checkpoints.
  ckpt_manager = os.path.join(estimator.model_dir, "evaluated_ckpts.json")
  tf.io.gfile.makedirs(estimator.model_dir)
  if tf.io.gfile.exists(ckpt_manager):
    with tf.io.gfile.GFile(ckpt_manager, "r") as f:
      visited = set(json.load(f))
  else:
    visited = set()

  # Iterate over checkpoint files as they appear.
  for ckpt in checkpoints_iterator(estimator.model_dir):
    current_step = int(os.path.basename(ckpt).split("-")[1])

    # Skip if already evaluated.
    if ckpt in visited:
      continue

    tf.logging.info("***** Starting evaluation *****")
    try:
      # Initialize resuls.
      results = dict(step=current_step)

      # Evaluate predictions concurrently.
      async_results = []

      # Generate captions.
      if caption_input_fn is not None:
        tf.logging.info("Writing captions predictions to disk...")
        with tf.io.gfile.GFile(ckpt + ".caption_pred", "w") as f:
          iterator = estimator.predict(
              input_fn=functools.partial(
                  caption_input_fn, mode=tf.estimator.ModeKeys.PREDICT),
              checkpoint_path=ckpt,
              yield_single_examples=True)
          i = 0
          for ex in iterator:
            i += 1
            f.write(json.dumps(ex, cls=io_utils.NumpyEncoder) + "\n")
            if i % 1000 == 0:
              tf.logging.info("Wrote %d predictions", i)
          tf.logging.info("Done. Wrote %d predictions.", i)

      # Evaluate captions.
      if caption_eval_fn is not None:
        async_results.append(
            threads.apply_async(caption_eval_fn, (ckpt + ".caption_pred",)))

      # Generate predictions for questions.
      if vqa_input_fn is not None:
        tf.logging.info("Writing question predictions to disk...")
        with tf.io.gfile.GFile(ckpt + ".question_pred", "w") as f:
          iterator = estimator.predict(
              input_fn=functools.partial(
                  vqa_input_fn, mode=tf.estimator.ModeKeys.PREDICT),
              checkpoint_path=ckpt,
              yield_single_examples=True)
          i = 0
          for ex in iterator:
            i += 1
            # Take top-1 caption.
            ex["token_ids"] = ex["token_ids"][:1]
            f.write(json.dumps(ex, cls=io_utils.NumpyEncoder) + "\n")
            if i % 1000 == 0:
              tf.logging.info("Wrote %d predictions", i)
          tf.logging.info("Done. Wrote %d predictions.", i)

      # Evaluate questions.
      if vqa_eval_fn is not None:
        async_results.append(
            threads.apply_async(vqa_eval_fn, (ckpt + ".question_pred",)))

      # Wait for results.
      for res in async_results:
        results.update(res.get())
      tf.logging.info(results)

      # Log all results.
      for k, v in results.items():
        summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
        summaries.add_summary(summary, current_step)

      # Save results.
      with tf.io.gfile.GFile(ckpt + ".eval", "w") as f:
        json.dump(
            results, f, cls=io_utils.NumpyEncoder, indent=2, sort_keys=True)

      # Mark checkpoint.
      visited.add(ckpt)
      with tf.io.gfile.GFile(ckpt_manager, "w") as f:
        json.dump(list(visited), f)

      # Maybe update best copy.
      if best_results is None or _compare_metrics(results, best_results):
        tf.logging.info("New best evaluation.")
        best_results = results
        for fname in tf.io.gfile.glob(ckpt + "*"):
          ext = os.path.splitext(fname)[-1]
          tf.logging.info("Copying %s to best...", fname)
          tf.io.gfile.copy(fname, best_path + ext, overwrite=True)

      is_final = max_checkpoint_number and current_step >= max_checkpoint_number
      if is_final:
        tf.logging.info("Evaluation finished after checkpoint step number %d.",
                        current_step)
        tf.logging.info("Terminating evaluation.")
        summaries.close()
        break

    except tf.errors.NotFoundError:
      tf.logging.info("Checkpoint %s no longer exists, skipping.", ckpt)

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
# Lint as: python3
"""Util library for saving, copying and moving checkpoints."""

import concurrent.futures
import os


import tensorflow.compat.v1 as tf

# Default target file name to copy the best checkpoint to.
BEST_CHECKPOINT_FILENAME = "best_checkpoint"

# Default file name for storing the best evaluation results.
BEST_EVAL_INFO_FILENAME = "best_eval_info"


def copy_checkpoint(checkpoint_path,
                    to_dir,
                    to_checkpoint_name = BEST_CHECKPOINT_FILENAME):
  """Copies a checkpoint to a new directory.

  Args:
    checkpoint_path: Specific checkpoint path to copy.
    to_dir: The target directory to copy to.
    to_checkpoint_name: The target checkpoint name to copy to.

  Raises:
    NotFoundError: When the given checkpoint is not found.
  """
  if not checkpoint_path:
    raise tf.errors.NotFoundError(None, None,
                                  "Checkpoint path must be non-empty")
  old_filenames = tf.io.gfile.glob(checkpoint_path + "*")
  if not old_filenames:
    raise tf.errors.NotFoundError(
        None, None, "Unable to find checkpoint: %s" % checkpoint_path)

  if not tf.io.gfile.exists(to_dir):
    tf.io.gfile.makedirs(to_dir)

  # Threaded copying helps to mitigate issues where large checkpoints do not
  # finish copying before being deleted.
  threads = []
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
  for old_filename in old_filenames:
    _, suffix = os.path.splitext(old_filename)
    new_filename = os.path.join(to_dir, to_checkpoint_name + suffix)
    threads.append(
        executor.submit(
            tf.io.gfile.copy, old_filename, new_filename, overwrite=True))
  concurrent.futures.wait(threads)
  tf.logging.info("Copied checkpoint %s to dir %s", checkpoint_path, to_dir)

  # Recreates a checkpoint file.
  new_checkpoint = os.path.join(to_dir, to_checkpoint_name)
  tf.train.update_checkpoint_state(to_dir, new_checkpoint)
  tf.logging.info("Writing new checkpoint file for %s", to_dir)


def update_eval_info(
    directory,
    eval_result,
    higher_is_better = True,
    eval_info_filename = BEST_EVAL_INFO_FILENAME):
  """Updates the eval info if the new result is better.

  Args:
    directory: The directory where the best eval info file is stored.
    eval_result: The new eval result.
    higher_is_better: Whether higher eval numbers are better.
    eval_info_filename: The name of the best eval file.

  Returns:
    Whether the new eval result is better than the previous one.
  """
  # Read the previous eval number and compare it to the current one.
  full_path = os.path.join(directory, eval_info_filename)
  if not tf.io.gfile.exists(full_path):
    is_better = True
  else:
    with tf.io.gfile.GFile(full_path, "r") as eval_file:
      previous_eval_result_str = eval_file.read()
    try:
      previous_eval_result = float(previous_eval_result_str)
      if higher_is_better:
        is_better = eval_result > previous_eval_result
      else:
        is_better = eval_result < previous_eval_result
    except ValueError:
      is_better = True
      tf.logging.info("Skip previous eval info because it is ill-formed.")

  if is_better:
    if not tf.io.gfile.exists(directory):
      tf.io.gfile.makedirs(directory)
    with tf.io.gfile.GFile(full_path, "w") as eval_file:
      eval_file.write("%f\n" % eval_result)

  return is_better


def save_checkpoint_if_best(eval_result,
                            checkpoint_path,
                            to_dir,
                            to_checkpoint_name = BEST_CHECKPOINT_FILENAME,
                            higher_is_better = True):
  """Copies a checkpoint if it is the best so far.

  Args:
    eval_result: The new eval result.
    checkpoint_path: Specific checkpoint path to compare and copy.
    to_dir: The target directory to copy to.
    to_checkpoint_name: The target checkpoint name to copy to.
    higher_is_better: Whether higher eval numbers are better.

  Returns:
    Whether the new eval result is better than the previous one.
  """
  is_better = update_eval_info(to_dir, eval_result, higher_is_better)

  if is_better:
    copy_checkpoint(checkpoint_path, to_dir, to_checkpoint_name)

  return is_better

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
"""Helps with warm starting the right variables from a pre-training checkpoint."""
import collections
import os
import re
import tensorflow.compat.v1 as tf


def _get_assignment_map_from_checkpoint(tvars, init_checkpoint,
                                        reinitialize_type_embeddings):
  """Compute the intersection of the current variables and checkpoint variables."""
  initialized_variable_names = {}

  name_to_variable = {}
  for var in tvars:
    name = var.name

    # This does not load the token type embeddings from BERT, and will force
    # re-initialization.
    if reinitialize_type_embeddings and "token_type_embeddings" in name:
      continue

    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def init_model_from_checkpoint(checkpoint_dir,
                               use_tpu=False,
                               checkpoint_file=None,
                               reinitialize_type_embeddings=False):
  """Initializes whitelisted parameters from pretrained checkpoint dir.

  Args:
    checkpoint_dir: Path to the checkpoint dir.
    use_tpu: Whether to use TPU to train.
    checkpoint_file: Name of the checkpoint file.
    reinitialize_type_embeddings: Whether to re-initialize the type embeddings
      used in the BERT model.

  Returns:
    Dictionary of whitelisted pretrained parameter names if warm_start_whitelist
    is set and scaffold_fn if use tpu.
  """
  if checkpoint_file:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
  else:
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  (assignment_map,
   initialized_variable_names) = _get_assignment_map_from_checkpoint(
       tf.trainable_variables(), checkpoint_path, reinitialize_type_embeddings)
  tf.logging.info("Pretrained parameter assignment_map: %s", assignment_map)
  scaffold_fn = None
  # We have to pass scaffold_fn to TPUEstimatorSpec for initing checkpoint from
  # Bert otherwise it will fail to init TPU system.
  if use_tpu:

    def tpu_scaffold():
      tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
      return tf.train.Scaffold()

    scaffold_fn = tpu_scaffold
  else:
    tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
  return initialized_variable_names, scaffold_fn

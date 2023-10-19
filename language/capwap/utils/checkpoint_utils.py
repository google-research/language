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
"""Utilities for dealing with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow.compat.v1 as tf


def log_variables(name, var_names):
  tf.logging.info("%s (%d total): %s", name, len(var_names),
                  random.sample(list(var_names), min(len(var_names), 5)))


def init_from_checkpoint(checkpoint_path,
                         checkpoint_prefix=None,
                         variable_prefix=None,
                         target_variables=None):
  """Initializes all of the variables using `init_checkpoint."""
  tf.logging.info("Loading variables from %s", checkpoint_path)
  checkpoint_variables = {
      name: name for name, _ in tf.train.list_variables(checkpoint_path)
  }
  if target_variables is None:
    target_variables = tf.trainable_variables()
  target_variables = {var.name.split(":")[0]: var for var in target_variables}

  if checkpoint_prefix is not None:
    checkpoint_variables = {
        checkpoint_prefix + "/" + name: varname
        for name, varname in checkpoint_variables.items()
    }
  if variable_prefix is not None:
    target_variables = {
        variable_prefix + "/" + name: var
        for name, var in target_variables.items()
    }

  checkpoint_var_names = set(checkpoint_variables.keys())
  target_var_names = set(target_variables.keys())
  intersected_var_names = target_var_names & checkpoint_var_names
  assignment_map = {
      checkpoint_variables[name]: target_variables[name]
      for name in intersected_var_names
  }
  tf.train.init_from_checkpoint(checkpoint_path, assignment_map)

  log_variables("Loaded variables", intersected_var_names)
  log_variables("Uninitialized variables",
                target_var_names - checkpoint_var_names)
  log_variables("Unused variables", checkpoint_var_names - target_var_names)

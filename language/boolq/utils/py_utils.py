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
"""Some non-tensorflow utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


def flatten_list(iterable_of_lists):
  """Unpack lists into a single list."""
  # pylint: disable=g-complex-comprehension
  return [x for sublist in iterable_of_lists for x in sublist]


def load_pickle(filename):
  """Load an object from a pickled file."""
  with tf.gfile.Open(filename, "rb") as f:
    return pickle.load(f)


def transpose_lists(lsts):
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def ensure_unicode(x):
  """Convert `x` to unicode, if it is not unicode already."""
  if isinstance(x, six.text_type):
    return x
  else:
    return six.text_type(str(x), "utf8")


def print_table(table):
  """Print the list of strings with evenly spaced columns."""
  # print while padding each column to the max column length
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  for row in table:
    print(" ".join(formats[i].format(row[i]) for i in range(len(row))))


def get_model_checkpoint(model_dir):
  """Select a checkpoint to use for a given model directory.

  Args:
    model_dir: the directory containing the target model
  Raises:
    ValueError: if there was a problem locating the checkpoint
  Returns:
    This method the "best-checkpoint" checkpoint as exported by
     `BestCheckpointExporter` if present, else the latest checkpoint
  """
  best_dir = os.path.join(model_dir, "export", "best-checkpoint")
  if tf.gfile.Exists(best_dir):
    best_dir_files = tf.gfile.ListDirectory(best_dir)
    if best_dir_files:
      tf.logging.info("Found best export, restoring from there")
      checkpoint_path = None
      for filename in tf.gfile.ListDirectory(best_dir):
        if filename.endswith(".index"):
          checkpoint_path = filename[:-len(".index")]
          break

      if checkpoint_path is None:
        raise ValueError("Unable to find checkpoint in " + best_dir)

      return os.path.join(best_dir, checkpoint_path)

  tf.logging.info("No best export found, restoring most recent")
  return tf.train.latest_checkpoint(model_dir)

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
"""Utilities for file I/O."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf


def make_empty_dir(path):
  """Makes an empty directory at `path`, deleting `path` first if needed."""
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)


def copy_files_to_dir(source_filepattern, dest_dir):
  """Copies files matching `source_filepattern` into `dest_dir`."""
  for source_path in tf.gfile.Glob(source_filepattern):
    dest_path = os.path.join(dest_dir, os.path.basename(source_path))
    tf.gfile.Copy(source_path, dest_path)


def set_file_contents(data, path):
  """Overwrites `path` with `data."""
  with tf.gfile.Open(path, "w") as output_file:
    output_file.write(data)

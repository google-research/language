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
"""Utilities for dealing with model exports."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re

import tensorflow.compat.v1 as tf


def export_paths(glob_pattern, regex_pattern):
  """Gets a list of exported paths, ordered by timestamp (ascending).

  The two patterns are necessary since glob doesn't support general regexes that
  are required for stricter matches that avoid temporary files.

  Args:
    glob_pattern (string): Input to tf.gfile.Glob.
    regex_pattern (string): Regular expression for additional filtering. This
      should have a match group named "timestamp".

  Returns:
    Filenames sorted by timestamp (later paths at the end).
  """
  matcher = re.compile(regex_pattern)
  filtered_exports = []  # (int timestamp, string path) pairs

  for filename in tf.gfile.Glob(glob_pattern):
    match = matcher.match(filename)
    if match:
      # Gets the timestamp as an integer, instead of comparing it lexically.
      timestamp = int(match.groupdict().get("timestamp", "0"))
      filtered_exports.append((timestamp, filename))

  return [filename for timestamp, filename in sorted(filtered_exports)]


def latest_export_path(glob_pattern, regex_pattern):
  """Get the latest exported path based on timestamp."""
  sorted_paths = export_paths(glob_pattern, regex_pattern)
  return sorted_paths[-1] if sorted_paths else None


def best_export_path(model_dir, best_prefix="best"):
  export_prefix = os.path.join(model_dir, "export", best_prefix)
  return latest_export_path(
      glob_pattern="{}/*".format(export_prefix),
      regex_pattern="{}/(?P<timestamp>[0-9]+)".format(export_prefix))


def tfhub_export_path(model_dir, hub_prefix, module_prefix):
  export_prefix = os.path.join(model_dir, "export", hub_prefix)
  return latest_export_path(
      glob_pattern="{}/*/{}".format(export_prefix, module_prefix),
      regex_pattern="{}/(?P<timestamp>[0-9]+)/{}".format(
          export_prefix, module_prefix))


def clean_tfhub_exports(model_dir, hub_prefix, exports_to_keep):
  export_prefix = os.path.join(model_dir, "export", hub_prefix)
  sorted_paths = export_paths(
      glob_pattern="{}/*".format(export_prefix),
      regex_pattern="{}/(?P<timestamp>[0-9]+)".format(export_prefix))
  if len(sorted_paths) > exports_to_keep:
    for to_remove in sorted_paths[:-exports_to_keep]:
      assert "/export/" in to_remove  # Safety with recursive deletion.
      tf.gfile.DeleteRecursively(to_remove)

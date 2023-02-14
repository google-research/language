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
"""Utilities for reading and writing jsonl files."""

import json
from tensorflow.io import gfile


def read(filepath, limit=None, verbose=False):
  """Read jsonl file to a List of Dicts."""
  data = []
  with gfile.GFile(filepath, "r") as jsonl_file:
    for idx, line in enumerate(jsonl_file):
      if limit is not None and idx >= limit:
        break
      if verbose and idx % 100 == 0:
        # Print the index every 100 lines.
        print("Processing line %s." % idx)
      try:
        data.append(json.loads(line))
      except json.JSONDecodeError as e:
        print("Failed to parse line: `%s`" % line)
        raise e
  print("Loaded %s lines from %s." % (len(data), filepath))
  return data


def write(filepath, rows):
  """Write a List of Dicts to jsonl file."""
  with gfile.GFile(filepath, "w") as jsonl_file:
    for row in rows:
      line = "%s\n" % json.dumps(row)
      jsonl_file.write(line)
  print("Wrote %s lines to %s." % (len(rows), filepath))

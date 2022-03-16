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
"""Utilties for reading and writing TXT dataset files."""

from tensorflow.io import gfile


def read_txt(filename):
  """Read file to list of lines."""
  examples = []
  with gfile.GFile(filename, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      examples.append(line)
  print("Loaded %s lines from %s." % (len(examples), filename))
  return examples


def write_txt(examples, filename):
  """Write examples to tsv file."""
  with gfile.GFile(filename, "w") as tsv_file:
    for example in examples:
      line = "%s\n" % example
      tsv_file.write(line)
  print("Wrote %s lines to %s." % (len(examples), filename))

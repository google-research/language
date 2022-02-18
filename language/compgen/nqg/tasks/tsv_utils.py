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
r"""Utilties for reading and writing files.

Expected format for TSV file is that each line has one example, with each
element separated by \t. The number of element should be the same as
expected_num_columns.

Expected format for examples in memory is a list where each element is:
(element_1, element_2, ...), or [element_1, element_2, ...]
The number of element should be the same as expected_num_columns.
"""

from tensorflow.io import gfile


def read_tsv(filename, expected_num_columns=2):
  """Read file to list of examples."""
  examples = []
  with gfile.GFile(filename, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      cols = line.split("\t")
      if len(cols) != expected_num_columns:
        raise ValueError("Line '%s' has %s columns (%s)" %
                         (line, len(cols), cols))
      examples.append(cols)
  print("Loaded %s examples from %s." % (len(examples), filename))
  return examples


def write_tsv(examples, filename, expected_num_columns=2):
  """Write examples to tsv file."""
  with gfile.GFile(filename, "w") as tsv_file:
    for example in examples:
      if len(example) != expected_num_columns:
        raise ValueError("Example '%s' has %s columns." %
                         (example, len(example)))
      example = "\t".join(example)
      line = "%s\n" % example
      tsv_file.write(line)
  print("Wrote %s examples to %s." % (len(examples), filename))


def merge_shared_tsvs(filename):
  """Merge multiple tsv files into one."""
  output_files = gfile.glob("%s-*-of-*" % filename)
  all_examples = []
  for output_file in output_files:
    examples = read_tsv(output_file)
    all_examples.extend(examples)
  write_tsv(all_examples, filename)

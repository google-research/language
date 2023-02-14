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
"""Utilties for reading and writing tsv and txt files."""

from tensorflow.io import gfile


def read_tsv(filepath, delimiter="\t", max_splits=-1):
  """Read file to list of rows."""
  rows = []
  with gfile.GFile(filepath, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      cols = line.split(delimiter, max_splits)
      rows.append(cols)
  print("Loaded %s rows from %s." % (len(rows), filepath))
  return rows


def write_tsv(rows, filepath, delimiter="\t"):
  """Write rows to tsv file."""
  with gfile.GFile(filepath, "w") as tsv_file:
    for row in rows:
      line = "%s\n" % delimiter.join([str(elem) for elem in row])
      tsv_file.write(line)
  print("Wrote %s rows to %s." % (len(rows), filepath))


def write_txt(rows, filepath):
  """Write newline separated text file."""
  with gfile.GFile(filepath, "w") as tsv_file:
    for row in rows:
      line = "%s\n" % row
      tsv_file.write(line)
  print("Wrote %s rows to %s." % (len(rows), filepath))


def read_txt(filepath):
  """Read newline separated text file."""
  rows = []
  with gfile.GFile(filepath, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      rows.append(line)
  print("Loaded %s rows from %s." % (len(rows), filepath))
  return rows

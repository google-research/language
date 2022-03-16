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
"""Merge source and target txt files to tsv."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("source", "", "Input txt file.")

flags.DEFINE_string("target", "", "Input txt file.")

flags.DEFINE_string("output", "", "Output tsv file.")


def read_txt(filename):
  """Read file to list of lines."""
  lines = []
  with gfile.GFile(filename, "r") as txt_file:
    for line in txt_file:
      line = line.decode().rstrip()
      lines.append(line)
  print("Loaded %s lines from %s." % (len(lines), filename))
  return lines


def main(unused_argv):
  source = read_txt(FLAGS.source)
  target = read_txt(FLAGS.target)
  examples = list(zip(source, target))
  tsv_utils.write_tsv(examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

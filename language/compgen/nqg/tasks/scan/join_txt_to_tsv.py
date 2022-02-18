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
"""Join source and target text files generated for MCD splits to TSV."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("source", "", "Source txt file.")

flags.DEFINE_string("target", "", "Target txt file.")

flags.DEFINE_string("output", "", "Joined tsv file.")


def read_examples(source_file, target_file):
  """Return list of (source, target) tuples."""
  sources = []
  targets = []

  with gfile.GFile(source_file, "r") as txt_file:
    for line in txt_file:
      sources.append(line.rstrip("\n"))

  with gfile.GFile(target_file, "r") as txt_file:
    for line in txt_file:
      targets.append(line.rstrip("\n"))

  examples = list(zip(sources, targets))
  return examples


def main(unused_argv):
  examples = read_examples(FLAGS.source, FLAGS.target)
  tsv_utils.write_tsv(examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

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
"""Randomly split dataset tsv file into two separate files."""

import random

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output_1", "",
                    "Output tsv file containing `num_examples_1` examples.")

flags.DEFINE_string("output_2", "",
                    "Output tsv file containing the remaining examples.")

flags.DEFINE_integer("num_examples_1", 3282, "Number of examples for output_1.")

flags.DEFINE_integer("seed", 1, "Seed for splitting examples.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  random.seed(FLAGS.seed)
  random.shuffle(examples)
  examples_1 = examples[:FLAGS.num_examples_1]
  examples_2 = examples[FLAGS.num_examples_1:]
  tsv_utils.write_tsv(examples_1, FLAGS.output_1)
  tsv_utils.write_tsv(examples_2, FLAGS.output_2)


if __name__ == "__main__":
  app.run(main)

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
"""Utility to merge tsv files."""

import random

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("input_1", "", "Input tsv file.")

flags.DEFINE_string("input_2", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output tsv file.")

flags.DEFINE_integer("duplicate_input_1", 1,
                     "Number of times to duplicate inputs in input_1.")


def main(unused_argv):
  input_1 = tsv_utils.read_tsv(FLAGS.input_1)
  input_2 = tsv_utils.read_tsv(FLAGS.input_2)
  outputs = input_1 * FLAGS.duplicate_input_1 + input_2
  random.shuffle(outputs)
  tsv_utils.write_tsv(outputs, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

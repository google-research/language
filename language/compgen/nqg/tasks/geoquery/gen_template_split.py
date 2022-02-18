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
"""Split dataset tsv file based on target templates."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import template_utils
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string(
    "output_1", "",
    "Output tsv file containing up to `max_num_examples_1` examples.")

flags.DEFINE_string("output_2", "",
                    "Output tsv file containing the remaining examples.")

flags.DEFINE_float("max_num_examples_1", 440,
                   "Maximum number of examples for output_1.")

flags.DEFINE_integer("seed", 1, "Seed for splitting examples.")


def funql_template_fn(target):
  """Simply returns target since entities are already anonymized in targets."""
  return target


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  examples_1, examples_2 = template_utils.split_by_template(
      examples,
      template_fn=funql_template_fn,
      max_num_examples_1=FLAGS.max_num_examples_1,
      seed=FLAGS.seed)
  tsv_utils.write_tsv(examples_1, FLAGS.output_1)
  tsv_utils.write_tsv(examples_2, FLAGS.output_2)


if __name__ == "__main__":
  app.run(main)

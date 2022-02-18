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
"""Split dataset based on length of source or target.

The expected usage of this script is to split a dataset (input) into a train
split (output_1) and a validation or test split (output_2).
"""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output_1", "", "Output tsv file for shorter examples.")

flags.DEFINE_string("output_2", "", "Output tsv file for longer examples.")

flags.DEFINE_integer("num_examples", 500, "Number of examples for output_1.")

flags.DEFINE_bool("use_target", False,
                  "If True, split based on target length rather than source.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  if FLAGS.use_target:
    sorted_examples = sorted(examples, key=lambda x: len(x[1].split(" ")))
  else:
    sorted_examples = sorted(examples, key=lambda x: len(x[0].split(" ")))
  examples_1 = sorted_examples[:FLAGS.num_examples]
  examples_2 = sorted_examples[FLAGS.num_examples:]
  tsv_utils.write_tsv(examples_1, FLAGS.output_1)
  tsv_utils.write_tsv(examples_2, FLAGS.output_2)


if __name__ == "__main__":
  app.run(main)

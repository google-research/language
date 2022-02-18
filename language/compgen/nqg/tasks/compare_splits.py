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
"""Compares whether two tsv files contain the same examples."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input_1", "", "Input tsv file 1.")

flags.DEFINE_string("input_2", "", "Input tsv file 2.")

flags.DEFINE_bool("ignore_order", False, "Whether to ignore order.")


def main(unused_argv):
  examples_1 = tsv_utils.read_tsv(FLAGS.input_1)
  examples_2 = tsv_utils.read_tsv(FLAGS.input_2)
  if examples_1 == examples_2:
    print("Examples are the same.")
  else:
    print("Examples are different.")
    if len(examples_1) != len(examples_2):
      print("Number of examples is different.")
    else:
      for idx, (example_1, example_2) in enumerate(zip(examples_1, examples_2)):
        if example_1 != example_2:
          print("First different example pair at idx %s:" % idx)
          print(example_1)
          print(example_2)
          break


if __name__ == "__main__":
  app.run(main)

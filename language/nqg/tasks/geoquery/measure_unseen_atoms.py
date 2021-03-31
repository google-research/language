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
"""Computes % of examples in input_2 containing an atom not input_1."""

from absl import app
from absl import flags

from language.nqg.tasks import mcd_utils
from language.nqg.tasks import tsv_utils
from language.nqg.tasks.geoquery import tmcd_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input_1", "", "Input tsv file.")

flags.DEFINE_string("input_2", "", "Input tsv file.")


def main(unused_argv):
  examples_1 = tsv_utils.read_tsv(FLAGS.input_1)
  examples_2 = tsv_utils.read_tsv(FLAGS.input_2)

  atoms_1 = mcd_utils.get_all_atoms(
      examples_1, get_atoms_fn=tmcd_utils.get_example_atoms)

  num_examples = 0
  num_examples_with_unseen_atom = 0
  for example in examples_2:
    atoms = tmcd_utils.get_example_atoms(example)
    num_examples += 1
    for atom in atoms:
      if atom not in atoms_1:
        print("New atom: %s" % atom)
        num_examples_with_unseen_atom += 1
        break

  print("num_examples: %s" % num_examples)
  print("num_examples_with_unseen_atom: %s" % num_examples_with_unseen_atom)
  print("pct: %s" % (float(num_examples_with_unseen_atom) / num_examples))


if __name__ == "__main__":
  app.run(main)

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
"""Split dataset tsv file based on TMCD methodology."""

import random

from absl import app
from absl import flags

from language.compgen.nqg.tasks import mcd_utils
from language.compgen.nqg.tasks import tsv_utils
from language.compgen.nqg.tasks.spider import tmcd_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output_1", "",
                    "Output tsv file containing `num_examples_1` examples.")

flags.DEFINE_string("output_2", "",
                    "Output tsv file containing the remaining examples.")

flags.DEFINE_integer("num_examples_1", 3282, "Number of examples for output_1.")

flags.DEFINE_integer("seed", 1, "Seed for splitting examples.")


class AtomAndCompoundCache(object):
  """Cache for computing atoms and compounds.

  As computing compounds for SQL requires parsing, it is therefore non-trivially
  expensive. This cache ensures parsing is only done once per example during
  the TMCD generation process.
  """

  def __init__(self):
    # Map of target strings to their atoms.
    self.target_to_atoms = {}
    # Map of target strings to their compounds.
    self.target_to_compounds = {}

  def get_atoms(self, example):
    key = tuple(example[1])
    if key not in self.target_to_atoms:
      atoms = tmcd_utils.get_example_atoms(example)
      self.target_to_atoms[key] = atoms
    return self.target_to_atoms[key]

  def get_compounds(self, example):
    key = tuple(example[1])
    if key not in self.target_to_compounds:
      compounds = tmcd_utils.get_example_compounds(example)
      self.target_to_compounds[key] = compounds
    return self.target_to_compounds[key]


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)

  # First, randomly split examples.
  random.seed(FLAGS.seed)
  random.shuffle(examples)
  examples_1 = examples[:FLAGS.num_examples_1]
  examples_2 = examples[FLAGS.num_examples_1:]

  # Initialize cache.
  cache = AtomAndCompoundCache()

  # Swap examples to meet atom constraint and maximize compound divergence.
  examples_1, examples_2 = mcd_utils.swap_examples(
      examples_1,
      examples_2,
      get_compounds_fn=cache.get_compounds,
      get_atoms_fn=cache.get_atoms,
      max_iterations=10000,
      max_divergence=None)
  tsv_utils.write_tsv(examples_1, FLAGS.output_1)
  tsv_utils.write_tsv(examples_2, FLAGS.output_2)


if __name__ == "__main__":
  app.run(main)

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
"""Tests for mcd_utils."""

from language.compgen.nqg.tasks import mcd_utils

import tensorflow as tf


# Define unigrams as atoms for tests.
def _get_atoms_fn(example):
  return set(example.split())


# Define bigrams as compounds for tests.
def _get_compounds_fn(example):
  tokens = example.split()
  bigrams = set()
  for idx in range(len(tokens) - 1):
    bigrams.add(" ".join(tokens[idx:idx + 2]))
  return bigrams


class McdUtilsTest(tf.test.TestCase):

  def test_divergence_is_0(self):
    examples_1 = ["jump twice", "walk thrice"]
    examples_2 = ["jump twice", "walk thrice"]
    compound_divergence = mcd_utils.measure_example_divergence(
        examples_1, examples_2, _get_compounds_fn)
    self.assertEqual(compound_divergence, 0.0)

  def test_divergence_is_1(self):
    examples_1 = ["jump thrice", "walk twice"]
    examples_2 = ["jump twice", "walk thrice"]
    compound_divergence = mcd_utils.measure_example_divergence(
        examples_1, examples_2, _get_compounds_fn)
    self.assertEqual(compound_divergence, 1.0)

  def test_swap_examples(self):
    examples_in_1 = ["jump twice", "walk twice", "jump thrice", "look and walk"]
    examples_in_2 = ["walk thrice", "walk and jump", "jump and walk"]
    compound_divergence_in = mcd_utils.measure_example_divergence(
        examples_in_1, examples_in_2, _get_compounds_fn)
    self.assertEqual(compound_divergence_in, 0.8)

    examples_out_1, examples_out_2 = mcd_utils.swap_examples(
        examples_in_1, examples_in_2, _get_compounds_fn, _get_atoms_fn)
    self.assertEqual(
        examples_out_1,
        ["jump and walk", "walk twice", "jump thrice", "look and walk"])
    self.assertEqual(examples_out_2,
                     ["walk thrice", "walk and jump", "jump twice"])

    compound_divergence_out = mcd_utils.measure_example_divergence(
        examples_out_1, examples_out_2, _get_compounds_fn)
    self.assertEqual(compound_divergence_out, 1.0)

  def test_get_all_compounds(self):
    examples = ["jump twice", "walk twice", "jump thrice", "look and walk"]
    compounds_to_count = mcd_utils.get_all_compounds(examples,
                                                     _get_compounds_fn)

    expected_compounds_to_count = {
        "jump twice": 1,
        "walk twice": 1,
        "jump thrice": 1,
        "look and": 1,
        "and walk": 1
    }
    self.assertDictEqual(expected_compounds_to_count, compounds_to_count)

  def test_compute_divergence(self):
    compound_counts_1 = {"a": 1, "b": 2, "c": 3}  # sum = 6
    compound_counts_2 = {"b": 4, "c": 5, "d": 6}  # sum = 15
    coef = 0.1

    divergence = mcd_utils.compute_divergence(compound_counts_1,
                                              compound_counts_2, coef)

    expected_divergence = 1.0 - (((2 / 6)**0.1 * (4 / 15)**0.9) +
                                 ((3 / 6)**0.1 * (5 / 15)**0.9))

    self.assertAlmostEqual(expected_divergence, divergence)


if __name__ == "__main__":
  tf.test.main()

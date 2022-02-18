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
"""Tests for tmcd_utils."""

import collections

from language.compgen.nqg.tasks.geoquery import tmcd_utils

import tensorflow as tf

FUNQL = ("answer ( intersection ( state , next_to_2 ( most ( state , loc_1 , "
         "city ) ) ) )")


class TmcdUtilsTest(tf.test.TestCase):

  def test_get_atoms_and_compounds(self):
    atoms = tmcd_utils.get_atoms(FUNQL)
    compounds = tmcd_utils.get_compounds(FUNQL)
    print(atoms)
    print(compounds)
    self.assertEqual(atoms, {
        "next_to_2", "loc_1", "most", "intersection", "state", "city", "answer"
    })
    expected_compounds = collections.Counter({
        "answer( intersection )": 1,
        "intersection( state , __ )": 1,
        "intersection( __ , next_to_2 )": 1,
        "next_to_2( most )": 1,
        "most( state , __ , __ )": 1,
        "most( __ , loc_1 , __ )": 1,
        "most( __ , __ , city )": 1
    })
    self.assertEqual(compounds, expected_compounds)

  def test_get_atoms_with_num_arguments(self):
    atoms = tmcd_utils.get_atoms_with_num_arguments(FUNQL)
    print(atoms)
    self.assertEqual(
        atoms, {
            "next_to_2_(1)", "loc_1", "most_(3)", "intersection_(2)", "state",
            "city", "answer_(1)"
        })


if __name__ == "__main__":
  tf.test.main()

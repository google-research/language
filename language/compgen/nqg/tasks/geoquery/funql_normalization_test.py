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
"""Tests for funql_reformatter."""

from language.compgen.nqg.tasks.geoquery import funql_normalization

import tensorflow as tf


class FunqlReformatterTest(tf.test.TestCase):

  def test_funql_reformatter_1(self):
    # which states have points higher than the highest point in colorado
    original = "answer(state(loc_1(place(higher_2(highest(place(loc_2(stateid(colorado)))))))))"
    normalized = funql_normalization.normalize_funql(original)
    restored = funql_normalization.restore_funql(normalized)
    expected_normalized = "answer(intersection(state,loc_1(intersection(place,higher_2(highest(intersection(place,loc_2(stateid(colorado)))))))))"
    self.assertEqual(normalized, expected_normalized)
    self.assertEqual(restored, original)

  def test_funql_reformatter_2(self):
    # what states border the states with the most cities
    original = "answer(state(next_to_2(most(state(loc_1(city(all)))))))"
    normalized = funql_normalization.normalize_funql(original)
    restored = funql_normalization.restore_funql(normalized)
    expected_normalized = "answer(intersection(state,next_to_2(most(state,loc_1,city))))"
    self.assertEqual(normalized, expected_normalized)
    self.assertEqual(restored, original)


if __name__ == "__main__":
  tf.test.main()

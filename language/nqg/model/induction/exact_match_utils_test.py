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
"""Tests for exact_match."""

from language.nqg.model.induction import exact_match_utils

import tensorflow as tf


class ExactMatchTest(tf.test.TestCase):

  def test_exact_match_1(self):
    dataset = [("salary between 8000 and 12000",
                "salaries between 8000 and 12000 .")]

    exact_match_rules = exact_match_utils.get_exact_match_rules(dataset)
    exact_match_rule_strings = {str(rule) for rule in exact_match_rules}
    self.assertEqual(exact_match_rule_strings,
                     {"between 8000 and 12000 ### between 8000 and 12000"})


if __name__ == "__main__":
  tf.test.main()

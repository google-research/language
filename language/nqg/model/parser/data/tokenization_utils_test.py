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
"""Tests for tokenization_utils."""

from language.nqg.model.parser import test_utils

from language.nqg.model.parser.data import tokenization_utils

import tensorflow as tf


class TokenizationUtilsTest(tf.test.TestCase):

  def test_get_wordpiece_inputs(self):
    tokenizer = test_utils.MockTokenizer()
    utterance = "what river traverses the most states"
    tokens = utterance.split(" ")
    (wordpiece_ids, num_wordpieces, token_start_wp_idx,
     token_end_wp_idx) = tokenization_utils.get_wordpiece_inputs(
         tokens, tokenizer, verbose=True)

    self.assertEqual(num_wordpieces, 8)
    self.assertEqual(wordpiece_ids, [1, 7, 8, 9, 10, 11, 12, 2])
    self.assertEqual(token_start_wp_idx, [1, 2, 3, 4, 5, 6])
    self.assertEqual(token_end_wp_idx, [1, 2, 3, 4, 5, 6])

  def test_get_wordpiece_inputs_special(self):
    tokenizer = test_utils.MockTokenizer()
    utterance = "what river traverses m0 and m1"
    tokens = utterance.split(" ")
    (wordpiece_ids, num_wordpieces, token_start_wp_idx,
     token_end_wp_idx) = tokenization_utils.get_wordpiece_inputs(
         tokens, tokenizer, verbose=True)

    self.assertEqual(num_wordpieces, 8)
    self.assertEqual(wordpiece_ids, [1, 7, 8, 9, 3, 13, 4, 2])
    self.assertEqual(token_start_wp_idx, [1, 2, 3, 4, 5, 6])
    self.assertEqual(token_end_wp_idx, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
  tf.test.main()

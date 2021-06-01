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
# Lint as: python3
"""Tests for language.search_agents.muzero.types.py."""

from absl.testing import parameterized
import tensorflow as tf

from language.search_agents.muzero import types


class TypesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'centered_around_answer',
          'substr': 'the answer',
          'text': 'some tokens before the answer and tokens after it',
          'number_of_words': 4,
          'expected': ('the answer', 'before the answer and')
      }, {
          'testcase_name': 'answer_at_beginning',
          'substr': 'the answer',
          'text': 'the answer and tokens after it',
          'number_of_words': 4,
          'expected': ('the answer', 'the answer and tokens')
      }, {
          'testcase_name': 'short_context',
          'substr': 'the answer',
          'text': 'before the answer after',
          'number_of_words': 8,
          'expected': ('the answer', 'before the answer after')
      }, {
          'testcase_name': 'too_short_context',
          'substr': 'the answer',
          'text': 'before the answer after',
          'number_of_words': 2,
          'expected': ('the answer', 'the answer')
      }, {
          'testcase_name': 'too_short_context_for_answer',
          'substr': 'the answer is long',
          'text': 'before the answer is long after',
          'number_of_words': 2,
          'expected': ('the answer is long', 'the answer is long')
      }, {
          'testcase_name': 'answer_missing',
          'substr': 'missing answer',
          'text': 'the answer is not in here',
          'number_of_words': 4,
          'expected': ('missing answer', 'the answer is not')
      }, {
          'testcase_name': 'answer_at_end',
          'substr': 'the answer',
          'text': 'many tokens then the answer',
          'number_of_words': 4,
          'expected': ('the answer', 'tokens then the answer')
      }, {
          'testcase_name': 'diacritics_missing_from_answer',
          'substr': 'the answer',
          'text': 'some tokens before the áñswer and tokens after it',
          'number_of_words': 4,
          'expected': ('the áñswer', 'before the áñswer and')
      }, {
          'testcase_name': 'unknown_character_in_answer',
          'substr': 'the ?nswer',
          'text': 'some tokens before the answer and tokens after it',
          'number_of_words': 4,
          'expected': ('the answer', 'before the answer and')
      }, {
          'testcase_name': 'unknown_character_missing_diacritics',
          'substr': 'the ?nswer',
          'text': 'some tokens before the áñswer and tokens after it',
          'number_of_words': 4,
          'expected': ('the áñswer', 'before the áñswer and')
      }, {
          'testcase_name': 'whitespace_normalization',
          'substr': 'the    answer',
          'text': '   before the answer    and  spaces  after   it',
          'number_of_words': 4,
          'expected': ('the answer', 'before the answer and')
      })
  def test_get_window_around_substr(self, text, substr, number_of_words,
                                    expected):
    self.assertEqual(
        types.HistoryEntry.get_window_around_substr(text, substr,
                                                    number_of_words), expected)


if __name__ == '__main__':
  tf.test.main()

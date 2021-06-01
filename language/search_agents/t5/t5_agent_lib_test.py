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
"""Tests for t5_agent.t5_agent_lib."""

from absl.testing import parameterized
from language.search_agents.t5 import t5_agent_lib
import tensorflow as tf

EMPTY_TERM = t5_agent_lib.Term(field='', term='', term_type='')


class T5AgentLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_refinements',
          'query': 'foo bar',
          'addition_terms': [],
          'subtraction_terms': [],
          'or_terms': [],
          'expected': "Query: 'foo bar'."
      }, {
          'testcase_name': 'add_terms',
          'query': 'foo bar',
          'addition_terms': [
              t5_agent_lib.Term('title', 'title term', ''),
              t5_agent_lib.Term('content', 'content term', ''),
              t5_agent_lib.Term('title', 'another title term', '')
          ],
          'subtraction_terms': [],
          'or_terms': [],
          'expected': "Query: 'foo bar'. Title must contain: 'title term'. "
                      "Content must contain: 'content term'. "
                      "Title must contain: 'another title term'."
      }, {
          'testcase_name': 'or_terms',
          'query': 'foo bar',
          'addition_terms': [],
          'subtraction_terms': [],
          'or_terms': [
              t5_agent_lib.Term('title', 'title term', ''),
              t5_agent_lib.Term('content', 'content term', ''),
              t5_agent_lib.Term('title', 'another title term', '')
          ],
          'expected': "Query: 'foo bar'. Should contain: "
                      "'title term', 'content term', 'another title term'."
      }, {
          'testcase_name': 'subtract_terms',
          'query': 'foo bar',
          'addition_terms': [],
          'subtraction_terms': [
              t5_agent_lib.Term('title', 'title term', ''),
              t5_agent_lib.Term('content', 'content term', ''),
              t5_agent_lib.Term('title', 'another title term', '')
          ],
          'or_terms': [],
          'expected': "Query: 'foo bar'. Title cannot contain: 'title term'. "
                      "Content cannot contain: 'content term'. "
                      "Title cannot contain: 'another title term'."
      })
  def test_query_to_prompt(self, query, addition_terms, subtraction_terms,
                           or_terms, expected):
    self.assertEqual(
        t5_agent_lib.query_to_prompt(query, addition_terms, subtraction_terms,
                                     or_terms), expected)

  @parameterized.named_parameters(
      {
          'testcase_name': 'regular_stop',
          'response': 'Stop.',
          'expected': (True, True, [EMPTY_TERM, EMPTY_TERM, EMPTY_TERM])
      }, {
          'testcase_name': 'unparseable_input',
          'response': 'Something unexpected.',
          'expected': (False, False, [EMPTY_TERM, EMPTY_TERM, EMPTY_TERM])
      }, {
          'testcase_name':
              'title_positive',
          'response':
              'Title must contain: \'foo bar\'.',
          'expected': (True, False, [
              t5_agent_lib.Term(field='title', term='foo bar', term_type=''),
              EMPTY_TERM, EMPTY_TERM
          ])
      }, {
          'testcase_name':
              'title_negative',
          'response':
              'Title cannot contain: \'foo bar\'.',
          'expected': (True, False, [
              EMPTY_TERM,
              t5_agent_lib.Term(field='title', term='foo bar', term_type=''),
              EMPTY_TERM
          ])
      }, {
          'testcase_name':
              'content_positive',
          'response':
              'Contents must contain: \'foo bar\'.',
          'expected': (True, False, [
              t5_agent_lib.Term(field='contents', term='foo bar', term_type=''),
              EMPTY_TERM, EMPTY_TERM
          ])
      }, {
          'testcase_name':
              'or_term',
          'response':
              'Should contain: \'foo bar\'.',
          'expected': (True, False, [
              EMPTY_TERM, EMPTY_TERM,
              t5_agent_lib.Term(field='', term='foo bar', term_type='')
          ])
      }, {
          'testcase_name':
              'content_negative',
          'response':
              'Contents cannot contain: \'foo bar\'.',
          'expected': (True, False, [
              EMPTY_TERM,
              t5_agent_lib.Term(field='contents', term='foo bar', term_type=''),
              EMPTY_TERM
          ])
      }, {
          'testcase_name':
              'extra_output',
          'response':
              'Ignore this. Contents cannot contain: \'foo bar\'. And this!',
          'expected': (True, False, [
              EMPTY_TERM,
              t5_agent_lib.Term(field='contents', term='foo bar', term_type=''),
              EMPTY_TERM
          ])
      }, {
          'testcase_name':
              'negative_and_positive_operator',
          'response':
              'Contents cannot contain: \'XX\'. Contents must contain: \'5\'.',
          'expected': (True, False, [
              t5_agent_lib.Term(field='contents', term='5', term_type=''),
              t5_agent_lib.Term(field='contents', term='XX', term_type=''),
              EMPTY_TERM
          ])
      })
  def test_parse_t5_response(self, response, expected):
    self.assertEqual(t5_agent_lib.parse_t5_response(response), expected)


if __name__ == '__main__':
  tf.test.main()

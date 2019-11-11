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
"""Tests for official_evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from language.xsp.evaluation import official_evaluation

NOISY_QUERY = (' SELECT sum( Population ) ,  avg(SurfaceArea)  FROM '
               'country WHERE Continent  =  "North America" AND '
               'SurfaceArea  >  3000 ;')
CLEAN_QUERY = ('select sum(population) , avg(surfacearea) from country '
               'where continent = \'north america\' and surfacearea > '
               '3000;')
WRONG_VALUE_QUERY = ('select sum(population) , avg(surfacearea) from country '
                     'where continent = \'north america\' and surfacearea > '
                     '5000;')
WRONG_COLS_QUERY = ('select capitol from country '
                    'where continent = \'north america\' and surfacearea > '
                    '5000;')
WRONG_TABS_QUERY = ('select sum(population) , avg(surfacearea) from state '
                    'where continent = \'north america\' and surfacearea > '
                    '3000;')
NO_ENTITIES_QUERY = ('select capitol from state;')


class OfficialEvaluationTest(absltest.TestCase):

  def test_normalize_sql_string(self):
    self.assertEqual(
        official_evaluation.normalize_sql_str(NOISY_QUERY), CLEAN_QUERY)

  def test_string_acc(self):
    self.assertFalse(
        official_evaluation.string_acc(NOISY_QUERY, WRONG_VALUE_QUERY))
    self.assertTrue(official_evaluation.string_acc(NOISY_QUERY, CLEAN_QUERY))

  def test_find_used_entities_in_string(self):
    cols = {'population', 'surfacearea', 'continent', 'capitol', 'country'}
    tables = {'country', 'state'}

    expected_cols = {'population', 'surfacearea', 'continent', 'country'}
    expected_tables = {'country'}

    self.assertEqual(
        official_evaluation.find_used_entities_in_string(
            NOISY_QUERY, cols, tables), (expected_cols, expected_tables))
    self.assertEqual(
        official_evaluation.find_used_entities_in_string(
            CLEAN_QUERY, cols, tables), (expected_cols, expected_tables))

  def test_compute_set_f1(self):
    self.assertEqual(official_evaluation.compute_set_f1(set(), set()), 1.)
    self.assertEqual(official_evaluation.compute_set_f1({'apple'}, set()), 0.)
    self.assertEqual(official_evaluation.compute_set_f1(set(), {'apple'}), 0.)
    self.assertEqual(
        official_evaluation.compute_set_f1({'apple'}, {'apple'}), 1.)
    self.assertEqual(
        official_evaluation.compute_set_f1({'apple', 'orange'},
                                           {'apple', 'orange'}), 1.)
    self.assertEqual(
        official_evaluation.compute_set_f1({'apple', 'orange'}, {'apple'}),
        2. / 3)
    self.assertEqual(
        official_evaluation.compute_set_f1({'apple'}, {'apple', 'orange'}),
        2. / 3)
    self.assertEqual(
        official_evaluation.compute_set_f1({'apple', 'pear'},
                                           {'apple', 'orange'}), 0.5)

  def test_col_tab_f1(self):
    schema = {
        'country': [{
            'field name': 'population'
        }, {
            'field name': 'surfacearea'
        }, {
            'field name': 'continent'
        }],
        'state': [{
            'field name': 'capitol'
        }, {
            'field name': 'country'
        }]
    }
    self.assertEqual(
        official_evaluation.col_tab_f1(schema, NOISY_QUERY, CLEAN_QUERY),
        (1.0, 1.0))
    self.assertEqual(
        official_evaluation.col_tab_f1(schema, NOISY_QUERY, WRONG_VALUE_QUERY),
        (1.0, 1.0))
    self.assertEqual(
        official_evaluation.col_tab_f1(schema, NOISY_QUERY, WRONG_COLS_QUERY),
        (0.75, 1.0))
    self.assertEqual(
        official_evaluation.col_tab_f1(schema, NOISY_QUERY, WRONG_TABS_QUERY),
        (6. / 7, 0.))
    self.assertEqual(
        official_evaluation.col_tab_f1(schema, CLEAN_QUERY, NO_ENTITIES_QUERY),
        (0., 0.))


if __name__ == '__main__':
  absltest.main()

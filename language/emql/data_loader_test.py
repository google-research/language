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
"""Tests for emql.data_loader."""

from language.emql import data_loader
import tensorflow.compat.v1 as tf

ROOT_DIR = './datasets/'


class DataLoaderTest(tf.test.TestCase):

  def test_loader_membership(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='membership',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    for one_data in loader.train_data_membership + loader.test_data_membership:
      self.assertGreater(len(one_data), 1)

  def test_loader_intersection_union(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='intersection',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    for set1, set2 in (loader.train_data_set_pair +
                       loader.test_data_set_pair):
      self.assertGreaterEqual(len(set1 & set2), 1)

  def test_loader_follow(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='set_follow',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    for subj_factids, rel_factids in (loader.train_data_follow +
                                      loader.test_data_follow):
      self.assertGreaterEqual(len(subj_factids & rel_factids), 1)

  def test_loader_metaqa(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='metaqa2',
        root_dir=ROOT_DIR + 'MetaQA/2hop/',
        kb_file='kb.txt',
        vocab_file='vocab.json')
    for question, _, answer_fact_ids in (
        loader.train_data_metaqa + loader.test_data_metaqa):
      self.assertGreater(len(question), 0)
      self.assertGreaterEqual(len(answer_fact_ids), 1)

  def test_loader_webqsp(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='webqsp',
        root_dir=ROOT_DIR + 'WebQSP/',
        kb_file='kb_webqsp_constraint2.txt')
    self.assertGreater(len(loader.train_data_webqsp), 2500)
    self.assertGreater(len(loader.test_data_webqsp), 1500)
    self.assertGreater(len(loader.fact2id), 1000000)

  def test_loader_query2box(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='query2box_uc',
        root_dir=ROOT_DIR + 'Query2Box/FB15k-237/',
        kb_file='kb.txt')
    self.assertIsNone(loader.train_data_query2box)
    self.assertEqual(len(loader.test_data_query2box), 5000)
    self.assertEqual(len(loader.test_data_query2box[0]), 3)

if __name__ == '__main__':
  tf.test.main()

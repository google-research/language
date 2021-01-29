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
"""Tests for emql.eval."""
import pickle

from language.emql import data_loader
from language.emql import eval as emql_eval
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile


class EvalTest(tf.test.TestCase):

  def setUp(self):
    super(EvalTest, self).setUp()
    self.task = 'ic'
    self.tmp_dir = self.create_tempdir().full_path + '/'

    # Construct sample KB.
    ent0, rel0, ent1, rel1, rel2 = 0, 0, 1, 1, 2
    answers = {
        ((ent0, (rel0,)), (ent1, (rel1,)), rel2): {3, 4, 5}
    }
    hard_answers = {
        ((ent0, (rel0,)), (ent1, (rel1,)), rel2): {5}
    }
    kb = ['ent0\trel0\tent1\n',
          'ent2\trel1\tent3\n',
          'ent4\trel2\tent5\n']
    ent2ind = {'ent%d' % i: i for i in range(6)}
    rel2ind = {'rel%d' % i: i for i in range(3)}
    ind2ent = {ind: ent for ent, ind in ent2ind.items()}
    ind2rel = {ind: rel for rel, ind in rel2ind.items()}

    # Dump to temp files.
    pickle.dump(answers,
                gfile.GFile(self.tmp_dir + 'test_ans_%s.pkl' % self.task, 'wb'))
    pickle.dump(
        hard_answers,
        gfile.GFile(self.tmp_dir + 'test_ans_%s_hard.pkl' % self.task, 'wb'))
    pickle.dump(ent2ind, gfile.GFile(self.tmp_dir + 'ent2ind.pkl', 'wb'))
    pickle.dump(rel2ind, gfile.GFile(self.tmp_dir + 'rel2ind.pkl', 'wb'))
    pickle.dump(ind2ent, gfile.GFile(self.tmp_dir + 'ind2ent.pkl', 'wb'))
    pickle.dump(ind2rel, gfile.GFile(self.tmp_dir + 'ind2rel.pkl', 'wb'))
    with gfile.GFile(self.tmp_dir + 'kb.txt', 'w') as f_kb:
      for line in kb:
        f_kb.write(line)

  def test_eval(self):
    root_dir = self.tmp_dir
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 64,
        'relation_emb_size': 64,
        'vocab_emb_size': 64,
        'train_entity_emb': False,
        'train_relation_emb': False,
        'intermediate_top_k': 10,
        'use_cm_sketch': True
    }

    loader = data_loader.DataLoader(
        params=params, name='query2box_ic',
        root_dir=root_dir,
        kb_file='kb.txt')

    q2b_metrics = emql_eval.Query2BoxMetrics(self.task, root_dir, loader)
    ent1, rel1, ent2, rel2, rel3 = 0, 0, 1, 2, 4
    features = np.array([ent1, rel1, ent2, rel2, rel3])
    # Predictions that are not in hard_answers will be skipped for
    # evaluation. We refer answers not in hard_answers as easy answers.
    # For clarity, we copy over the all_answers and
    # hard_answers of the query below:
    #     all_answers = {3, 4, 5}
    #     easy_answers = {3, 4}
    #     hard_answers = {5}
    answer_ids = np.array([4, 2, 5])
    tf_prediction = {'query': features, 'answer_ids': answer_ids}
    q2b_metrics.eval(tf_prediction)

    # hits@1 is 0 because the first non easy_answers is 2, but it's
    # not in correct hard_answers. hits@3 and hits@10 is 1.0 because
    # because the second non easy_answer 5 is in correct hard_answers.
    self.assertEqual(q2b_metrics.metrics['hits@1'], 0.0)
    self.assertEqual(q2b_metrics.metrics['hits@3'], 1.0)
    self.assertEqual(q2b_metrics.metrics['hits@10'], 1.0)
    self.assertEqual(q2b_metrics.metrics['mrr'], 0.5)


if __name__ == '__main__':
  tf.test.main()

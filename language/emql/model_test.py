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
"""Tests for emql.model."""

from language.emql import data_loader
from language.emql import model
from language.emql import module
import numpy as np
import tensorflow.compat.v1 as tf

ROOT_DIR = './datasets/'
KB_INDEX_DIR = ''



class ModelTest(tf.test.TestCase):

  def setUp(self):
    super(ModelTest, self).setUp()
    self.sess = tf.Session()

  def test_model_membership(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='membership',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    my_model = model.EmQL('membership', params, loader)
    self.sess.run(tf.global_variables_initializer())

    entity_ids = tf.constant([[0, 1, -1, -1], [2, -1, -1, -1]],
                             dtype=tf.int32)
    labels = np.zeros((2, loader.num_entities), dtype=np.float32)
    labels[0, 0] = 1.0
    labels[0, 1] = 1.0
    labels[1, 2] = 1.0
    labels = tf.constant(labels, dtype=tf.float32)
    loss, tensors = my_model.model_membership((entity_ids, labels), None)
    self.assertGreater(loss.eval(session=self.sess), 0)
    logits = tensors['logits']
    logits = logits.eval(session=self.sess)
    self.assertGreater(logits[0, 0], 0)
    self.assertGreater(logits[0, 1], 0)
    self.assertGreater(logits[1, 2], 0)

  def test_model_intersection(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='intersection',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    my_model = model.EmQL('intersection', params, loader)
    self.sess.run(tf.global_variables_initializer())
    candidate_set1 = np.zeros((1, loader.num_entities), dtype=np.float32)
    candidate_set1[0, 1] = 1.0
    candidate_set1[0, 2] = 1.0
    candidate_set2 = np.zeros((1, loader.num_entities), dtype=np.float32)
    candidate_set2[0, 2] = 1.0
    candidate_set2[0, 3] = 1.0
    labels = candidate_set1 * candidate_set2

    loss, tensors = my_model.model_intersection(
        (candidate_set1, candidate_set2, labels), params)
    self.assertGreater(loss.eval(session=self.sess), 0)
    logits = tensors['logits']
    logits = logits.eval(session=self.sess)
    self.assertGreater(logits[0, 2], logits[0, 1])
    self.assertGreater(logits[0, 2], logits[0, 3])

  def test_model_follow(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
    }
    loader = data_loader.DataLoader(
        params=params,
        name='set_follow',
        root_dir=ROOT_DIR + 'WikiMovies/',
        kb_file='kb.txt')
    my_model = model.EmQL('set_follow', params, loader)
    self.sess.run(tf.global_variables_initializer())

    subject_set = np.zeros((1, loader.num_facts), dtype=np.float32)
    subject_set[0, 1] = 1.0
    subject_set[0, 2] = 1.0
    relation_set = np.zeros((1, loader.num_facts), dtype=np.float32)
    relation_set[0, 2] = 1.0
    relation_set[0, 3] = 1.0
    labels = subject_set * relation_set

    loss, tensors = my_model.model_follow(
        (subject_set, relation_set, labels), params)
    self.assertGreater(loss.eval(session=self.sess), 0)
    logits = tensors['logits']
    logits = logits.eval(session=self.sess)
    self.assertGreater(logits[0, 2], logits[0, 1])
    self.assertGreater(logits[0, 2], logits[0, 3])

  def test_model_metaqa(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
        'intermediate_top_k': 10,
        'use_cm_sketch': True
    }
    loader = data_loader.DataLoader(
        params=params,
        name='metaqa2',
        root_dir=ROOT_DIR + 'MetaQA/2hop/',
        kb_file='kb.txt',
        vocab_file='vocab.json')
    my_model = model.EmQL('metaqa2', params, loader)
    question = tf.constant([[0, 1, 3, -1, -1]], dtype=tf.int32)
    q_entity_id = 0
    question_entity_id = tf.constant([q_entity_id], dtype=tf.int32)
    question_entity_sketch = loader.cm_context.get_sketch(xs=[0])
    question_entity_sketch = tf.constant(
        question_entity_sketch, dtype=tf.float32)
    question_entity_sketch = tf.expand_dims(question_entity_sketch, axis=0)
    answers = np.zeros((1, loader.num_facts), dtype=np.float32)
    answers[0, 100] = 1
    answers = tf.constant(answers, dtype=tf.float32)
    loss, _ = my_model.model_metaqa(
        (question, question_entity_id, question_entity_sketch, answers),
        params,
        hop=2,
        top_k=params['intermediate_top_k'])
    self.sess.run(tf.global_variables_initializer())
    self.assertGreater(loss.eval(session=self.sess), 0)

  def test_sketch(self):
    params = {
        'cm_width': 100,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
        'intermediate_top_k': 10,
        'use_cm_sketch': True
    }
    loader = data_loader.DataLoader(
        params=params,
        name='metaqa2',
        root_dir=ROOT_DIR + 'MetaQA/2hop/',
        kb_file='kb.txt',
        vocab_file='vocab.json')
    my_model = model.EmQL('metaqa2', params, loader)
    self.sess.run(tf.global_variables_initializer())

    entity_weights_np = np.array([[0.1, 0.9, 0.3]], dtype=np.float32)
    entity_ids = tf.constant([[0, 1, 2]], dtype=tf.int32)
    entity_weights = tf.constant(entity_weights_np, dtype=tf.float32)
    sketch = module.create_cm_sketch(
        entity_ids, entity_weights, my_model.all_entity_sketches,
        params['cm_width'])
    _, entity_weights_from_sketch = module.check_topk_fact_eligible(
        entity_ids, sketch, my_model.all_entity_sketches, params)
    entity_weights_from_sketch_np = entity_weights_from_sketch.eval(
        session=self.sess).astype(np.float32)
    self.assertAllClose(
        entity_weights_from_sketch_np, entity_weights_np)

  def test_model_webqsp(self):
    kb_index_str = KB_INDEX_DIR
    bert_handle_str = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 160,
        'relation_emb_size': 768,
        'vocab_emb_size': 64,
        'train_entity_emb': True,
        'train_relation_emb': True,
        'use_cm_sketch': True,
        'kb_index': kb_index_str,
        'bert_handle': bert_handle_str,
        'train_bert': False,
        'intermediate_top_k': 1000
    }
    loader = data_loader.DataLoader(
        params=params,
        name='webqsp',
        root_dir=ROOT_DIR + 'WebQSP/',
        kb_file='kb_webqsp_constraint2.txt')
    my_model = model.EmQL('webqsp', params, loader)
    question = tf.constant([[0, 1, 3, 0, 0]], dtype=tf.int32)
    segment_ids = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.int32)
    question_mask = tf.constant([[1, 1, 1, 0, 0]], dtype=tf.int32)
    q_entity_id = 0
    question_entity_id = tf.constant([q_entity_id], dtype=tf.int32)
    constraint_entity_id = tf.constant([q_entity_id], dtype=tf.int32)
    question_entity_sketch = loader.cm_context.get_sketch(xs=[0])
    question_entity_sketch = tf.constant(
        question_entity_sketch, dtype=tf.float32)
    question_entity_sketch = tf.expand_dims(question_entity_sketch, axis=0)
    constraint_entity_sketch = question_entity_sketch
    answers = tf.constant([[0, 1, 3, -1, -1]], dtype=tf.int32)
    loss, _ = my_model.model_webqsp(
        (question, segment_ids, question_mask, question_entity_id,
         question_entity_sketch, constraint_entity_id, constraint_entity_sketch,
         answers),
        params,
        top_k=params['intermediate_top_k'])
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    self.assertGreater(loss.eval(session=self.sess), 0)

  def test_model_query2box(self):
    params = {
        'cm_width': 10,
        'cm_depth': 10,
        'max_set': 100,
        'entity_emb_size': 64,
        'relation_emb_size': 64,
        'vocab_emb_size': 64,
        'train_entity_emb': False,
        'train_relation_emb': False,
        'intermediate_top_k': 5,
        'use_cm_sketch': True
    }
    loader = data_loader.DataLoader(
        params=params,
        name='query2box_uc',
        root_dir=ROOT_DIR + 'Query2Box/FB15k-237/',
        kb_file='kb.txt')
    my_model = model.EmQL('query2box_uc', params, loader)
    self.sess.run(tf.global_variables_initializer())

    ent1 = np.array([1, 2], dtype=np.int32)
    rel1 = np.array([11, 12], dtype=np.int32)
    ent2 = np.array([3, 4], dtype=np.int32)
    rel2 = np.array([13, 14], dtype=np.int32)
    rel3 = np.array([15, 16], dtype=np.int32)
    loss, tensors = my_model.model_query2box(
        'query2box_uc', (ent1, rel1, ent2, rel2, rel3), params)

    # query2box will not be trained so the loss should always be 0.
    self.assertEqual(loss.eval(session=self.sess), 0.0)
    answer_ids = tensors['answer_ids']
    answer_ids = answer_ids.eval(session=self.sess)
    self.assertEqual(answer_ids.shape[1], params['intermediate_top_k'])

if __name__ == '__main__':
  tf.test.main()

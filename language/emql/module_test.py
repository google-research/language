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
"""Tests for emql.module."""

from language.emql import module
import numpy as np
import tensorflow.compat.v1 as tf


class ModuleTest(tf.test.TestCase):

  def setUp(self):
    super(ModuleTest, self).setUp()
    self.sess = tf.Session()

  def test_sketch(self):
    """Test function check_topk_fact_eligible() and create_cm_sketch()."""
    params = {
        'cm_width': 10,
        'cm_depth': 3
    }
    self.sess.run(tf.global_variables_initializer())
    all_entity_sketches = tf.constant(
        [[1, 3, 5], [2, 4, 6], [7, 8, 9]],
        dtype=tf.int32)

    entity_weights_np = np.array([[0.1, 0.9]], dtype=np.float32)
    entity_ids = tf.constant([[0, 1]], dtype=tf.int32)
    entity_weights = tf.constant(entity_weights_np, dtype=tf.float32)
    sketch = module.create_cm_sketch(
        entity_ids, entity_weights, all_entity_sketches, params['cm_width'])
    oracle_sketch_np = np.zeros((1, 3, 10), dtype=np.float32)
    for entity_id in entity_ids.eval(session=self.sess)[0]:
      entity_weight = entity_weights.eval(session=self.sess)[0, entity_id]
      for i, j in enumerate(
          all_entity_sketches.eval(session=self.sess)[entity_id]):
        oracle_sketch_np[0, i, j] = entity_weight
    self.assertAllClose(sketch, oracle_sketch_np)

    _, entity_weights_from_sketch = module.check_topk_fact_eligible(
        entity_ids, sketch, all_entity_sketches, params)
    self.assertAllClose(entity_weights_from_sketch, entity_weights_np)

  def test_encode_op(self):
    params = {
        'cm_width': 100,
        'cm_depth': 5,
        'intermediate_top_k': 2
    }
    self.sess.run(tf.global_variables_initializer())
    all_entity_sketches = tf.constant(
        [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [11, 13, 15, 17, 19],
         [12, 14, 16, 18, 20]],
        dtype=tf.int32)
    set_ids = tf.constant([[0, 3, 0, 0]], dtype=tf.int32)
    set_mask = tf.constant([[1, 1, 0, 0]], dtype=tf.int32)
    embeddings_mat = tf.constant(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [-0.1, -0.2, -0.3],
         [-0.2, -0.3, -0.4]], dtype=tf.float32)

    set_embs, set_sketch = module.encode_op(
        params, set_ids, set_mask, embeddings_mat, all_entity_sketches)
    # The value of set_embs should be the centroid of vector embeddings_mat[0]
    # ([0.1, 0.2, 0.3]) and embeddings_mat[3] ([-0.2, -0.3, -0.4]).
    self.assertAllClose(
        set_embs, np.array([[-0.05, -0.05, -0.05]], np.float32))
    _, set_weights_from_sketch = module.check_topk_fact_eligible(
        tf.constant([[0, 1, 2, 3]], dtype=tf.int32), set_sketch,
        all_entity_sketches, params)
    # The returned weights should be the probability distribution of entities
    # in the set. Because only entity with id 0 and 3 are in the set, they
    # should each have the probability 0.5.
    self.assertAllClose(
        set_weights_from_sketch,
        np.array([[0.5, 0, 0, 0.5]], np.float32))

  def test_decode_op(self):
    params = {
        'cm_width': 100,
        'cm_depth': 5,
        'intermediate_top_k': 2
    }
    self.sess.run(tf.global_variables_initializer())
    all_entity_sketches = tf.constant(
        [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [11, 13, 15, 17, 19],
         [12, 14, 16, 18, 20]],
        dtype=tf.int32)

    embeddings_mat = tf.constant(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [-0.1, -0.2, -0.3],
         [-0.2, -0.3, -0.4]], dtype=tf.float32)
    set1_ids = tf.constant([[0, 1]], dtype=tf.int32)
    weights = tf.constant(
        np.array([[1, 1]], dtype=np.float32), dtype=tf.float32)

    set1_embs = tf.nn.embedding_lookup(embeddings_mat, set1_ids)  # b, s, d
    set1_embs = tf.reduce_sum(set1_embs, axis=1)  # b, d
    set1_sketch = module.create_cm_sketch(
        set1_ids, weights, all_entity_sketches, params['cm_width'])

    topk_ids, _ = module.decode_op(
        params, set1_embs, set1_sketch, embeddings_mat, all_entity_sketches)
    # The returned value should be entity in the set, scored by the centroid
    # [0.15, 0.25, 0.35]. Since the embedding of entity 1 ([0.2, 0.3, 0.4])
    # has the larger inner product than entity 0 ([0.1, 0.2, 0.3]).
    self.assertAllEqual(topk_ids, np.array([[1, 0]]))

  def test_intersection_union_op(self):
    params = {
        'cm_width': 100,
        'cm_depth': 5
    }
    self.sess.run(tf.global_variables_initializer())
    all_entity_sketches = tf.constant(
        [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [11, 13, 15, 17, 19],
         [12, 14, 16, 18, 20]],
        dtype=tf.int32)

    set1_ids = tf.constant([[0, 1, 2]], dtype=tf.int32)
    set2_ids = tf.constant([[1, 2, 3]], dtype=tf.int32)
    weights = tf.constant(
        np.array([[1, 1, 1]], dtype=np.float32), dtype=tf.float32)
    set1_embs = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
    set1_sketch = module.create_cm_sketch(
        set1_ids, weights, all_entity_sketches, params['cm_width'])
    set2_embs = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    set2_sketch = module.create_cm_sketch(
        set2_ids, weights, all_entity_sketches, params['cm_width'])

    # intersection
    with self.subTest(name='EmQL-intersection'):
      intersection_embs, intersection_sketch = module.intersection_op(
          set1_embs, set1_sketch, set2_embs, set2_sketch)
      # We first assert the embeddings (soft-type) of the intersected set
      # is the sum of the centroid of two sets, i.e. set1_embs + set2_embs
      self.assertAllClose(
          intersection_embs, np.array([[1.1, 2.2, 3.3]], np.float32))
      _, intersection_weights_from_sketch = module.check_topk_fact_eligible(
          tf.constant([[0, 1, 2, 3]], dtype=tf.int32), intersection_sketch,
          all_entity_sketches, params)
      # We assert only entity 1 and entity 2 have non-zero weights, since they
      # are the intersection of the two sets.
      self.assertAllClose(
          intersection_weights_from_sketch, np.array([[0.0, 1.0, 1.0, 0.0]]))

    # union
    with self.subTest(name='EmQL-union'):
      union_embs, union_sketch = module.union_op(set1_embs, set1_sketch,
                                                 set2_embs, set2_sketch)
      # The embeddings (soft-type) of the unioned set should be the sum of
      # the centroid of two sets, too.
      self.assertAllClose(
          union_embs, np.array([[1.1, 2.2, 3.3]], np.float32))
      _, union_weights_from_sketch = module.check_topk_fact_eligible(
          tf.constant([[0, 1, 2, 3]], dtype=tf.int32), union_sketch,
          all_entity_sketches, params)
      # We assert all entity 0, 1, 2, 3 have non-zero weights, because the
      # unioned set contains all 4 entities.
      self.assertAllClose(
          union_weights_from_sketch,
          np.array([[1.0, 1.0, 1.0, 1.0]], np.float32))

  def test_follow_op(self):
    params = {
        'cm_width': 100,
        'cm_depth': 5,
        'intermediate_top_k': 2
    }
    self.sess.run(tf.global_variables_initializer())

    # Define global kb tensors.
    all_entity_sketches = tf.constant(
        [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [11, 13, 15, 17, 19],
         [12, 14, 16, 18, 20]],
        dtype=tf.int32)
    all_relation_sketches = tf.constant(
        [[21, 23, 25, 27, 29], [22, 24, 26, 28, 30]],
        dtype=tf.int32)
    all_fact_subjids = tf.constant([0, 1], dtype=tf.int32)
    all_fact_relids = tf.constant([0, 1], dtype=tf.int32)
    all_fact_objids = tf.constant([2, 3], dtype=tf.int32)

    # Define embedding matrices.
    entity_embeddings_mat = tf.constant(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [-0.1, -0.2, -0.3],
         [-0.2, -0.3, -0.4]], dtype=tf.float32)
    relation_embeddings_mat = tf.constant(
        [[10, 11, 12], [-10, -11, -12]], dtype=tf.float32)
    kb_embeddings_mat = tf.concat([
        tf.nn.embedding_lookup(entity_embeddings_mat, all_fact_subjids),
        tf.nn.embedding_lookup(relation_embeddings_mat, all_fact_relids),
        tf.nn.embedding_lookup(entity_embeddings_mat, all_fact_objids)
    ], axis=1)

    # Construct entity set examples
    set1_ids = tf.constant([[0, 1]], dtype=tf.int32)
    set1_weights = tf.constant(
        np.array([[1, 1]], dtype=np.float32), dtype=tf.float32)
    set1_embs = tf.nn.embedding_lookup(entity_embeddings_mat, set1_ids)
    set1_embs = tf.reduce_sum(set1_embs, axis=1)  # b, d
    set1_sketch = module.create_cm_sketch(
        set1_ids, set1_weights, all_entity_sketches, params['cm_width'])

    # Construct relation set examples
    rel_id = tf.constant([[0]], dtype=tf.int32)
    rel_weights = tf.constant(
        np.array([[1]], dtype=np.float32), dtype=tf.float32)
    rel_embs = tf.nn.embedding_lookup(relation_embeddings_mat, rel_id)
    rel_embs = tf.reduce_sum(rel_embs, axis=1)  # b, d
    rel_sketch = module.create_cm_sketch(
        rel_id, rel_weights, all_relation_sketches, params['cm_width'])

    # Execute follow op and check returned sketch.
    obj_embs, obj_sketch, topk_fact_logits = module.follow_op(
        params, set1_embs, set1_sketch, rel_embs, rel_sketch,
        all_fact_subjids, all_fact_relids, all_fact_objids,
        entity_embeddings_mat, kb_embeddings_mat,
        all_entity_sketches, all_relation_sketches)
    _, obj_weights_from_sketch = module.check_topk_fact_eligible(
        tf.constant([[0, 1, 2, 3]], dtype=tf.int32), obj_sketch,
        all_entity_sketches, params)
    # Because set_1 contains entity 0, 1, and the relation to follow is 0,
    # the follow results should be the object of fact (0, 0, 2). So entity
    # 2 should hold the only non-zero probability in the returned weighted
    # from the count-min sketch.
    self.assertAllClose(
        obj_embs, np.array([[-0.1, -0.2, -0.3]]))
    self.assertAllClose(
        obj_weights_from_sketch, np.array([[0.0, 0.0, 1.0, 0.0]]))
    self.assertAllClose(
        topk_fact_logits[:, 0], np.array([121.78]))
    self.assertLess(
        topk_fact_logits[0, 1].eval(session=self.sess), 5e-5)  # very small

if __name__ == '__main__':
  tf.test.main()

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
"""EmQL modules for logical inferrence on knowledge base.

Implemented operations (see examples below):
  encode_op
  decode_op
  intersection_op
  union_op
  follow_op

An EmQL representation of a set of entities consists of two parts:
a dense embedding and a sparse count-min sketch. The dense embedding
describes the "soft-type" of entities in the set, computed as the weighted
average of entity embeddings in the set. The sparse count-min
sketch is computed from the entity ids in the set.

To derive the EmQL representation, we pre-compute a few tensors for the KB:
1. entity_embeddings_mat: entity embeddings matrix randomly initialized or
      loaded from checkpoints.
2. relation_embeddings_mat: entity embeddings matrix randomly initialized or
      loaded from checkpoints
3. kb_embeddings_mat: derived from entity_embeddings_mat and
      relation_embeddings_mat by concatenating subject embedding, relation
      embedding, and object embedding.
4. all_entity_sketches: precomputed hash ids of entities. Each entity is
      hashed "cm_depth" steps. The hash values is from [0, cm_width).
5. all_relation_sketches: precomputed hash ids of relations. Each relation is
      hashed "cm_depth" steps. The hash values is from [0, cm_width).
6. all_fact_subjids: the subject entity id of a fact.
6. all_fact_objids: the object entity id of a fact.
6. all_fact_relids: the relation id of a fact.


To encode a set of entities into its EmQL representation, we run the encode_op()
with a tensor of entity ids, their masks, the embedding matrix, and their
hashes. For example,
    a_s, b_s = encode_op(
        params, [[1, 2, 0]], [[1, 1, 0]],
        entity_embeddings_mat, all_entity_sketches)

To decode the EmQL embeddings, we use the decode_op().
    ids = decode_op(params, a_s, b_s,
                    entity_embeddings_mat, all_entity_sketches)

Logical operators are executed on EmQL representations:
    a_intersect, b_intersect = intersection_op(a_s1, a_s2, b_s1, b_s2)
    a_union, b_union = union_op(a_s1, a_s2, b_s1, b_s2)

We can further run follow_op on the EmQL representation:
    a_follow, b_follow = follow_op(
        params, a_sub, b_sub, a_rel, b_rel,
        all_fact_subjids, all_fact_relids, all_fact_objids,
        entity_embeddings_mat, kb_embeddings_mat,
        all_entity_sketches, all_relation_sketches)
"""


from absl import flags
from language.emql import util
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

_VERY_NEG = -1e6
ParamsType = Dict[str, Any]


def encode_op(params,
              set_ids, set_mask,
              embeddings_mat, all_sketches
              ):
  """Encode set to EmQL representations.

  Args:
    params: a dictionary of hyper-parameters.
    set_ids: a tf.Tensor of element ids in the set.
    set_mask: a 0/1 tf.Tensor of mask of element ids.
    embeddings_mat: embedding matrix of shape [vocab_size, d]
    all_sketches: precomputed hash ids of all candidate elements.

  Returns:
    dense embedding and sparse count-min sketch of the EmQL representation.
  """
  set_embs = tf.nn.embedding_lookup(embeddings_mat, set_ids)  # b, s, d
  set_mask = tf.cast(tf.expand_dims(set_mask, axis=2), dtype=tf.float32)
  set_embs = tf.reduce_sum(set_embs * set_mask, axis=1)  # b, d
  set_count = tf.reduce_sum(set_mask, axis=1)  # b, 1
  set_embs /= set_count  # b, d

  set_softmax = tf.squeeze(set_mask, axis=2) / set_count  # b, s
  set_sketch = create_cm_sketch(
      set_ids, set_softmax, all_sketches, params['cm_width'])
  return set_embs, set_sketch


def decode_op(params,
              set_embs, set_sketch,
              embeddings_mat, all_sketches
              ):
  """Decode set representations to member ids.

  Args:
    params: a dictionary of hyper-parameters.
    set_embs: dense embeddings of set.
    set_sketch: sparse count-min sketch of set.
    embeddings_mat: embedding lookup table for elements in the set.
    all_sketches: precomputed hash ids of all candidate elements.

  Returns:
    Ids of the top k predicted elements.
  """
  topk_logits, topk_ids = util.topk_search_fn(
      set_embs, embeddings_mat, params['intermediate_top_k'])
  # batch_size, top_k     batch_size, top_k
  if set_sketch is not None:
    is_topk_eligible, _ = check_topk_fact_eligible(
        topk_ids, set_sketch, all_sketches, params)
    topk_logits += (1 - is_topk_eligible) * _VERY_NEG
    # batch_size, top_k
  _, argmax = tf.nn.top_k(topk_logits, k=params['intermediate_top_k'])
  # batch_size, top_k
  topk_ids = tf.gather(topk_ids, argmax, batch_dims=1)
  # batch_size, top_k
  return topk_ids, topk_logits


def check_topk_fact_eligible(
    topk_subj_ids, query_entity_sketch,
    all_entity_sketches,
    params):
  """Extract probs from sketch.

  Args:
    topk_subj_ids: batch_size, topk
    query_entity_sketch: batch_size, depth, width
    all_entity_sketches: num_entities, depth
    params: hyper-parameters

  Returns:
    if topk_subj_ids are eligible and their probabilities
  """
  intermediate_top_k = tf.shape(topk_subj_ids)[1]
  topk_subj_ids = tf.reshape(topk_subj_ids, shape=[-1])
  # batch_size * topk
  topk_subj_sketches = tf.gather(all_entity_sketches, topk_subj_ids)
  # batch_size * topk, cm_depth
  topk_subj_sketches = tf.reshape(
      topk_subj_sketches,
      shape=[-1, intermediate_top_k, params['cm_depth']])
  # batch_size, topk, depth
  topk_fact_eligibility_bits = tf.gather(
      query_entity_sketch,
      tf.transpose(topk_subj_sketches, perm=[0, 2, 1]),
      batch_dims=2, axis=2)
  # batch_size, depth, topk
  topk_fact_prev_probs = tf.reduce_min(topk_fact_eligibility_bits, axis=1)
  # batch_size, topk
  is_topk_fact_eligible = tf.cast(
      topk_fact_prev_probs > 0.0, dtype=tf.float32)
  # batch_size, topk

  return is_topk_fact_eligible, topk_fact_prev_probs


def create_cm_sketch(topk_obj_ids,
                     topk_obj_weights,
                     all_entity_sketches,
                     cm_width):
  """Create cm sketches for a set of weighted entities.

  Args:
    topk_obj_ids: batch_size, topk
    topk_obj_weights: batch_size, topk
    all_entity_sketches: num_entities, depth
    cm_width: width of count-min sketch

  Returns:
    k hot dense vectors: batch_size, depth, width
  """
  topk_fact_obj_sketches = tf.gather(
      all_entity_sketches, topk_obj_ids, axis=0)
  # batch_size, topk, depth
  batch_size = tf.shape(topk_fact_obj_sketches)[0]
  topk = tf.shape(topk_fact_obj_sketches)[1]
  cm_depth = tf.shape(topk_fact_obj_sketches)[2]

  # We first create a sparse matrix from the hash values. We will then
  # convert it into dense matrix. This is more efficient than creating
  # k one-hot vectors and then aggregating them into one k-hot vector.

  # First prepare ids of non-zero values in the sparse matrix
  flattened_topk_hash_ids = tf.reshape(topk_fact_obj_sketches, shape=[-1])
  # batch_size * topk * depth
  topk_obj_weights = tf.tile(
      tf.expand_dims(topk_obj_weights, axis=2),
      multiples=[1, 1, cm_depth])
  # batch_size, topk, depth
  flattened_topk_obj_weights = tf.reshape(
      topk_obj_weights, shape=[-1])
  # batch_size * topk * depth
  batch_ids = tf.range(batch_size)
  # batch_size,
  batch_ids = tf.expand_dims(tf.expand_dims(batch_ids, axis=1), axis=2)
  # batch_size, 1, 1
  batch_ids = tf.tile(batch_ids, multiples=[1, topk, cm_depth])
  # batch_size, topk, depth
  flattened_batch_ids = tf.reshape(batch_ids, shape=[-1])
  # batch_size * topk * depth
  depth_ids = tf.range(cm_depth)
  # depth,
  depth_ids = tf.expand_dims(tf.expand_dims(depth_ids, axis=0), axis=1)
  # 1, 1, depth
  depth_ids = tf.tile(depth_ids, multiples=[batch_size, topk, 1])
  # batch_size, topk, depth
  flattened_depth_ids = tf.reshape(depth_ids, shape=[-1])
  # batch_size * topk * depth
  sparse_value_ids = tf.cast(tf.stack(
      [flattened_batch_ids, flattened_depth_ids, flattened_topk_hash_ids],
      axis=1), dtype=tf.int64)

  # Then prepare values of non-zero values in the sparse matrix. Values
  # are sorted to ascending order. If there are duplicates, later (larger)
  # values will be kept.
  sorted_orders = tf.argsort(
      flattened_topk_obj_weights, direction='ASCENDING', stable=True)
  # batch_size * topk * depth
  sorted_flattened_topk_obj_weights = tf.gather(
      flattened_topk_obj_weights, sorted_orders)
  sorted_sparse_value_ids = tf.gather(sparse_value_ids, sorted_orders)

  # Finally create sketch in sparse tensors and convert it to dense tensors.
  # We donot validate indices here. If multiple values are about to be assigned
  # to the same row and column, we will keep the last value, because the last
  # value is the larger one. This behaviour is by design.
  sparse_k_hot_sketch = tf.SparseTensor(
      indices=sorted_sparse_value_ids,
      values=sorted_flattened_topk_obj_weights,
      dense_shape=[batch_size, cm_depth, cm_width])
  dense_k_hot_sketch = tf.sparse.to_dense(
      sparse_k_hot_sketch, validate_indices=False)
  # batch_size, cm_depth, cm_width
  return dense_k_hot_sketch


def intersection_op(set1_embs, set1_sketch,
                    set2_embs,
                    set2_sketch):
  """Compute intersection between two sets.

  Args:
    set1_embs: dense embeddings of set1.
    set1_sketch: sparse count-min sketch of set1.
    set2_embs: dense embeddings of set2.
    set2_sketch: sparse count-min sketch of set2.
  Returns:
    Set embeddings and sketches for intersected set.
  """
  intersection_embs = set1_embs + set2_embs
  intersection_sketch = tf.minimum(set1_sketch, set2_sketch)
  return intersection_embs, intersection_sketch


def union_op(set1_embs, set1_sketch,
             set2_embs, set2_sketch
             ):
  """Compute union between two sets.

  Args:
    set1_embs: dense embeddings of set1.
    set1_sketch: sparse count-min sketch of set1.
    set2_embs: dense embeddings of set2.
    set2_sketch: sparse count-min sketch of set2.
  Returns:
    Set embeddings and sketches for unioned set.
  """
  union_embs = set1_embs + set2_embs
  union_sketch = tf.maximum(set1_sketch, set2_sketch)
  return union_embs, union_sketch


def follow_op(params,
              subject_embs, subject_sketch,
              relation_embs, relation_sketch,
              all_fact_subjids, all_fact_relids,
              all_fact_objids,
              entity_embeddings_mat, kb_embeddings_mat,
              all_entity_sketches, all_relation_sketches
              ):
  """Run follow() operation with EmQL representations.

  This module executes the follow() operations by retrieving facts
  against the subjects and relations of the query, and mapping the
  retrieved facts to their objects. Please see the EmQL paper for
  more details.

  Args:
    params: a dictionary of hyper-parameters.
    subject_embs: a dense tf.Tensor of shape [b, d] for embedding part
      of the set of subject entities.
    subject_sketch: a tf.Tensor of shape [b, cm_depth, cm_width] for the
      sparse sketch part of the EmQL set representation for subject entities.
      No sparse filtering will be applied if None.
    relation_embs: a dense tf.Tensor of shape [b, d] for embedding part
      of the set of relations.
    relation_sketch: a tf.Tensor of shape [b, cm_depth, cm_width] for the
      sparse sketch part of the EmQL set representation for relations.
      No sparse filtering will be applied if None.
    all_fact_subjids: a tf.Tensor that stores global information about
      the subject ids of all facts in the KB.
    all_fact_relids: a tf.Tensor that stores global information about
      the relation ids of all facts in the KB.
    all_fact_objids: a tf.Tensor that stores global information about
      the object ids of all facts in the KB.
    entity_embeddings_mat: a tf.Tensor of the entity embedding matrix.
    kb_embeddings_mat: a tf.Tensor of the kb embedding matrix as a
      concatenation of subject, relation, and object embeddings.
    all_entity_sketches: a tf.Tensor of precomputed entity id hashes.
      The hashes will be used to compute the count-min sketches. One
      could also compute the sketches on-the-fly if efficiency permitted.
    all_relation_sketches: a tf.Tensor of precomputed relation id hashes
      (similar to all_entity_sketches above).

  Returns:
    Set embeddings and sketches for returned object entities, and the logits
    of the top k facts (that automatically become the logits of the top k
    object entities).
  """
  query_embs = tf.concat(
      [subject_embs, relation_embs, tf.zeros_like(subject_embs)],
      axis=1)

  # Retrieve top k facts that match the queries, and check if the top
  # k facts are eligible by checking the count-min sketch. Subject
  # entities must have non-zero probability to be considered as
  # eligible.
  topk_fact_logits, topk_fact_ids = util.topk_search_fn(
      query_embs, kb_embeddings_mat, params['intermediate_top_k'])
  # batch_size, top_k     batch_size, top_k
  topk_subj_ids = tf.gather(all_fact_subjids, topk_fact_ids)
  # batch_size, topk
  topk_rel_ids = tf.gather(all_fact_relids, topk_fact_ids)
  # batch_size, topk
  topk_obj_ids = tf.gather(all_fact_objids, topk_fact_ids)
  # batch_size, topk

  # We check if the subject and relation of the top k retrieved facts
  # will pass their sketches. The sketch filters are applied if they are
  # not None. Facts that do not pass the sketch will be masked with
  # a very negative value assigned.
  if subject_sketch is not None:
    is_topk_fact_eligible, _ = check_topk_fact_eligible(
        topk_subj_ids, subject_sketch, all_entity_sketches, params)
    topk_fact_logits += (1 - is_topk_fact_eligible) * _VERY_NEG
    # batch_size, top_k

  if relation_sketch is not None:
    assert all_relation_sketches is not None
    is_topk_fact_eligible, _ = check_topk_fact_eligible(
        topk_rel_ids, relation_sketch, all_relation_sketches, params)
    topk_fact_logits += (1 - is_topk_fact_eligible) * _VERY_NEG
    # batch_size, top_k

  # Construct query embeddings for next iteration.
  topk_fact_obj_embs = tf.nn.embedding_lookup(
      entity_embeddings_mat, topk_obj_ids)
  # batch_size, top_k, hidden_size
  topk_softmax = tf.nn.softmax(topk_fact_logits)
  # batch_size, top_k, 1
  object_embs = tf.reduce_sum(
      topk_fact_obj_embs * tf.expand_dims(topk_softmax, axis=2), axis=1)
  # batch_size, hidden_size

  # Update sketches.
  object_sketch = create_cm_sketch(
      topk_obj_ids, topk_softmax, all_entity_sketches,
      cm_width=params['cm_width'])

  # topk_fact_logits are also returned to help compute loss against the logits.
  # They are used as the logits of object entities of the follow results.
  return object_embs, object_sketch, topk_fact_logits

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
r"""Implementation of EmQL model on different tasks.

We implement EmQL model that contains a dense retrieval with sparse filtering.
There are a few different training targets: membership, intersection, union,
follow, set_follow, metaqa2 and metaqa3. The first few targets (except for
the last two), are trained to represent a knowledge base in the embedding
space. The KB embeddings are them used in the downstream tasks metaqa2 and
metaqa3.

This file mainly consists of three groups of functions:
1. model definitions: model_xxx() and helper functions
2. evaluation functions: run_tf_evaluation()
3. prediction functions: get_tf_xxx_prediction()
which are called from build_model_fn(). Models and predictions are selected
and computed based on the names of targets.

"""



from absl import flags
from language.emql import module
from language.emql import util
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub

FLAGS = flags.FLAGS
VERY_NEG = -1e6
ZERO_THRES = 1e-5
TensorTupleType = Tuple[tf.Tensor, Ellipsis]
ParamsType = Dict[str, Any]
ModelReturnType = Tuple[tf.Tensor, Dict[str, tf.Tensor]]


class EmQL(object):
  """EmQL Implementation."""

  def __init__(self, name, params, data_loader):

    with tf.variable_scope('embeddings_mat', reuse=tf.AUTO_REUSE):
      self.entity_embeddings_mat = tf.get_variable(
          name='entity_embeddings_mat',
          shape=[data_loader.num_entities, params['entity_emb_size']],
          initializer=tf.random_normal_initializer(),
          trainable=params['train_entity_emb'])

      self.relation_embeddings_mat = tf.get_variable(
          name='relation_embeddings_mat',
          shape=[data_loader.num_relations, params['relation_emb_size']],
          initializer=tf.random_normal_initializer(),
          trainable=params['train_relation_emb'])

      if name in ['metaqa2', 'metaqa3']:
        self.word_embeddings_mat = tf.get_variable(
            name='word_embeddings_mat',
            shape=[data_loader.num_vocab, params['vocab_emb_size']],
            initializer=tf.random_normal_initializer(),
            trainable=True)

      ### Convert fact info to tf tensors.
      self.all_entity_ids = tf.constant(
          data_loader.all_entity_ids, dtype=tf.int32)
      self.all_entity_sketches = tf.constant(
          data_loader.all_entity_sketches, dtype=tf.int32)
      self.all_relation_sketches = tf.constant(
          data_loader.all_relation_sketches, dtype=tf.int32)
      self.all_fact_subjids = tf.constant(
          data_loader.all_fact_subjids, dtype=tf.int32)
      self.all_fact_relids = tf.constant(
          data_loader.all_fact_relids, dtype=tf.int32)
      self.all_fact_objids = tf.constant(
          data_loader.all_fact_objids, dtype=tf.int32)

      # Construct fact embeddings.
      self.kb_embeddings_mat = self.load_kb_embeddings(name, params)

      # Construct other utilities
      if name == 'webqsp':
        self.all_entity_is_cvt = tf.constant(
            data_loader.all_entity_is_cvt, dtype=tf.float32)
        self.bert_module = hub.Module(
            params['bert_handle'],
            tags={'train'} if params['train_bert'] else {},
            trainable=params['train_bert'])

  def load_kb_embeddings(self, name, params):
    """Construct or load KB embeddings.

    Args:
      name: task name
      params: params

    Returns:
      a tensor for kb embeddings
    """
    if (name in ['set_follow', 'metaqa2', 'metaqa3'] or
        name.startswith('query2box')):
      all_fact_subj_embs = tf.nn.embedding_lookup(self.entity_embeddings_mat,
                                                  self.all_fact_subjids)
      # num_facts, hidden_size
      all_fact_rel_embs = tf.nn.embedding_lookup(self.relation_embeddings_mat,
                                                 self.all_fact_relids)
      # num_facts, hidden_size
      all_fact_obj_embs = tf.nn.embedding_lookup(self.entity_embeddings_mat,
                                                 self.all_fact_objids)
      # num_facts, hidden_size
      kb_embs = tf.concat(
          [all_fact_subj_embs, all_fact_rel_embs, all_fact_obj_embs], axis=1)
      # num_fact, hidden_size * 3
      return kb_embs

    elif name in ['webqsp']:
      return util.load_db_checkpoint(
          var_name='embeddings_mat/kb_embeddings_mat',
          checkpoint_dir=params['kb_index'],
          cpu=False,
          trainable=False)

  ##############################################################
  ########################### Models ###########################
  ##############################################################

  def get_tf_model(self, name, features,
                   params):
    """Select model.

    Args:
      name: model name
      features: features
        candidate_subj_ids -- batch_size, num_candidate
      params: hyper-parameters

    Returns:
      logits
    """
    if name == 'membership':
      return self.model_membership(features, params)
    elif name == 'intersection' or name == 'union':
      return self.model_intersection(features, params)
    elif name == 'follow' or name == 'set_follow':
      return self.model_follow(features, params)
    elif name == 'mixture':
      return self.model_mixture(features, params)
    elif name == 'metaqa2' or name == 'metaqa3':
      num_hops = int(name[-1])
      return self.model_metaqa(
          features, params, hop=num_hops, top_k=params['intermediate_top_k'])
    elif name == 'webqsp':
      return self.model_webqsp(
          features, params, top_k=params['intermediate_top_k'])
    elif name.startswith('query2box'):
      return self.model_query2box(name, features, params)
    else:
      raise ValueError('name not recognized')

  def model_membership(self, features,
                       unused_params):
    """Compute logits for set membership.

    A representation of set is the average of embeddings of entities in
    the set.

    Args:
      features:
        -- entity_ids: ids of entities in the set, padded with -1
                       e.g. [1, 5, 3, -1, -1]
                       # batch_size, max_set
        -- labels: if the i'th entity is in the set, 1 for True, 0 for False
                   e.g. [0, 1, 0, 1, 0, 1, 0, 0, ...]
                   # batch_size, num_entities
      unused_params: hyper-parameters

    Returns:
      predictions and losses
    """
    entity_ids, labels = features
    entity_mask = tf.cast(tf.not_equal(entity_ids, -1), dtype=tf.int32)
    # batch_size, max_set
    entity_embs = tf.nn.embedding_lookup(
        self.entity_embeddings_mat, entity_ids * entity_mask)
    # batch_size, max_set, hidden_size
    entity_mask_3d = tf.cast(
        tf.expand_dims(entity_mask, axis=2), dtype=tf.float32)
    # batch_size, max_set, 1
    set_embs = tf.reduce_sum(entity_embs * entity_mask_3d, axis=1)
    # batch_size, hidden_size
    set_embs /= tf.reduce_sum(entity_mask_3d, axis=1)
    # batch_size, hidden_size
    logits = tf.matmul(set_embs, self.entity_embeddings_mat, transpose_b=True)
    # batch_size, num_entities
    loss = self.compute_tf_loss(logits, labels)
    return loss, {'logits': logits, 'labels': labels}

  def model_intersection(self, features,
                         params):
    """Compute logits for intersection.

    A representation of intersection is the sum of representations of sets
    that are being intersected.

    Args:
      features: features
        -- candidate_set1: a k-hot vector of length num_entities representing
                           the entities in the first set.
                           e.g. [1, 0, 1, 0, 1, 1, 0, 0]
        -- candidate_set2: a k-hot vector of length num_entities representing
                           the entities in the first set.
                           e.g. [1, 1, 0, 0, 1, 0, 0, 0]
        -- labels: a k-hot vector of length num_entities representing
                           the entities in the intersection of two sets.
                           e.g. [1, 0, 0, 0, 1, 0, 0, 0]
      params: hyper-parameters

    Returns:
      predictions and losses
    """
    candidate_set1, candidate_set2, labels = features
    candidate_set1_mask = tf.expand_dims(candidate_set1, axis=2)
    # batch_size, num_candidate, 1
    candidate_set2_mask = tf.expand_dims(candidate_set2, axis=2)
    # batch_size, num_candidate, 1
    candidate_embs = self.entity_embeddings_mat
    # batch_size, num_candidate, hidden_size
    set1_embs_query = tf.reduce_sum(
        candidate_embs * candidate_set1_mask, axis=1, keepdims=True) \
        / tf.reduce_sum(candidate_set1_mask, axis=1, keepdims=True)
    # batch_size, 1, hidden_size
    set2_embs_query = tf.reduce_sum(
        candidate_embs * candidate_set2_mask, axis=1, keepdims=True) \
        / tf.reduce_sum(candidate_set2_mask, axis=1, keepdims=True)
    # batch_size, 1, hidden_size
    intersection_embs_query = set1_embs_query + set2_embs_query
    # batch_size, 1, hidden_size
    logits, _ = self.compute_logits(candidate_embs, intersection_embs_query,
                                    params['entity_emb_size'])
    loss = self.compute_tf_loss(logits, labels)
    return loss, {'logits': logits, 'labels': labels}

  def model_follow(self, features,
                   params):
    """Compute logits for follow operation.

    A follow operation is considered as the intersection of a set of
    facts with correct subjects and a set of facts with correct relation.

    Args:
      features: features
        -- subject_set: all facts that have the correct subject
        -- relation_set: all facts that have the correct relation
        -- labels: facts that have both correct subject and relation
      params: hyper-parameters

    Returns:
      predictions and losses
    """
    subject_set, relation_set, labels = features
    subject_set_mask = tf.expand_dims(subject_set, axis=2)
    # batch_size, num_candidate, 1
    relation_set_mask = tf.expand_dims(relation_set, axis=2)
    # batch_size, num_candidate, 1
    candidate_embs = self.load_kb_embeddings('set_follow', params)
    # batch_size, num_candidate, hidden_size * 3

    set1_embs_query = tf.reduce_sum(
        candidate_embs * subject_set_mask, axis=1, keepdims=True) \
        / tf.reduce_sum(subject_set_mask, axis=1, keepdims=True)
    # batch_size, 1, hidden_size * 3
    set2_embs_query = tf.reduce_sum(
        candidate_embs * relation_set_mask, axis=1, keepdims=True) \
        / tf.reduce_sum(relation_set_mask, axis=1, keepdims=True)
    # batch_size, 1, hidden_size * 3
    intersection_embs_query = set1_embs_query + set2_embs_query
    # batch_size, 1, hidden_size * 3
    logits, _ = self.compute_logits(
        candidate_embs, intersection_embs_query,
        params['entity_emb_size'] * 2 + params['relation_emb_size'])
    loss = self.compute_tf_loss(logits, labels)
    return loss, {'logits': logits, 'labels': labels}

  def model_mixture(self, features,
                    params):
    """Jointly train four tasks.

    m_xxx stands for feature and label to train the set membership task.
    p_xxx stands for a pair of set to train intersection and union.
    f_xxx stands for features to train the follow operation.

    Args:
      features: features
      params: hyper-parameters

    Returns:
      predictions and losses
    """
    m_padded_ent_ids, m_labels, \
    p_candidate_set1, p_candidate_set2, p_union_labels, p_intersection_labels, \
    f_subject_set, f_relation_set, f_labels = features

    # Run model on the task of set membership.
    m_features = (m_padded_ent_ids, m_labels)
    m_loss, m_tensors = self.model_membership(m_features, params)

    # Run model on the task of intersection and union.
    p_features = (p_candidate_set1, p_candidate_set2, p_intersection_labels)
    p_intersection_loss, p_tensors = self.model_intersection(
        p_features, params)
    p_union_loss = self.compute_tf_loss(
        p_tensors['logits'], p_union_labels)

    # Run model on the task of set follow.
    f_features = (f_subject_set, f_relation_set, f_labels)
    f_loss, f_tensors = self.model_follow(f_features, params)

    loss = m_loss + p_intersection_loss + p_union_loss + f_loss
    tensors = dict()
    tensors['membership_logits'] = m_tensors['logits']
    tensors['membership_labels'] = m_labels
    tensors['intersection_logits'] = p_tensors['logits']
    tensors['intersection_labels'] = p_intersection_labels
    tensors['union_logits'] = p_tensors['logits']
    tensors['union_labels'] = p_union_labels
    tensors['set_follows_logits'] = f_tensors['logits']
    tensors['set_follows_labels'] = f_labels
    return loss, tensors

  def model_metaqa(self, features, params,
                   hop, top_k):
    """Compute logits for MetaQA multi-hop reasoning task.

    MetaQA model is made of 4 different parts:
    1. Construct KB triple query to retrieve the top k triples from KB.
    2. Check if retrieved triples are eligible by filtering them with sketch.
    3. Check if any retrieved triples should be excluded, i.e. if their
       object entities have been visited in the previous iterations.
    4. Update sketch and query embeddings from the next iteration.

    Args:
      features:
        -- question: a list of token ids for vocabs padded with -1
        -- question_entity_id: question entity id
        -- question_entity_sketch: a count-min sketch with only one element
                                   (the quesion entity)
        -- answer_labels: a k-hot vector marking if an entity is an answer
      params: hyper-parameters
      hop: number of hops
      top_k: select top_k facts at each iteration

    Returns:
      predictions and losses
    """
    tensors = {}
    question, question_entity_id, question_entity_sketch, answer_labels = features

    # Get text encoding of questions and question question entities.
    question_embs = self.compute_average_embs(question)
    # batch_size, hidden_size
    question_entity_embs = tf.nn.embedding_lookup(
        self.entity_embeddings_mat, question_entity_id)
    # batch_size, hidden_size

    # Set initial query embeddings and sketchs.
    query_entity_embs = question_entity_embs
    # batch_size, hidden_size
    query_entity_sketch = question_entity_sketch
    # batch_size, depth, width
    excluded_entity_sketch = question_entity_sketch
    # batch_size, depth, width

    # Begin iteration.
    for i in range(hop):
      # Construct queries by mapping original question embeddings to another
      # space. Queries are constructed as a concatenation of subject, relation,
      # and placeholders for object embeddings.
      with tf.variable_scope('question_emb_ffn_%s' % i):
        question_embs = tf.keras.layers.Dense(
            units=params['relation_emb_size'])(
                question_embs)
        # batch_size, hidden_size
        query_embs = tf.concat(
            [query_entity_embs, question_embs,
             tf.zeros_like(query_entity_embs)],
            axis=1)
        # batch_size, hidden_size * 3

      # Retrieve top k facts that match the queries, and check if the top
      # k facts are eligible by checking the count-min sketch. Subject
      # entities must have non-zero probability to be considered as
      # eligible.
      topk_fact_logits, topk_fact_ids = util.topk_search_fn(
          query_embs, self.kb_embeddings_mat, top_k)
      # batch_size, top_k     batch_size, top_k
      topk_subj_ids = tf.gather(self.all_fact_subjids, topk_fact_ids)
      # batch_size, topk
      is_topk_fact_eligible, _ = module.check_topk_fact_eligible(
          topk_subj_ids, query_entity_sketch, self.all_entity_sketches, params)

      # Entities visited before should also be excluded. Similarly, we first
      # get the probability from sketch and exclude those with non-zero
      # probability in previous iterations.
      topk_obj_ids = tf.gather(self.all_fact_objids, topk_fact_ids)
      # batch_size, topk
      is_topk_fact_excluded, _ = module.check_topk_fact_eligible(
          topk_obj_ids, excluded_entity_sketch,
          self.all_entity_sketches, params)

      # Apply the filtering results to logits of topk facts.
      if params['use_cm_sketch']:
        topk_fact_logits += (
            1 - is_topk_fact_eligible + is_topk_fact_excluded) * VERY_NEG
        # batch_size, top_k

      # Construct query embeddings for next iteration.
      topk_fact_obj_embs = tf.nn.embedding_lookup(
          self.entity_embeddings_mat, topk_obj_ids)
      # batch_size, top_k, hidden_size
      topk_softmax = tf.nn.softmax(topk_fact_logits)
      # batch_size, top_k, 1
      query_entity_embs = tf.reduce_sum(
          topk_fact_obj_embs * tf.expand_dims(topk_softmax, axis=2), axis=1)
      # batch_size, hidden_size

      # Update sketches.
      query_entity_sketch = module.create_cm_sketch(
          topk_obj_ids, topk_softmax, self.all_entity_sketches,
          cm_width=params['cm_width'])
      # batch_size, depth, width
      excluded_entity_sketch += query_entity_sketch
      # batch_size, depth, width

    # We only compute loss on the topk retrieval results at the last iteration.
    # No intermediate training signal is required.
    topk_fact_labels = tf.gather(
        answer_labels, topk_fact_ids, batch_dims=1, axis=1)
    topk_fact_loss = self.compute_tf_loss(topk_fact_logits, topk_fact_labels)
    tensors = {
        'logits': topk_fact_logits,
        'labels': topk_fact_labels,
        'candidates': topk_fact_ids
    }
    return topk_fact_loss, tensors

  def compute_average_embs(self, question):
    """Compute the text encoding of questions.

    We take a bag of word approach. Question encoding is an average pooling
    of word embeddings.

    Args:
      question: a list of token ids

    Returns:
      a tensor for question encoding
    """
    question_mask = tf.cast(tf.not_equal(question, -1), tf.int32)
    # batch_size, max_question_len
    question_mask_3d = tf.cast(
        tf.expand_dims(question_mask, axis=2), tf.float32)
    question_embs = tf.nn.embedding_lookup(
        self.word_embeddings_mat, question * question_mask)
    # batch_size, max_question_len, hidden_size
    question_embs = tf.reduce_sum(question_embs * question_mask_3d, axis=1)
    # batch_size, hidden_size
    question_len = tf.reduce_sum(question_mask, axis=1, keepdims=True)
    # batch_size, 1
    question_embs /= tf.cast(question_len, tf.float32)
    # batch_size, hidden_size
    return question_embs

  def model_webqsp(self, features, params,
                   top_k):
    """Compute logits for more evaluation.

    Args:
      features: features
      params: hyper-parameters
      top_k: top k to retrieve at each hop

    Returns:
      predictions and losses
    """

    tensors = dict()
    tensors['intermediate_logits'] = list()
    tensors['intermediate_labels'] = list()
    tensors['intermediate_objs'] = list()
    tensors['intermediate_answerable'] = list()

    (question_token_ids, segment_ids, question_mask, question_entity_id,
     question_entity_sketch, constraint_entity_id, constraint_entity_sketch,
     answer_ids) = features

    # Compute question embeddings and question entity embeddings.
    question_embs = self.compute_bert_cls_embs(question_token_ids, segment_ids,
                                               question_mask)
    # batch_size, bert_hidden_size
    question_entity_embs = tf.nn.embedding_lookup(self.entity_embeddings_mat,
                                                  question_entity_id)
    # batch_size, hidden_size

    # Initialize query embeddings before iteration.
    query_entity_embs = question_entity_embs
    # batch_size, hidden_size
    query_entity_sketch = question_entity_sketch
    # batch_size, depth, width

    # Questions in WebQuestionsSP are either one hop questions or
    # two hop questions where the intermediate entities is CVT
    # entities. In this experiment, we set the max hop to 2.
    for hid in range(2):
      # Compute query embeddings of relation by projecting the original
      # question embeddings into another space.
      with tf.variable_scope('question_emb_ffn_%d' % hid):
        query_relation_embs = tf.keras.layers.Dense(
            units=params['relation_emb_size'])(
                question_embs)
        # batch_size, hidden_size

      # We concatenate the subject, relation, and object embeddings to form
      # a query. Note that we set relation embeddings as 0, because
      # we observe that this could makes the initial retrieval process
      # more stable at training time. This retrieval will only return a fact id.
      # We will recompute the similarity score with non-zero query relation
      # embeddings which will eventually be used to compute logits.
      # Another possibility is to set a small coeffient, \alpha, with small
      # values in the beginning, and slightly increase it as training going.
      query_embs = tf.concat([
          query_entity_embs,
          tf.zeros_like(query_relation_embs),
          tf.zeros_like(query_entity_embs)
      ],
                             axis=1)
      # batch_size, hiddent_size * 3

      # Retrieve the topk facts and gather their subjects and objects.
      _, topk_fact_ids = util.topk_search_fn(
          query_embs, self.kb_embeddings_mat, top_k)
      # batch_size, topk
      topk_subj_ids = tf.gather(self.all_fact_subjids, topk_fact_ids)
      # batch_size, topk
      topk_obj_ids = tf.gather(self.all_fact_objids, topk_fact_ids)
      # batch_size, topk

      # We check if the retrieved triple is eligible. To do so, we check
      # if the subject of the triple passes the cm-sketch with non-zero
      # probability. The probability of entities in the sketch is computed
      # from the previous iterations (or directly initialized) as the
      # softmax of logits. To recover the logits from softmax, we take the
      # log values and further mask out those that are not eligible.
      is_topk_fact_eligible, topk_fact_prev_probs = \
          module.check_topk_fact_eligible(
              topk_subj_ids, query_entity_sketch,
              self.all_entity_sketches, params)
      # batch_size, topk
      topk_fact_prev_logits = tf.math.log(topk_fact_prev_probs + 1e-6)
      # batch_size, topk
      topk_fact_prev_logits += (1.0 - is_topk_fact_eligible) * VERY_NEG
      # batch_size, topk
      with tf.variable_scope('topk_fact_eligibility_%d' % hid):
        # We learn another bias term here. This helps to adjust how
        # significant signals from previous iterations contribute to
        # later predictions.
        topk_fact_prev_logit_bias = tf.get_variable(
            name='topk_fact_logit_bias',
            dtype=tf.float32,
            shape=[1, 1],
            initializer=tf.random_normal_initializer(),
            trainable=True)
        topk_fact_prev_logits += topk_fact_prev_logit_bias

      # Now, we take the full fact embedding and compute the similarity
      # score between query and facts (with query relation embedding). We
      # further added the logits from previous iterations (after passing
      # the sketch) if we would like to use cm sketch.
      query_embs = tf.concat(
          [tf.zeros_like(query_entity_embs), query_relation_embs], axis=1)
      # batch_size, hidden_size * 2
      topk_fact_logit = self.compute_topk_fact_logits(topk_fact_ids, query_embs,
                                                      params)
      # batch_size, topk
      if params['use_cm_sketch']:
        topk_fact_logit += topk_fact_prev_logits
      # batch_size, topk

      # We filter the logits of CVT and regular entities. In the WebQuestionsSP
      # dataset, questions are either 1 hop questions or 2 hop questions
      # that travel through a CVT node. Thus, logits of regular entities
      # are considered to be answer candidates. Logits of CVT entities
      # will be passed to the next hop.
      # To distinguish CVT nodes, we could add another bit to entity embeddings
      # which we refer to as "hard" type. For simplicity, we use another
      # vector to store such information (just not appended to the end of
      # the embedding table).
      is_topk_obj_cvt = tf.gather(self.all_entity_is_cvt, topk_obj_ids)
      # batch_size, topk
      topk_ent_fact_logit = topk_fact_logit + is_topk_obj_cvt * VERY_NEG
      # batch_size, topk
      topk_cvt_fact_logit = topk_fact_logit + (1.0 - is_topk_obj_cvt) * VERY_NEG
      # batch_size, topk
      tensors['intermediate_logits'].append(topk_ent_fact_logit)
      tensors['intermediate_objs'].append(topk_obj_ids)

      # Then we compute the new query embedding and cm-sketch for the next
      # iteration.
      query_entity_embs, query_entity_sketch = \
          self.compute_query_embs_and_sketch(
              topk_obj_ids, topk_cvt_fact_logit, params)

      # Finally, we check if any of the retrieved facts actually contain
      # the correct answer. We treated them as labels and store them
      # for future use. We also compute how many percent of questions are
      # retrieved but not correctly answered. This is the upper bound of
      # the performance of our model.
      _, topk_fact_labels = util.compute_x_in_set(topk_obj_ids, answer_ids)
      topk_fact_labels = tf.cast(topk_fact_labels, dtype=tf.float32)
      # batch_size, topk
      topk_fact_labels *= is_topk_fact_eligible
      # batch_size, topk
      _, topk_objs_in_answers = util.compute_x_in_set(topk_obj_ids, answer_ids)
      # batch_size, topk
      topk_objs_in_answers = tf.logical_and(topk_objs_in_answers,
                                            is_topk_fact_eligible > 0.0)
      # batch_size, topk
      topk_objs_in_answers = tf.reduce_any(topk_objs_in_answers, axis=1)
      # batch_size
      tensors['intermediate_labels'].append(topk_fact_labels)
      tensors['intermediate_answerable'].append(topk_objs_in_answers)

    # After a few iterations, we concatenate the logits, labels and predictions
    # to unify the retrieval and prediction results of all iterations. The
    # concatenated results will be used for final prediction.
    concat_topk_obj_ids = tf.concat(tensors['intermediate_objs'], axis=1)
    # batch_size, topk * 2
    concat_topk_fact_logit = tf.concat(tensors['intermediate_logits'], axis=1)
    # batch_size, topk * 2
    concat_topk_fact_labels = tf.concat(tensors['intermediate_labels'], axis=1)
    # batch_size, topk * 2

    # We observe that there are ties between top predicted facts. They share
    # similar prediction scores but only a few of them satisfy the constraint.
    # Thus, we implement a constraint module to discriminate those entities.
    # We first compute an average of entity embeddings of tied facts.
    # Constraint entity embeddings are directly loaded from embedding table.
    concat_topk_fact_best_logit = tf.reduce_max(
        concat_topk_fact_logit, axis=1, keepdims=True)
    # batch_size, 1
    filtered_concat_topk_fact_best_logit = tf.cast(
        tf.equal(concat_topk_fact_logit, concat_topk_fact_best_logit),
        dtype=tf.float32)
    # batch_size, topk * 2
    concat_topk_subj_query_embs, concat_topk_subj_sketches = \
        self.compute_query_embs_and_sketch(
            concat_topk_obj_ids, filtered_concat_topk_fact_best_logit, params)
    # batch_size, topk * 2, hidden_size
    constraint_entity_embs = util.embedding_lookup_with_padding(
        self.entity_embeddings_mat, constraint_entity_id)
    # batch_size, hidden_size
    with tf.variable_scope('question_emb_ffn_constraint'):
      # Project question embeddings to get query relation embeddings.
      constraint_relation_embs = tf.keras.layers.Dense(
          units=params['relation_emb_size'])(
              question_embs)
      # batch_size, hidden_size
    constraint_query_embs = tf.concat([
        concat_topk_subj_query_embs,
        tf.zeros_like(constraint_relation_embs), constraint_entity_embs
    ],
                                      axis=1)
    # batch_size, hiddent_size * 3
    constraint_topk_fact_logits, constraint_topk_fact_ids = \
        util.topk_search_fn(constraint_query_embs, self.kb_embeddings_mat,
                            top_k)
    # batch_size, topk

    # Similar as previous retrieval steps, we check if retrieved facts for
    # constraints are eligible. We mask out logits of ineligible facts.
    constraint_topk_subj_ids = tf.gather(self.all_fact_subjids,
                                         constraint_topk_fact_ids)
    # batch_size, topk
    is_constraint_topk_subj_eligible, _ = module.check_topk_fact_eligible(
        constraint_topk_subj_ids, concat_topk_subj_sketches,
        self.all_entity_sketches, params)
    # batch_size, topk
    constraint_topk_obj_ids = tf.gather(self.all_fact_objids,
                                        constraint_topk_fact_ids)
    # batch_size, topk
    is_constraint_topk_obj_eligible, _ = module.check_topk_fact_eligible(
        constraint_topk_obj_ids, constraint_entity_sketch,
        self.all_entity_sketches, params)
    # batch_size, topk
    is_constraint_topk_eligible = tf.minimum(is_constraint_topk_subj_eligible,
                                             is_constraint_topk_obj_eligible)
    # batch_size, topk
    constraint_topk_fact_logits += (1.0 -
                                    is_constraint_topk_eligible) * VERY_NEG
    # batch_size, topk
    constraint_topk_fact_logits = tf.nn.relu(constraint_topk_fact_logits)
    # batch_size, topk

    # We need to add the logits from constraints to the logits at the
    # end of reasoning. However, the order of subject and object entities
    # does not match. We first find the mapping, map the logits, and add
    # it to the original logits.
    constraint_topk_fact_logits_mapped_to_concat = self.map_constraint_logits(
        concat_topk_obj_ids, constraint_topk_subj_ids,
        constraint_topk_fact_logits)
    concat_topk_fact_logit_with_constraint = concat_topk_fact_logit + \
        constraint_topk_fact_logits_mapped_to_concat
    # batch_size, topk * 2

    # Finally compute loss, cache a few tensors.
    answerable_at_topk = tf.metrics.mean(
        tf.cast(
            tf.logical_or(tensors['intermediate_answerable'][0],
                          tensors['intermediate_answerable'][1]),
            dtype=tf.float32))
    loss = self.compute_tf_loss(concat_topk_fact_logit, concat_topk_fact_labels)
    tensors['logits'] = concat_topk_fact_logit_with_constraint
    tensors['labels'] = concat_topk_fact_labels
    tensors['answerable_at_topk'] = answerable_at_topk

    return loss, tensors

  def compute_bert_cls_embs(self, token_ids, segment_ids,
                            masks):
    """Return the embedding of CLS token in BERT.

    Args:
      token_ids: BERT token ids
      segment_ids: BERT segment ids
      masks: BERT mask

    Returns:
      BERT CLS embedding
    """
    bert_outputs = self.bert_module(
        inputs={
            'input_ids': token_ids,
            'segment_ids': segment_ids,
            'input_mask': masks
        },
        signature='tokens',
        as_dict=True)
    cls_embs = bert_outputs['sequence_output']
    # batch_size, max_seq_length, hidden_size
    cls_embs = cls_embs[:, 0, :]
    # batch_size, hidden_size
    return cls_embs

  def compute_query_embs_and_sketch(
      self, entity_ids, logits,
      params):
    """Compute embeddings and sketch with logits of entities.

    Given entity_ids and logits in the same order, compute weighted average
    of entity embeddings and a sketch that stores their weights.

    Args:
      entity_ids: entity ids
      logits: logits before softmax
      params: params

    Returns:
      A weighted average of embeddings and a cm sketch for the entities
    """
    topk_softmax = tf.nn.softmax(logits)
    # batch_size, topk
    topk_ent_embs = tf.nn.embedding_lookup(self.entity_embeddings_mat,
                                           entity_ids)
    # batch_size, topk, hidden_size
    query_entity_embs = tf.reduce_sum(
        topk_ent_embs * tf.expand_dims(topk_softmax, axis=2), axis=1)
    # batch_size, hidden_size
    query_entity_sketch = module.create_cm_sketch(
        entity_ids, topk_softmax, self.all_entity_sketches,
        cm_width=params['cm_width'])
    # batch_size, depth, width
    return query_entity_embs, query_entity_sketch

  def compute_topk_fact_logits(self, topk_fact_idx,
                               query_embs,
                               params):
    """Get the logits between query and facts with topk_fact_idx.

    Args:
      topk_fact_idx: topk ids from scam -- batch_size, topk
      query_embs: embeddings of query -- batch_size, hidden_size * 2
      params: flags

    Returns:
      topk_fact_embs, topk_mask, topk_labels
    """
    topk_subj_ids = tf.gather(self.all_fact_subjids, topk_fact_idx)
    # batch_size, topk
    topk_rel_ids = tf.gather(self.all_fact_relids, topk_fact_idx)
    # batch_size, topk
    topk_mask = tf.cast(topk_subj_ids >= 0, dtype=tf.int32)
    # batch_size, topk

    topk_subj_embs = tf.nn.embedding_lookup(self.entity_embeddings_mat,
                                            topk_subj_ids * topk_mask)
    # batch_size, topk, hidden_size
    topk_rel_embs = tf.nn.embedding_lookup(self.relation_embeddings_mat,
                                           topk_rel_ids * topk_mask)
    # batch_size, topk, hidden_size

    # compute logits for top k facts
    topk_fact_embs = tf.concat((topk_subj_embs, topk_rel_embs), axis=2)
    # batch_size, topk, hidden_size * 2
    topk_fact_embs *= tf.cast(
        tf.expand_dims(topk_mask, axis=2), dtype=tf.float32)
    # batch_size, topk, hidden_size * 2

    query_embs = tf.expand_dims(query_embs, axis=2)
    # batch_size, hidden_size * 2, 1
    topk_fact_logit = tf.matmul(topk_fact_embs, query_embs) / tf.sqrt(
        float(params['entity_emb_size'] + params['relation_emb_size']))
    topk_fact_logit = tf.squeeze(topk_fact_logit, axis=2)
    # batch_size, topk

    return topk_fact_logit

  def map_constraint_logits(self, original_ids,
                            constraint_ids,
                            constraint_logits):
    """Map constraint logits to original if ids match.

    constraint_logits is logits for constraint_ids. If an id in constrain_ids
    also appear in original_ids, we will map its logit to the corresponding
    position as in original_ids.

    Args:
      original_ids: order to map to
      constraint_ids: order to map from
      constraint_logits: logits in the same order as constraint_ids

    Returns:
      logits in the order of original_ids if exist, otherwise 0.
    """
    constraint_logits_mapping, _ = util.compute_x_in_set(
        constraint_ids, original_ids)
    # batch_size, topk * 2, topk
    constraint_logits_mapping = tf.cast(
        constraint_logits_mapping, dtype=tf.float32)
    # batch_size, topk * 2, topk
    mapped_constraint_logits = tf.matmul(
        constraint_logits_mapping, tf.expand_dims(constraint_logits, axis=2))
    # batch_size, topk * 2, 1
    mapped_constraint_logits = tf.squeeze(mapped_constraint_logits, axis=2)
    # batch_size, topk * 2
    return mapped_constraint_logits

  def compute_logits(self, candidates, query,
                     hidden_size):
    """Compute logits between query embedding and candidate embeddings.

    Args:
      candidates: candidate embeddings
      query: query embeddings
      hidden_size: hidden size

    Returns:
      L2 logits and MIPS logits
    """

    l2_logits = - tf.reduce_sum(
        tf.square(candidates - query), axis=2, keepdims=True)
    # batch_size, num_candidate, 1
    l2_logits = l2_logits / tf.sqrt(float(hidden_size))
    # batch_size, num_candidate, 1
    l2_logits = tf.squeeze(l2_logits, axis=2)
    # batch_size, num_candidate

    mips_logits = tf.matmul(
        candidates, query, transpose_b=True) / tf.sqrt(float(hidden_size))
    # batch_size, num_candidate, 1
    mips_logits = tf.squeeze(mips_logits, axis=2)
    # batch_size, num_candidate
    return mips_logits, l2_logits

  def compute_tf_loss(self, logits, labels):
    """Compute loss between logits and labels.

    Args:
      logits: batch_size, num_candidate
      labels: batch_size, num_candidate

    Returns:
      loss and train_op
    """
    labels = tf.cast(labels, tf.float32)
    # If labels contains all zeros, replace them with all ones
    is_label_all_zeros = tf.cast(tf.reduce_all(
        tf.equal(labels, 0.0), axis=1, keepdims=True), dtype=tf.float32)
    # batch_size, 1
    padded_labels = tf.tile(is_label_all_zeros, [1, labels.shape[1]])
    # batch_size, num_candidate
    labels += padded_labels
    # batch_size, num_candidate

    # Also zero out logits if their labels are all zeros.
    logits *= (1 - is_label_all_zeros)
    # batch_size, num_candidate

    labels_sum = tf.reduce_sum(labels, axis=1, keepdims=True)
    labels = labels / labels_sum
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    return loss

  def create_one_hot_sketch(self, element, all_sketches, cm_width):
    """Wrapper to create the initial sketch for an one-hot vector."""
    element = tf.expand_dims(element, axis=1)
    softmax = tf.ones_like(element, dtype=tf.float32)
    return module.create_cm_sketch(element, softmax, all_sketches, cm_width)

  def simple_follow_op(self, params, ent_embs, rel_embs,
                       ent_sketch, rel_sketch):
    """Wrapper of follow operation inside EmQL."""
    if not params['use_cm_sketch']:
      ent_sketch, rel_sketch = None, None

    object_embs, object_sketch, unused_topk_fact_logits = module.follow_op(
        params, ent_embs, ent_sketch, rel_embs, rel_sketch,
        self.all_fact_subjids, self.all_fact_relids, self.all_fact_objids,
        self.entity_embeddings_mat, self.kb_embeddings_mat,
        self.all_entity_sketches, self.all_relation_sketches)
    return object_embs, object_sketch

  def simple_entity_encode_op(self, params, ent):
    """Wrapper to encode a single entity into its EmQL representation."""
    set_ids = tf.expand_dims(ent, axis=1)
    set_mask = tf.ones_like(set_ids, dtype=tf.float32)
    return module.encode_op(params, set_ids, set_mask,
                            self.entity_embeddings_mat,
                            self.all_entity_sketches)

  def simple_relation_encode_op(self, params, rel):
    """Wrapper to encode a single relation into its EmQL representation."""
    set_ids = tf.expand_dims(rel, axis=1)
    set_mask = tf.ones_like(set_ids, dtype=tf.float32)
    return module.encode_op(params, set_ids, set_mask,
                            self.relation_embeddings_mat,
                            self.all_relation_sketches)

  def model_query2box(self, name, features,
                      params):
    """Model query2box for 9 different tasks.

    This function simulate 9 compositional queries as defined in the
    Query2Box paper:
      1c: X.follow(R)
      2c: X.follow(R1).follow(R2)
      3c: X.follow(R1).follow(R2).follow(R3)
      2i: X1.follow(R1) & X2.follow(R2)
      3i: X1.follow(R1) & X2.follow(R2) & X3.follow(R3)
      ic: (X1.follow(R1) & X2.follow(R2)).follow(R3)
      ci: X1.follow(R1) & X2.follow(R2).follow(R3)
      2u: X1.follow(R1) | X2.follow(R2)
      uc: (X1.follow(R1) | X2.follow(R2)).follow(R3)
    Modules (follow, intersection, union, decode) are defined in module.py.

    Args:
      name: name of the task, e.g. "query2box_2i".
      features: a tuple of tf.Tensors. The order is decided by each task.
      params: a dictionary of hyper-parameters

    Returns:
      loss and predictions
    """
    task = name.split('_')[-1]

    if task == '1c':  # X.follow(R)
      ent, rel1 = features
      ent_embs, ent_sketch = self.simple_entity_encode_op(params, ent)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)

      answer_embs, answer_sketch = self.simple_follow_op(
          params, ent_embs, rel1_embs, ent_sketch, rel1_sketch)

    elif task == '2c':  # X.follow(R1).follow(R2)
      ent, rel1, rel2 = features
      ent_embs, ent_sketch = self.simple_entity_encode_op(params, ent)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent_embs, rel1_embs, ent_sketch, rel1_sketch)
      answer_embs, answer_sketch = self.simple_follow_op(
          params, obj1_embs, rel2_embs, obj1_sketch, rel2_sketch)

    elif task == '3c':  # X.follow(R1).follow(R2).follow(R3)
      ent, rel1, rel2, rel3 = features
      ent_embs, ent_sketch = self.simple_entity_encode_op(params, ent)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)
      rel3_embs, rel3_sketch = self.simple_relation_encode_op(params, rel3)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent_embs, rel1_embs, ent_sketch, rel1_sketch)
      obj2_embs, obj2_sketch = self.simple_follow_op(
          params, obj1_embs, rel2_embs, obj1_sketch, rel2_sketch)
      answer_embs, answer_sketch = self.simple_follow_op(
          params, obj2_embs, rel3_embs, obj2_sketch, rel3_sketch)

    elif task == '2i' or task == '2u':  # X1.follow(R1) & / | X2.follow(R2)
      ent1, rel1, ent2, rel2 = features
      ent1_embs, ent1_sketch = self.simple_entity_encode_op(params, ent1)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      ent2_embs, ent2_sketch = self.simple_entity_encode_op(params, ent2)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent1_embs, rel1_embs, ent1_sketch, rel1_sketch)
      obj2_embs, obj2_sketch = self.simple_follow_op(
          params, ent2_embs, rel2_embs, ent2_sketch, rel2_sketch)
      if task == '2i':
        answer_embs, answer_sketch = module.intersection_op(
            obj1_embs, obj1_sketch, obj2_embs, obj2_sketch)
      elif task == '2u':
        answer_embs, answer_sketch = module.union_op(
            obj1_embs, obj1_sketch, obj2_embs, obj2_sketch)

    elif task == '3i':  # X1.follow(R1) & X2.follow(R2) & X3.follow(R3)
      ent1, rel1, ent2, rel2, ent3, rel3 = features
      ent1_embs, ent1_sketch = self.simple_entity_encode_op(params, ent1)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      ent2_embs, ent2_sketch = self.simple_entity_encode_op(params, ent2)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)
      ent3_embs, ent3_sketch = self.simple_entity_encode_op(params, ent3)
      rel3_embs, rel3_sketch = self.simple_relation_encode_op(params, rel3)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent1_embs, rel1_embs, ent1_sketch, rel1_sketch)
      obj2_embs, obj2_sketch = self.simple_follow_op(
          params, ent2_embs, rel2_embs, ent2_sketch, rel2_sketch)
      obj3_embs, obj3_sketch = self.simple_follow_op(
          params, ent3_embs, rel3_embs, ent3_sketch, rel3_sketch)
      answer_embs, answer_sketch = module.intersection_op(
          obj1_embs, obj1_sketch, obj2_embs, obj2_sketch)
      answer_embs, answer_sketch = module.intersection_op(
          answer_embs, answer_sketch, obj3_embs, obj3_sketch)

    elif task == 'ic' or task == 'uc':
      # (X1.follow(R1) & / | X2.follow(R2)).follow(R3)
      ent1, rel1, ent2, rel2, rel3 = features
      ent1_embs, ent1_sketch = self.simple_entity_encode_op(params, ent1)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      ent2_embs, ent2_sketch = self.simple_entity_encode_op(params, ent2)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)
      rel3_embs, rel3_sketch = self.simple_relation_encode_op(params, rel3)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent1_embs, rel1_embs, ent1_sketch, rel1_sketch)
      obj2_embs, obj2_sketch = self.simple_follow_op(
          params, ent2_embs, rel2_embs, ent2_sketch, rel2_sketch)
      if task == 'ic':
        answer_embs, answer_sketch = module.intersection_op(
            obj1_embs, obj1_sketch, obj2_embs, obj2_sketch)
      elif task == 'uc':
        answer_embs, answer_sketch = module.union_op(
            obj1_embs, obj1_sketch, obj2_embs, obj2_sketch)
      answer_embs, answer_sketch = self.simple_follow_op(
          params, answer_embs, rel3_embs, answer_sketch, rel3_sketch)

    elif task == 'ci':  # X1.follow(R1) & X2.follow(R2).follow(R3)
      ent1, rel1, rel2, ent2, rel3 = features
      ent1_embs, ent1_sketch = self.simple_entity_encode_op(params, ent1)
      rel1_embs, rel1_sketch = self.simple_relation_encode_op(params, rel1)
      rel2_embs, rel2_sketch = self.simple_relation_encode_op(params, rel2)
      ent2_embs, ent2_sketch = self.simple_entity_encode_op(params, ent2)
      rel3_embs, rel3_sketch = self.simple_relation_encode_op(params, rel3)

      obj1_embs, obj1_sketch = self.simple_follow_op(
          params, ent1_embs, rel1_embs, ent1_sketch, rel1_sketch)
      obj2_embs, obj2_sketch = self.simple_follow_op(
          params, obj1_embs, rel2_embs, obj1_sketch, rel2_sketch)
      obj3_embs, obj3_sketch = self.simple_follow_op(
          params, ent2_embs, rel3_embs, ent2_sketch, rel3_sketch)
      answer_embs, answer_sketch = module.intersection_op(
          obj2_embs, obj2_sketch, obj3_embs, obj3_sketch)

    else:
      raise ValueError('task: %s not recognized' % task)

    # Decode from set representation to a list of entities. We will apply a
    # null sketch for decoding at query time.
    answer_sketch = None
    answer_ids, unused_answer_logits = module.decode_op(
        params, answer_embs, answer_sketch,
        self.entity_embeddings_mat, self.all_entity_sketches)

    return tf.constant(0.0), {'answer_ids': answer_ids}

  ##############################################################
  ######################### Evaluation #########################
  ##############################################################

  def run_tf_evaluation(self, logits, labels,
                        prefix = ''):
    """Compute evaluation metrics.

    Args:
      logits: batch_size, num_candidate
      labels: batch_size, num_candidate
      prefix: prefix for evaluation key name

    Returns:
      evaluation metrics
    """
    hits_at_one = util.compute_hits_at_k(logits, labels, k=1)
    hits_at_five = util.compute_hits_at_k(logits, labels, k=5)

    recall_at_one = util.compute_recall_at_k(logits, labels, k=1)
    recall_at_five = util.compute_recall_at_k(logits, labels, k=5)

    average_precision_at_5 = \
        util.compute_average_precision_at_k(logits, labels, k=5)

    evaluations = {
        prefix + 'hits@1': tf.metrics.mean(hits_at_one),
        prefix + 'hits@5': tf.metrics.mean(hits_at_five),
        prefix + 'recall@1': tf.metrics.mean(recall_at_one),
        prefix + 'recall@5': tf.metrics.mean(recall_at_five),
        prefix + 'map@5': tf.metrics.mean(average_precision_at_5),
    }

    return evaluations

  def get_tf_prediction(self, name, features, tensors):
    # raise NotImplementedError
    if name.startswith('query2box'):
      return {
          'query': tf.concat(
              [tf.expand_dims(f, axis=-1) for f in features], axis=-1),
          'answer_ids': tensors['answer_ids']
      }
    else:
      raise ValueError


def build_model_fn(
    name, data_loader, eval_name,
    eval_metric_at_k):
  """Build model function.

  Args:
    name: name of the model -- 'membership' or 'intersection' or 'union'
    data_loader: data loader
    eval_name: if model contains several sub-models
    eval_metric_at_k: top k for evaluation metrics

  Returns:
    model function
  """
  del eval_name, eval_metric_at_k

  def model_fn(features, labels, mode,  # pylint: disable=unused-argument
               params):
    """Wrapper function to select model mode.

    This function is called by tf.estimator.train_and_evaluate function in the
    background.

    Args:
      features: features
      labels: unused labels
      mode: tf.estimator.ModeKeys.PREDICT or TRAIN or EVAL
      params: extra params

    Returns:
      A tf.estimator spec

    """
    emql = EmQL(name, params, data_loader)

    # Define behaviors for different operationstrain / eval / pred
    if mode == tf_estimator.ModeKeys.TRAIN:
      loss, tensors = emql.get_tf_model(name, features, params)
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      gvs = optimizer.compute_gradients(loss)
      capped_gvs = [(tf.clip_by_norm(grad, params['gradient_clip']), var)
                    for grad, var in gvs if grad is not None]
      train_op = optimizer.apply_gradients(
          capped_gvs, global_step=tf.train.get_global_step())
      return tf_estimator.EstimatorSpec(
          mode=mode, train_op=train_op, loss=loss)

    elif mode == tf_estimator.ModeKeys.EVAL:
      loss, tensors = emql.get_tf_model(name, features, params)

      if name == 'mixture':
        evaluations = dict()
        evaluations.update(emql.run_tf_evaluation(
            tensors['membership_logits'], tensors['membership_labels'],
            prefix='membership_'))
        evaluations.update(emql.run_tf_evaluation(
            tensors['intersection_logits'], tensors['intersection_labels'],
            prefix='intersection_'))
        evaluations.update(emql.run_tf_evaluation(
            tensors['union_logits'], tensors['union_labels'],
            prefix='union_'))
        evaluations.update(emql.run_tf_evaluation(
            tensors['set_follows_logits'], tensors['set_follows_labels'],
            prefix='set_follows_'))
      else:
        evaluations = emql.run_tf_evaluation(
            tensors['logits'], tensors['labels'])
      return tf_estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=evaluations)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      loss, tensors = emql.get_tf_model(name, features, params)
      predictions = emql.get_tf_prediction(name, features, tensors)
      return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

    else:
      raise ValueError('illegal mode %r' % mode)

  return model_fn

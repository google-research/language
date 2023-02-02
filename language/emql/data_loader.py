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
"""Data loader for EmQL."""

import json
import pickle
import random


from absl import flags
from language.emql import cm_sketch
from language.emql import util
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile
from tqdm import tqdm

FLAGS = flags.FLAGS

# seeds
SEED = 3
random.seed(SEED)
np.random.seed(SEED)

# other const
TRAIN_SPLIT = 0.8
DATA_PER_EPOCH = 10000


class DataLoader(object):
  r"""Load and construct training data for EmQL.

  We first load KB and vocab and then construct training/testing datasets
  from the KB. Datasets are constructed in the funnction
  construct_dataset() with different task names. We further convert
  it to tf.Dataset in the function build_input_fn().

  Input data is a plain text file where each line is a KB triple.
  Subject, relation, and object are separated by '\t'. For example:

  Inception \t directed_by \t Christopher_Nolan \n
  Pittsburgh \t located_in \t Pennsylvania \n

  The KB is stored in a dictionary of dictionaries. The first level has
  subject entities as it keys, and the second level has relations as keys.
  The value in the second level is a set of object entities. For example,

    {'Pittsburgh': {'located_in': set(['Pennsylvania'])}}

  entity2id and relation2id store a mapping from entities and relations to
  their ids. For example,

    {'Pittsburgh': 0, 'Pensylvania': 1, ...}
    {'located_in': 0, 'directed_by': 1, ...}

  """

  def __init__(self,
               params,
               name,
               root_dir,
               kb_file,
               vocab_file = None):
    # Load knowledge base.
    self.root_dir = root_dir
    self.max_set = params['max_set']
    self.kb, self.fact2id, self.entity2id, self.relation2id = \
        self.load_kb(name, root_dir + kb_file)
    self.id2entity = {idx: ent for ent, idx in self.entity2id.items()}
    self.id2fact = {idx: fact for fact, idx in self.fact2id.items()}
    self.id2relation = {idx: rel for rel, idx in self.relation2id.items()}
    self.num_entities = len(self.entity2id)
    self.num_relations = len(self.relation2id)
    self.num_facts = len(self.fact2id)

    # Load vocab.
    if vocab_file is not None:
      self.word2id = self.load_vocab(self.root_dir + vocab_file)
      self.id2word = {idx: word for word, idx in self.word2id.items()}
      self.num_vocab = len(self.word2id)

    # Construct count min sketch for entities in the KB.
    self.cm_depth = params['cm_depth']
    self.cm_width = params['cm_width']
    self.cm_context = cm_sketch.CountMinContext(
        depth=self.cm_depth, width=self.cm_width, n=self.num_entities)
    self.cm_context_rel = cm_sketch.CountMinContext(
        depth=self.cm_depth, width=self.cm_width, n=self.num_relations)

    # Record fact info in numpy array.
    self.all_entity_ids = np.arange(self.num_entities)
    self.all_entity_sketches = np.array(
        [self.cm_context.get_hashes(i) for i in range(self.num_entities)])
    self.all_relation_sketches = np.array(
        [self.cm_context_rel.get_hashes(i) for i in range(self.num_relations)])
    self.all_fact_subjids = np.array(
        [self.entity2id[self.id2fact[i][0]] for i in range(self.num_facts)])
    self.all_fact_relids = np.array(
        [self.relation2id[self.id2fact[i][1]] for i in range(self.num_facts)])
    self.all_fact_objids = np.array(
        [self.entity2id[self.id2fact[i][2]] for i in range(self.num_facts)])

    # Construct training data for different datasets
    self.construct_dataset(name, train_split=TRAIN_SPLIT)

  def load_kb(
      self, name, kb_file
  ):
    r"""Load KB from file.

    We store KB as a dictionary of dictionaries, where keys are subjects and
    values are dictionaries of facts. Each dictionary of facts is constructed
    with relations as keys and objects as values.

    Input formats are different in different tasks. In the "webqsp" task,
    subject, relation, and object entities are split by "\t".

    Args:
      name: task name
      kb_file: kb file

    Returns:
      a dictionary of dictionaries, and mappings from fact/relation/entity to id
    """

    def get_delimiter():
      if name == 'webqsp' or name.startswith('query2box'):
        return '\t'
      else:
        return '|'

    def set_has_inverse_fact():
      if name == 'webqsp':
        return False
      else:
        return True

    def set_reorder():
      if name == 'webqsp':
        return lambda ps: (ps[1], ps[0], ps[2])
      else:
        return lambda ps: ps

    def set_inv_suffix():
      if name.startswith('query2box'):
        suffix = '_reverse'
      else:
        suffix = '-inv'
      return suffix

    # Prepare for loading the KB.
    self._inv_suffix = set_inv_suffix()
    delimiter = get_delimiter()
    has_inverse_fact = set_has_inverse_fact()
    reorder = set_reorder()

    kb = dict()
    fact2id = dict()
    entity2id = dict()
    relation2id = dict()
    with gfile.GFile(kb_file, 'r') as f_in:
      for line in tqdm(f_in):
        subj, rel, obj = reorder(tuple(line.strip().split(delimiter)))
        rel_inv = self._inverse_rel(rel)

        if (subj, rel, obj) not in fact2id:
          fact2id[(subj, rel, obj)] = len(fact2id)
          if has_inverse_fact:
            assert (obj, rel_inv, subj) not in fact2id
            fact2id[(obj, rel_inv, subj)] = len(fact2id)

        # Load KB facts.
        if subj not in kb:
          kb[subj] = dict()
        if rel not in kb[subj]:
          kb[subj][rel] = set()
        kb[subj][rel].add(obj)

        # Build a mapping from relations and entities to ids.
        if rel not in relation2id:
          relation2id[rel] = len(relation2id)
        if subj not in entity2id:
          entity2id[subj] = len(entity2id)
        if obj not in entity2id:
          entity2id[obj] = len(entity2id)

        if has_inverse_fact:
          # assert (obj, rel_inv, subj) not in fact2id
          # fact2id[(obj, rel_inv, subj)] = len(fact2id)
          # Load reverse KB facts.
          if obj not in kb:
            kb[obj] = dict()
          if rel_inv not in kb[obj]:
            kb[obj][rel_inv] = set()
          kb[obj][rel_inv].add(subj)
          # Build a mapping from inverse relations to ids.
          if rel_inv not in relation2id:
            relation2id[rel_inv] = len(relation2id)

    return kb, fact2id, entity2id, relation2id

  def load_vocab(self, vocab_file):
    """Load vocab from file.

    Args:
      vocab_file: vocab file

    Returns:
      a mapping from word to id
    """
    word2id = dict()
    with gfile.GFile(vocab_file, 'r') as f_in:
      for line in tqdm(f_in):
        word = json.loads(line)
        assert word not in word2id
        word2id[word] = len(word2id)
    return word2id

  ##############################################################
  #################### Construct Data ##########################
  ##############################################################

  def construct_dataset(self, name, train_split):
    """Construct datasets for training and testing.

    Args:
      name: name of the learning task
      train_split: percentage of data for training
    """
    if name in ['membership', 'mixture']:
      self.train_data_membership, self.test_data_membership = \
          self.construct_membership(train_split)
    if name in ['intersection', 'union', 'mixture']:
      self.train_data_set_pair, self.test_data_set_pair = \
          self.construct_set_pairs(train_split)
    if name == 'follow':
      self.train_data_follow, self.test_data_follow = \
          self.construct_follow_facts(train_split)
    if name in ['set_follow', 'mixture']:
      self.train_data_follow, self.test_data_follow = \
          self.construct_set_follow_facts(train_split)
    if name in ['metaqa2', 'metaqa3']:
      self.train_data_metaqa, self.test_data_metaqa = \
          self.construct_metaqa_data(
              train_file='train.json', test_file='test.json')
    if name in ['webqsp']:
      self.bert_tokenizer = util.BertTokenizer()
      self.train_data_webqsp, self.test_data_webqsp = self.construct_webqsp(
          train_file='train.json', test_file='test.json')
      self.all_entity_is_cvt = self.check_cvt_entity()
    if name.startswith('query2box'):
      self.q2b_id2entity = pickle.load(
          gfile.GFile(self.root_dir + 'ind2ent.pkl', 'rb'))
      self.q2b_id2relation = pickle.load(
          gfile.GFile(self.root_dir + 'ind2rel.pkl', 'rb'))
      self.train_data_query2box, self.test_data_query2box = \
          self.construct_query2box(name)

  def construct_membership(
      self, train_split):
    """Construct sets from KB.

    A natural set is defined as a set of entities that share the same
    property, e.g. movies directed by Nolan, or actors appear in Inception.

    Args:
      train_split: percentage of data for training

    Returns:
      a list of train sets and test sets
    """
    train_data = list()
    test_data = list()

    for unused_subj_id, subj in self.id2entity.items():
      all_objs = set()
      for _, objs in self.kb[subj].items():
        all_objs.update(objs)
      # Group entities by graph structure. Entities within distance one
      # are considered as an element in the set.
      if len(all_objs) > 1:
        one_data = all_objs
      else:
        continue

      if random.random() < train_split:
        train_data.append(one_data)
      else:
        test_data.append(one_data)

    tf.logging.info('set examples: %d train, %d test', len(train_data),
                    len(test_data))
    return train_data, test_data

  def construct_set_pairs(
      self, train_split):
    """Construct pairs of sets from KB with at least one entity in common.

    Args:
      train_split: percentage of data for training

    Returns:
      a list of train pairs and test pairs
    """
    random.seed(SEED)
    subj2objs = dict()
    for subj, facts in self.kb.items():
      all_objs = set()
      for unused_rel, objs in facts.items():
        all_objs.update(objs)
      subj2objs[subj] = all_objs

    train_pairs = list()
    test_pairs = list()
    for target, objs in subj2objs.items():
      if len(objs) <= 1:
        continue
      # Find candidate sets which have at least 2 entities.
      candidate_sets = list()
      for obj in objs:
        for unused_rel, subjs in self.kb[obj].items():
          if len(subjs) > 1 and target in subjs:
            candidate_sets.append(subjs)
      # Randomly pick some pair of sets.
      num_pairs = int(len(candidate_sets) / 2)
      for i in range(num_pairs):
        set1 = candidate_sets[2 * i]
        set2 = candidate_sets[2 * i + 1]
        if random.random() < train_split:
          train_pairs.append((set1, set2))
        else:
          test_pairs.append((set1, set2))
    tf.logging.info(
        'set_pair (intersection & union) examples: %d train, %d test',
        len(train_pairs), len(test_pairs))

    return train_pairs, test_pairs

  def construct_set_follow_facts(
      self, train_split):
    """Construct datasets from KB that a set of entities follow a relation.

    Note that the results of a set-follow operation can be multiple entities.
    For example, given a natural set of movies that stars Robert Downey Jr.,
    and a relation "stars", the results are all actors who co-star with
    Robert Downey Jr. in some movies.

    Args:
      train_split: percentage of data for training

    Returns:
      a list of train pairs and test pairs
    """
    random.seed(SEED)
    subj2factids = dict()
    rel2factids = dict()
    for subj, facts in self.kb.items():
      if subj not in subj2factids:
        subj2factids[subj] = set()
      for rel, objs in facts.items():
        if rel not in rel2factids:
          rel2factids[rel] = set()
        factids = [self.fact2id[(subj, rel, obj)] for obj in objs]
        subj2factids[subj].update(factids)
        rel2factids[rel].update(factids)

    train_facts = list()
    test_facts = list()
    for subj, facts in self.kb.items():
      for rel, objs in facts.items():
        # For convenience, we take the object and inverse relation as samples.
        if len(objs) <= 1:
          continue
        subj_factids = set().union(*[subj2factids[obj] for obj in objs])
        rel_factids = rel2factids[self._inverse_rel(rel)]
        if random.random() < train_split:
          train_facts.append((subj_factids, rel_factids))
        else:
          test_facts.append((subj_factids, rel_factids))
    tf.logging.info('set_follow examples: %d train, %d test',
                    len(train_facts), len(test_facts))
    return train_facts, test_facts

  def construct_metaqa_data(self, train_file,
                            test_file):
    """Load MetaQA datasets from files.

    Args:
      train_file: filename for train data
      test_file: filename for test data

    Returns:
      a list of train pairs and test pairs

    """
    self.max_question_len = 0
    train_data, test_data = list(), list()
    for filename, data in zip([train_file, test_file], [train_data, test_data]):
      with gfile.GFile(self.root_dir + filename) as f:
        for line in f:
          line = json.loads(line)
          question = [self.word2id[word] for word in line['question'].split()]
          self.max_question_len = max(self.max_question_len, len(question))
          question_entity = line['question_entities'][0]
          question_entity_id = self.entity2id[question_entity]

          answers = set(line['answers'])
          answer_fact_ids = [
              f_id for (subj, relation, obj), f_id in self.fact2id.items()
              if obj in answers
          ]
          data.append([question, question_entity_id, answer_fact_ids])

    return train_data, test_data

  def construct_webqsp(self,
                       train_file,
                       test_file,
                       max_webqsp_ans = 1000):
    """Load WebQuestionsSP datasets from files.

    Args:
      train_file: filename for train data
      test_file: filename for test data
      max_webqsp_ans: max number of answers, dropped otherwise

    Returns:
      a list of train data and test data

    """
    self.max_webqsp_ans = max_webqsp_ans
    train_data, test_data = [], []
    self.ent2ent_relations = set()
    self.ent2cvt_relations = set()
    self.cvt2ent_relations = set()
    for filename, data_split in zip([train_file, test_file],
                                    [train_data, test_data]):
      all_data = json.load(gfile.GFile(self.root_dir + filename))
      for data in all_data['Questions']:
        qid = data['QuestionId']

        # Parse question and question entities.
        question = data['ProcessedQuestion']
        _, (question_token_ids, segment_ids, question_mask) = \
            self.bert_tokenizer.tokenize(question)
        question_entity = self.convert_fbid(data['Parses'][0]['TopicEntityMid'])
        if question_entity is None or question_entity not in self.entity2id:
          tf.logging.warn('Question %s illegal TopicEntityMid %s ' %
                          (qid, question_entity))
          continue
        question_entity_id = self.entity2id[question_entity]

        # Parse answers. Convert answers into a list of ids and pad with -1.
        answers = set()
        for parse in data['Parses']:
          answers.update([
              self.convert_fbid(ans['AnswerArgument'])
              if ans['AnswerType'] == 'Entity' else ans['AnswerArgument']
              for ans in parse['Answers']
          ])
        answers = list(answers)[:max_webqsp_ans]
        answer_ids = np.array(
            [self.entity2id[ans] for ans in answers if ans in self.entity2id],
            dtype=np.int32)
        if len(answer_ids) == 0:  # pylint: disable=g-explicit-length-test
          tf.logging.warn('Question %s no answers' % (qid))
          continue
        answer_ids = np.pad(
            answer_ids, (0, max_webqsp_ans - len(answer_ids)),
            mode='constant',
            constant_values=-1)

        # Parse constraints.
        constraint_entity_id = -1
        constraints = data['Parses'][0]['Constraints']
        if constraints and constraints[0]['ArgumentType'] == 'Entity':
          constraint = constraints[0]
          constraint_entity = self.convert_fbid(constraint['Argument'])
          constraint_entity_id = self.entity2id[constraint_entity]

        # Add one data point to the list.
        data_split.append(
            (question_token_ids, segment_ids, question_mask, question_entity_id,
             constraint_entity_id, answer_ids))

        # We further check if a relation is a ent2ent or ent2cvt relation.
        # Here, we simply assume that if the inferential chain is longer
        # than 2, then the intermediate entity is a cvt entity. This only
        # holds for the WebQuestionsSP dataset.
        inferential_chain = data['Parses'][0]['InferentialChain']
        if inferential_chain is None:
          continue
        inferential_chain = ['<fb:%s>' % rel for rel in inferential_chain]
        if len(inferential_chain) == 1:
          self.ent2ent_relations.add(inferential_chain[0])
        elif len(inferential_chain) == 2:
          self.ent2cvt_relations.add(inferential_chain[0])
          self.cvt2ent_relations.add(inferential_chain[1])

    tf.logging.info('WebQuestions data loaded successfully!')
    tf.logging.info('train: %d, test: %d' % (len(train_data), len(test_data)))
    return train_data, test_data

  def check_cvt_entity(self):
    """This function check if an entity is cvt.

    We simply assume that tail entities of a ent2cvt relation is a cvt entity.
    And head entities of a cvt2ent relation is a cvt entity.

    Returns:
      A zero-one vector indicating if an entity is cvt
    """
    cvt_entities = set()
    ent_entities = set()
    for subj, facts in tqdm(self.kb.items()):
      for rel, objs in facts.items():
        if rel in self.ent2ent_relations:
          ent_entities.add(subj)
          ent_entities.update(objs)
          if subj in cvt_entities or (set(objs) & cvt_entities):
            pass
        elif rel in self.ent2cvt_relations:
          ent_entities.add(subj)
          cvt_entities.update(objs)
          if subj in cvt_entities or (set(objs) & ent_entities):
            pass
        elif rel in self.cvt2ent_relations:
          cvt_entities.add(subj)
          ent_entities.update(objs)
          if subj in ent_entities or (set(objs) & cvt_entities):
            pass

    tf.logging.info('cvt_entities: %d ent_entities: %d' %
                    (len(cvt_entities), len(ent_entities)))

    all_entity_is_cvt = np.zeros(len(self.id2entity), dtype=np.int32)
    for entity in cvt_entities:
      all_entity_is_cvt[self.entity2id[entity]] = 1
    return all_entity_is_cvt

  def construct_query2box(self, name):
    """Load query2box data into a list of examples."""
    task = name.split('_')[-1]
    # we don't train our model in the same way as query2box,
    # so we won't load their training data.
    test_data = pickle.load(
        gfile.GFile(self.root_dir + 'test_ans_%s.pkl' % task, 'rb'))

    converted_test_data = list()
    for query, unused_answers in test_data.items():
      converted_test_data.append(query)

    return None, converted_test_data

  ##############################################################
  ###################### Build tf.dataset ######################
  ##############################################################

  def build_input_fn(self,
                     name,
                     batch_size,
                     mode,
                     epochs = 1,
                     n_take = -1,
                     shuffle = False):
    """Construct tf.Dataset object as input.

    Args:
      name: name of training task
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: number of epochs to run
      n_take: number of random samples selected
      shuffle: if shuffle the dataset or not (train vs. test)

    Returns:
      input_fn
    """
    if name == 'membership':
      return self.build_membership_input_fn(batch_size, mode, epochs, n_take,
                                            shuffle)
    elif name == 'intersection' or name == 'union':
      return self.build_intersection_input_fn(name, batch_size, mode, epochs,
                                              n_take, shuffle)
    elif name == 'follow' or name == 'set_follow':
      return self.build_follow_input_fn(batch_size, mode, epochs, n_take,
                                        shuffle)
    elif name in ['metaqa2', 'metaqa3']:
      return self.build_metaqa_input_fn(batch_size, mode, epochs, n_take,
                                        shuffle)
    elif name == 'mixture':
      return self.build_mixture_input_fn(batch_size, mode, epochs, n_take)
    elif name == 'webqsp':
      return self.build_webqsp_input_fn(batch_size, mode, epochs, n_take,
                                        shuffle)
    elif name.startswith('query2box'):
      return self.build_query2box_input_fn(
          name, batch_size, mode)
    else:
      raise ValueError('name not recognized: %s' % name)

  def build_membership_input_fn(
      self,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1,
      shuffle = False):
    """Build set membership input_fn.

    Args:
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: epochs
      n_take: number of examples to pick in each epoch
      shuffle: if shuffle dataset

    Returns:
      input_fn
    """
    if mode == 'train':
      data = self.train_data_membership
    elif mode == 'eval':
      data = self.test_data_membership
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def membership_generator():
      """Generate set intersections.

      Yields:
        a tuple
      """
      if shuffle:
        random_ids = np.random.permutation(len(data))
      else:
        random_ids = np.arange(len(data))

      for data_id in random_ids:
        objs = data[data_id]
        yield self.gen_one_membership_data(objs), 0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      ds = tf.data.Dataset.from_generator(
          generator=membership_generator,
          output_types=((tf.int32, tf.float32), tf.int32),
          output_shapes=((tf.TensorShape([self.max_set]),
                          tf.TensorShape([self.num_entities])),
                         tf.TensorShape([])))

      if n_take > 0:
        ds = ds.take(n_take)
      ds = ds.repeat(epochs)
      ds = ds.batch(batch_size)
      return ds

    return input_fn

  def build_intersection_input_fn(
      self,
      name,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1,
      shuffle = False):
    """Build input function for set intersections.

    Args:
      name: 'intersection' or 'union'
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: number of epochs
      n_take: size of each epoch
      shuffle: if shuffle or not

    Returns:
      input_fn
    """
    if mode == 'train':
      set_pairs = self.train_data_set_pair
    elif mode == 'eval':
      set_pairs = self.test_data_set_pair
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def set_intersection_generator():
      """Generate set intersections.

      Yields:
        a tuple
      """
      for set1, set2 in set_pairs:
        yield self.gen_one_pair_data(name, set1, set2), 0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      tensor_shape = self.num_entities
      ds = tf.data.Dataset.from_generator(
          generator=set_intersection_generator,
          output_types=((tf.float32, tf.float32, tf.float32), tf.int32),
          output_shapes=((tf.TensorShape([tensor_shape]),
                          tf.TensorShape([tensor_shape]),
                          tf.TensorShape([tensor_shape])), tf.TensorShape([])))

      if n_take > 0:
        ds = ds.take(n_take)
      if shuffle:
        ds = ds.shuffle(SEED)
      ds = ds.repeat(epochs)
      ds = ds.batch(batch_size)
      return ds

    return input_fn

  def build_follow_input_fn(
      self,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1,
      shuffle = False):
    """Build input function for set-follow operations.

    Args:
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: number of epochs
      n_take: size of each epoch
      shuffle: if shuffle or not

    Returns:
      input_fn
    """
    if mode == 'train':
      follow_facts = self.train_data_follow
    elif mode == 'eval':
      follow_facts = self.test_data_follow
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def fact_follow_generator():
      """Generate set intersections.

      Yields:
        a tuple
      """
      for subj_factids, rel_factids in follow_facts:
        # set-follow can be treated as set intersections between a set
        # of facts that have correct subjects and a set of facts that
        # have correct relations.
        yield self.gen_one_follow_data(subj_factids, rel_factids), 0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      tensor_shape = self.num_facts
      ds = tf.data.Dataset.from_generator(
          generator=fact_follow_generator,
          output_types=((tf.float32, tf.float32, tf.float32), tf.int32),
          output_shapes=((tf.TensorShape([tensor_shape]),
                          tf.TensorShape([tensor_shape]),
                          tf.TensorShape([tensor_shape])), tf.TensorShape([])))

      if n_take > 0:
        ds = ds.take(n_take)
      if shuffle:
        ds = ds.shuffle(SEED)
      ds = ds.repeat(epochs)
      ds = ds.batch(batch_size)
      return ds

    return input_fn

  def build_mixture_input_fn(
      self,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1):
    """Build mixture input_fn of membership, intersection, union, set_follow.

    Args:
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: epochs
      n_take: number of examples to pick in each epoch

    Returns:
      input_fn
    """
    if mode == 'train':
      set_members = self.train_data_membership
      set_pairs = self.train_data_set_pair
      set_follows = self.train_data_follow
    elif mode == 'eval':
      set_members = self.test_data_membership
      set_pairs = self.test_data_set_pair
      set_follows = self.test_data_follow
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def mixture_generator():
      """Generate set intersections.

      Yields:
        a tuple with one data for each task
      """
      data_per_epoch = DATA_PER_EPOCH if n_take == -1 else n_take
      for _ in range(data_per_epoch * epochs):
        # Get model results for the task of membership.
        m_objs = set_members[np.random.randint(len(set_members))]
        m_padded_ent_ids, m_labels = self.gen_one_membership_data(m_objs)

        # Get model results for the task of intersection and union.
        p_set1, p_set2 = set_pairs[np.random.randint(len(set_pairs))]
        p_candidate_set1, p_candidate_set2, p_union_labels = \
            self.gen_one_pair_data('union', p_set1, p_set2)
        p_intersection_labels = np.minimum(p_candidate_set1, p_candidate_set2)

        # Get model results for the task of set_follows.
        f_subj_factids, f_rel_factids = set_follows[np.random.randint(
            len(set_follows))]
        f_candidate_set1, f_candidate_set2, f_labels = self.gen_one_follow_data(
            f_subj_factids, f_rel_factids)

        yield (m_padded_ent_ids, m_labels, p_candidate_set1, p_candidate_set2,
               p_union_labels, p_intersection_labels, f_candidate_set1,
               f_candidate_set2, f_labels), 0.0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      ds = tf.data.Dataset.from_generator(
          generator=mixture_generator,
          output_types=((tf.int32, tf.float32, tf.float32, tf.float32,
                         tf.float32, tf.float32, tf.float32, tf.float32,
                         tf.float32), tf.float32),
          output_shapes=((tf.TensorShape([self.max_set]),
                          tf.TensorShape([self.num_entities]),
                          tf.TensorShape([self.num_entities]),
                          tf.TensorShape([self.num_entities]),
                          tf.TensorShape([self.num_entities]),
                          tf.TensorShape([self.num_entities]),
                          tf.TensorShape([self.num_facts]),
                          tf.TensorShape([self.num_facts]),
                          tf.TensorShape([self.num_facts])),
                         tf.TensorShape([])))

      ds = ds.batch(batch_size)
      return ds

    return input_fn

  def build_metaqa_input_fn(
      self,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1,
      shuffle = False):
    """Return tensors in required shape.

    Args:
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: epochs
      n_take: number of examples to pick in each epoch
      shuffle: if shuffle dataset

    Returns:
      tf.Dataset
    """
    if mode == 'train':
      data = self.train_data_metaqa
    elif mode == 'eval':
      data = self.test_data_metaqa
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def metaqa_generator():
      """Generate metaqa question answer pair.

      Yields:
        a tuple
      """
      for one_data in data:
        question, question_entity_id, answer_fact_ids = one_data
        question = np.array(question + [-1] *
                            (self.max_question_len - len(question)))
        question_entity_sketch = \
            self.cm_context.get_sketch(xs=[question_entity_id])
        answers = np.zeros(self.num_facts, dtype=bool)
        answers[answer_fact_ids] = True

        yield (question, question_entity_id, question_entity_sketch, answers), 0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      tensor_shape = self.num_facts
      ds = tf.data.Dataset.from_generator(
          generator=metaqa_generator,
          output_types=((tf.int32, tf.int32, tf.float32, tf.float32), tf.int32),
          output_shapes=((tf.TensorShape([self.max_question_len]),
                          tf.TensorShape([]),
                          tf.TensorShape(
                              [self.cm_context.depth, self.cm_context.width]),
                          tf.TensorShape([tensor_shape])), tf.TensorShape([])))

      if n_take > 0:
        ds = ds.take(n_take)
      if shuffle:
        ds = ds.shuffle(SEED)
      ds = ds.repeat(epochs)
      ds = ds.batch(batch_size)
      return ds

    return input_fn

  def build_webqsp_input_fn(
      self,
      batch_size,
      mode,
      epochs = 1,
      n_take = -1,
      shuffle = False):
    """Return tensors in required shape.

    Args:
      batch_size: batch_size
      mode: 'train' or 'test'
      epochs: epochs
      n_take: number of examples to pick in each epoch
      shuffle: if shuffle dataset

    Returns:
      tf.Dataset
    """
    if mode == 'train':
      data = self.train_data_webqsp
    elif mode == 'eval':
      data = self.test_data_webqsp
    else:
      raise ValueError('mode not recognized: %s' % mode)

    def webqsp_generator():
      """Generate set intersections.

      Yields:
        a tuple of features and label
      """
      for (question_token_ids, segment_ids, question_mask, question_entity_id,
           constraint_entity_id, answer_ids) in data:
        # question
        question_entity_sketch = np.zeros((self.cm_depth, self.cm_width),
                                          dtype=np.float32)
        for i, j in enumerate(self.all_entity_sketches[question_entity_id]):
          question_entity_sketch[i, j] = 1.0
        # constraint
        constraint_entity_sketch = np.zeros((self.cm_depth, self.cm_width),
                                            dtype=np.float32)
        if constraint_entity_id != -1:
          for i, j in enumerate(self.all_entity_sketches[constraint_entity_id]):
            constraint_entity_sketch[i, j] = 1.0
        yield (question_token_ids, segment_ids, question_mask,
               question_entity_id, question_entity_sketch, constraint_entity_id,
               constraint_entity_sketch, answer_ids), 0

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      ds = tf.data.Dataset.from_generator(
          generator=webqsp_generator,
          output_types=((tf.int32, tf.int32, tf.int32, tf.int32, tf.float32,
                         tf.int32, tf.float32, tf.int32), tf.int32),
          output_shapes=((tf.TensorShape([self.bert_tokenizer.max_seq_length]),
                          tf.TensorShape([self.bert_tokenizer.max_seq_length]),
                          tf.TensorShape([self.bert_tokenizer.max_seq_length]),
                          tf.TensorShape([]),
                          tf.TensorShape([self.cm_depth,
                                          self.cm_width]), tf.TensorShape([]),
                          tf.TensorShape([self.cm_depth, self.cm_width]),
                          tf.TensorShape([self.max_webqsp_ans])),
                         tf.TensorShape([])))

      if n_take > 0:
        ds = ds.take(n_take)
      if shuffle:
        ds = ds.shuffle(SEED)
      ds = ds.repeat(epochs)
      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return ds

    return input_fn

  def build_query2box_input_fn(
      self,
      name,
      batch_size,
      mode):
    """Build input function.

    Build input function and return the inputs in their required
    formats:
        1c: ent, rel
        2c: ent, rel1, rel2
        3c: ent, rel1, rel2, rel3
        2i or 2u: ent1, rel1, ent2, rel2
        3i: ent1, rel1, ent2, rel2, ent3, rel3
        ic or uc: ent1, rel1, ent2, rel2, rel3
        ci: ent1, rel1, rel2, ent2, rel3

    Args:
      name: name of the experiment.
      batch_size: batch size
      mode: 'train' or 'eval'

    Returns:
      tf.data.Dataset

    """

    task = name.split('_')[-1]

    if mode == 'eval':
      data = self.test_data_query2box
    else:
      raise ValueError('Only test mode allowed for query2box: mode = %s' % mode)

    def convert_entid(q2b_entid):
      # KB might be loaded in different ways such that entity ids
      # won't match. So this function converts q2b ids to our
      # entity ids.
      return self.entity2id[self.q2b_id2entity[q2b_entid]]

    def convert_relid(q2b_relid):
      return self.relation2id[self.q2b_id2relation[q2b_relid]]

    def query2box_generator():
      """Convert query2box features to EmQL features.

      We simply hard-code their input format for parsing, since there's not
      a simple way to automatically serialized to our input format.

      Yields:
        a tuple of features and label
      """
      for query in data:
        if task == '1c':
          assert len(query) == 1
          ent, rels = query[0]
          yield (convert_entid(ent), convert_relid(rels[0])), 0
        elif task == '2c':
          assert len(query) == 1
          ent, rels = query[0]
          yield (convert_entid(ent),
                 convert_relid(rels[0]),
                 convert_relid(rels[1])), 0
        elif task == '3c':
          assert len(query) == 1
          ent, rels = query[0]
          yield (convert_entid(ent),
                 convert_relid(rels[0]),
                 convert_relid(rels[1]),
                 convert_relid(rels[2])), 0
        elif task == '2i' or task == '2u':
          assert len(query) == 2
          ent1, rels1 = query[0]
          ent2, rels2 = query[1]
          yield (convert_entid(ent1), convert_relid(rels1[0]),
                 convert_entid(ent2), convert_relid(rels2[0])), 0
        elif task == '3i':
          assert len(query) == 3
          ent1, rels1 = query[0]
          ent2, rels2 = query[1]
          ent3, rels3 = query[2]
          yield (convert_entid(ent1), convert_relid(rels1[0]),
                 convert_entid(ent2), convert_relid(rels2[0]),
                 convert_entid(ent3), convert_relid(rels3[0])), 0
        elif task == 'ic' or task == 'uc':
          assert len(query) == 3
          ent1, rels1 = query[0]
          ent2, rels2 = query[1]
          rel3 = query[2]
          yield (convert_entid(ent1), convert_relid(rels1[0]),
                 convert_entid(ent2), convert_relid(rels2[0]),
                 convert_relid(rel3)), 0
        elif task == 'ci':
          assert len(query) == 2
          ent1, rels1 = query[0]
          ent2, rels2 = query[1]
          yield (convert_entid(ent1), convert_relid(rels1[0]),
                 convert_relid(rels1[1]),
                 convert_entid(ent2), convert_relid(rels2[0])), 0
        else:
          raise ValueError

    def input_fn():
      """Create tf.dataset for estimator.

      Returns:
        tf dataset
      """
      # output_types
      if task == '1c':
        output_types = tuple([tf.int32] * 2)
        output_shapes = tuple([tf.TensorShape([])] * 2)
      elif task == '2c':
        output_types = tuple([tf.int32] * 3)
        output_shapes = tuple([tf.TensorShape([])] * 3)
      elif task == '3c':
        output_types = tuple([tf.int32] * 4)
        output_shapes = tuple([tf.TensorShape([])] * 4)
      elif task == '2i' or task == '2u':
        output_types = tuple([tf.int32] * 4)
        output_shapes = tuple([tf.TensorShape([])] * 4)
      elif task == '3i':
        output_types = tuple([tf.int32] * 6)
        output_shapes = tuple([tf.TensorShape([])] * 6)
      elif task == 'ic' or task == 'uc':
        output_types = tuple([tf.int32] * 5)
        output_shapes = tuple([tf.TensorShape([])] * 5)
      elif task == 'ci':
        output_types = tuple([tf.int32] * 5)
        output_shapes = tuple([tf.TensorShape([])] * 5)
      else:
        raise ValueError

      ds = tf.data.Dataset.from_generator(
          generator=query2box_generator,
          output_types=(output_types, tf.int32),
          output_shapes=(output_shapes, tf.TensorShape([])))

      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return ds

    return input_fn

  ##############################################################
  ########################### Helper ###########################
  ##############################################################

  def gen_one_membership_data(self,
                              objs):
    padded_ent_ids = np.full(self.max_set, -1, dtype=np.int32)
    labels = np.zeros(self.num_entities, dtype=np.bool_)
    for i, obj in enumerate(objs):
      obj_id = self.entity2id[obj]
      if i < self.max_set:
        padded_ent_ids[i] = obj_id
      labels[obj_id] = True
    return padded_ent_ids, labels

  def gen_one_pair_data(self, name, set1,
                        set2):
    """Generate one data point for set intersection.

    Args:
      name: 'intersection' or 'union'
      set1: a list of entities in set 1
      set2: a list of entities in set 2

    Returns:
      k hot vectors of set1, set2 and labels
    """
    assert len(set1 & set2) >= 1
    candidate_set1 = np.zeros(self.num_entities)
    candidate_set1[[self.entity2id[e] for e in set1]] = 1.0
    candidate_set2 = np.zeros(self.num_entities)
    candidate_set2[[self.entity2id[e] for e in set2]] = 1.0

    if name == 'union':
      return (candidate_set1, candidate_set2,
              np.maximum(candidate_set1, candidate_set2))
    elif name == 'intersection':
      return (candidate_set1, candidate_set2,
              np.minimum(candidate_set1, candidate_set2))
    else:
      raise ValueError('name not recognised: %s' % name)

  def gen_one_follow_data(self, subj_factids,
                          rel_factids):
    assert len(subj_factids & rel_factids) >= 1
    subj_set1 = np.zeros(self.num_facts, dtype=np.float32)
    subj_set1[[fid for fid in subj_factids]] = 1.0
    rel_set2 = np.zeros(self.num_facts, dtype=np.float32)
    rel_set2[[fid for fid in rel_factids]] = 1.0
    labels = np.minimum(subj_set1, rel_set2)
    return subj_set1, rel_set2, labels

  def _inverse_rel(self, rel):
    suffix = self._inv_suffix
    if rel.endswith(suffix):
      return rel[:-len(suffix)]
    else:
      return rel + suffix

  def convert_fbid(self, fbid):
    if fbid is None:
      return None
    assert '.' in fbid
    return '<fb:%s>' % fbid

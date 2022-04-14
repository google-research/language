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
# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension
"""MuZero Search Agent env factory."""

import collections
import functools
import pickle
import random
import string
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from absl import logging
import dataclasses
import grpc
import gym
from language.search_agents import environment_pb2
from language.search_agents.muzero import bert_state_lib
from language.search_agents.muzero import common_flags
from language.search_agents.muzero import grammar_lib
from language.search_agents.muzero import server
from language.search_agents.muzero import state_tree
from language.search_agents.muzero import types
from language.search_agents.muzero import utils
import numpy as np
import pygtrie
from seed_rl.common import common_flags as seed_common_flags  # pylint: disable=unused-import
import tensorflow as tf
import transformers

from muzero import core as mzcore
from muzero import learner_flags
from official.nlp.bert import configs


@dataclasses.dataclass
class ValidWords:
  tfidf_tokens: Dict[int, int]
  word_piece_actions: Sequence[int]
  words: List[str]
  full_words: Set[str]


@dataclasses.dataclass
class ValidWordsByType:
  all_valid_words: ValidWords
  question_valid_words: ValidWords
  answer_valid_words: ValidWords
  document_valid_words: ValidWords
  title_valid_words: ValidWords
  diff_valid_words: ValidWords
  intersect_valid_words: ValidWords


@dataclasses.dataclass
class Observations:
  observation: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
  valid_words: ValidWordsByType


def new_valid_words() -> ValidWords:
  return ValidWords(
      tfidf_tokens=collections.defaultdict(int),
      word_piece_actions=[],
      words=[],
      full_words=set())


def get_descriptor() -> mzcore.EnvironmentDescriptor:
  sequence_length = common_flags.BERT_SEQUENCE_LENGTH.value
  bert_config = configs.BertConfig.from_json_file(
      common_flags.BERT_CONFIG.value)

  grammar_config = grammar_lib.grammar_config_from_flags()
  max_len_type_vocab = max(map(len, bert_state_lib.TYPE_VOCABS.values()))
  tokenizer = bert_state_lib.get_tokenizer()
  grammar = grammar_lib.construct_grammar(
      grammar_config=grammar_config, vocab=list(tokenizer.get_vocab().keys()))

  observation_space = gym.spaces.Tuple([
      gym.spaces.Box(0, bert_config.vocab_size, (sequence_length,), np.int32),
      gym.spaces.Box(0, max_len_type_vocab,
                     (sequence_length, len(bert_state_lib.TYPE_VOCABS)),
                     np.int32),
      gym.spaces.Box(-np.inf, np.inf,
                     (sequence_length, len(bert_state_lib.FLOAT_NAMES)),
                     np.float32),
      gym.spaces.Box(0,
                     len(grammar.productions()) + 1,
                     (common_flags.N_ACTIONS_ENCODING.value,),
                     np.int32),  # +1 for mask
  ])

  max_episode_length = common_flags.MAX_NUM_ACTIONS.value
  # If you change rewards / add new rewards, make sure to update the bounds.
  min_possible_score, max_possible_score = {
      'curiosity+dcg': (-1, 1),
  }.get(common_flags.REWARD.value, (-1, 1))
  min_possible_cumulative_score, max_possible_cumulative_score = {
      'curiosity+dcg': (-2, 2),
  }.get(common_flags.REWARD.value, (min_possible_score, max_possible_score))

  logging.info('Max episode length: %d; Score range: [%.2f, %.2f]',
               max_episode_length, min_possible_score, max_possible_score)

  # Additional statistics that we want to track 'in' the learner
  learner_stats = (
      ('ndcg_score', tf.float32),
      ('ndcg_score_improvement', tf.float32),
      ('em_at_1', tf.float32),
      ('em_at_1_improvement', tf.float32),
      (f'em_at_{int(common_flags.K.value)}', tf.float32),
      (f'em_at_{int(common_flags.K.value)}_improvement', tf.float32),
      (f'recall_at_{int(common_flags.K.value)}', tf.float32),
      (f'recall_at_{int(common_flags.K.value)}_improvement', tf.float32),
      ('recall_at_1', tf.float32),
      ('recall_at_1_improvement', tf.float32),
      ('documents_explored', tf.float32),
  )

  return mzcore.EnvironmentDescriptor(
      observation_space=observation_space,
      action_space=gym.spaces.Discrete(len(grammar.productions())),
      reward_range=mzcore.Range(min_possible_score, max_possible_score),
      value_range=mzcore.Range(min_possible_cumulative_score,
                               max_possible_cumulative_score),
      pretraining_space=gym.spaces.Tuple([
          observation_space,
          gym.spaces.Box(0, len(grammar.productions()), (), np.int32),  # action
          gym.spaces.Box(0., 1., (), np.float32),  # reward
          gym.spaces.Box(0., 1., (), np.float32),  # value
          gym.spaces.Box(0., 1., (), np.float32),  # mask
      ] * common_flags.PRETRAINING_NUM_UNROLL_STEPS.value),
      extras={
          'bert_config':
              bert_config,
          'sequence_length':
              sequence_length,
          'float_names':
              bert_state_lib.FLOAT_NAMES,
          'type_vocabs':
              bert_state_lib.TYPE_VOCABS,
          'num_float_features':
              len(bert_state_lib.FLOAT_NAMES),
          'type_vocab_sizes': [
              len(v) for v in bert_state_lib.TYPE_VOCABS.values()
          ],
          'grammar':
              grammar,
          'max_episode_length':
              max_episode_length +
              5,  # we never want the agent to finish the episode
          'learner_stats':
              learner_stats,
          # Only set this if `learner` does not specify an already pre-trained
          # checkpoint.
          'bert_init_ckpt':
              common_flags.BERT_INIT_CKPT.value
              if learner_flags.INIT_CHECKPOINT.value is None else None,
          'action_encoder_hidden_size':
              common_flags.ACTION_ENCODER_HIDDEN_SIZE.value,
          'tokenizer':
              tokenizer,
          'grammar_config':
              grammar_config,
          'pretraining_num_unroll_steps':
              common_flags.PRETRAINING_NUM_UNROLL_STEPS.value,
      })


def to_action_tuple(
    word: str,
    grammar: state_tree.NQCFG,
    tokenizer: transformers.BertTokenizer,
    valid_actions: Optional[Collection[int]] = None) -> Tuple[int, ...]:

  # If you want to programmatically exclude certain kinds of tokens from the
  # tries, add the logic here and return an empty tuple for `word`s you do not
  # want to see included.
  if (common_flags.EXCLUDE_PUNCTUATION_FROM_TRIE.value == 1 and
      word in string.punctuation):
    return ()

  # We need to skip `unks` that aren't even covered by word pieces.
  tokens = tokenizer.tokenize(word)
  actions = []
  for i, token in enumerate(tokens):
    # The vocabulary might contain items like `toxin’s` which the BERT tokenizer
    # does tokenize into several initial word pieces.  Presence of such a
    # sequence in the trie would be inconsistent with our insistence of deriving
    # words as exactly one initial word piece followed by several non-initial
    # ones.  We can still return the "word" assembled so far, as this will match
    # the grammar, but cannot accumulate any more tokens.
    if i > 0 and not token.startswith('##'):
      break
    token = state_tree.NQStateTree.clean_escape_characters(token)
    if token not in grammar.terminal_to_action:
      return ()
    token_action = grammar.terminal_to_action[token]

    # If we are in a "constrained" setting, we need to ensure that every token
    # of a word is actually present.  This situation can arise if the state
    # truncation cuts off the final word piece(s) of a multi-piece sequence.
    # Implementing the constraint here allows us to not worry about it in the
    # transition model; otherwise, we would not just need to check for sub-trie
    # presence for a prefix to determine recursion, but check explicitly that at
    # least one path along the subtrie is "constructible".
    if valid_actions is not None:
      if token_action not in valid_actions:
        return ()

    actions.append(token_action)
  return tuple(actions)


class NQEnv(gym.Env):
  """NQ environment."""

  def __init__(self,
               nq_server: server.NQServer,
               state: Optional[types.EnvState] = None,
               random_state: Optional[np.random.RandomState] = None,
               training: bool = True,
               stop_after_seeing_new_results: bool = False):
    super().__init__()
    self.nq_server = nq_server
    self.training = training

    self.first_time = True  # Used for initial debug logging

    self.stop_after_seeing_new_results = stop_after_seeing_new_results

    self.descriptor = get_descriptor()
    self.grammar = self.descriptor.extras['grammar']
    self.tokenizer = self.descriptor.extras['tokenizer']
    self.action_space = len(self.grammar.productions())

    self.idf_lookup = utils.IDFLookup.get_instance(
        path=common_flags.IDF_LOOKUP_PATH.value)

    trie_start_time = tf.timestamp()
    if common_flags.GLOBAL_TRIE_PATH.value is None:
      self.global_trie = pygtrie.Trie.fromkeys((x for x in map(
          functools.partial(
              to_action_tuple, grammar=self.grammar, tokenizer=self.tokenizer),
          self.idf_lookup.lookup) if x))
      self._logging_info('Built trie of size %s in %s s', len(self.global_trie),
                         tf.timestamp() - trie_start_time)
    else:
      with tf.io.gfile.GFile(common_flags.GLOBAL_TRIE_PATH.value,
                             'rb') as trie_f:
        self.global_trie = pickle.load(trie_f)
      self._logging_info('Restored trie of size %s in %s s',
                         len(self.global_trie),
                         tf.timestamp() - trie_start_time)

    # The value of the global steps in the learner is updated in step()
    self.training_steps = 0

    # Trie for the current results.  We only build this the first time after
    # a new set of results is obtained.  A value of `None` indicates that for
    # the current set of results, it has not been built yet.
    self.known_word_tries = None  # type: Optional[state_tree.KnownWordTries]
    self.valid_word_actions = None  # type: Optional[state_tree.ValidWordActions]
    self.use_rf_restrict = False

    self.state = state
    if state and state.tree is None:
      self.state.tree = state_tree.NQStateTree(grammar=self.grammar)

    self.bert_config: configs.BertConfig = self.descriptor.extras['bert_config']
    self.sequence_length: int = self.descriptor.extras['sequence_length']
    self.action_history = []
    self.n_episode = 0

    self._rand = np.random.RandomState()
    if random_state:
      self._rand.set_state(random_state)

  def _logging_info(self, prefix, *args, **kwargs):
    if self.training:
      prefix = 'TRAIN: ' + prefix
    else:
      prefix = 'TEST : ' + prefix

    logging.info(prefix, *args, **kwargs)

  def _get_query(self, index: Optional[int] = None):
    try:
      query = self.nq_server.get_query(
          index=index, dataset_type='TRAIN' if self.training else 'DEV')
    except grpc.RpcError as rpc_exception:
      raise mzcore.RLEnvironmentError from rpc_exception
    return query

  def _get_env_output(
      self,
      query: str,
      original_query: environment_pb2.GetQueryResponse,
      documents: Optional[Sequence[environment_pb2.Document]] = None
  ) -> types.HistoryEntry:
    """Query the environment with a new query.

    Args:
      query:  Query (reformulation) issued by the agent.
      original_query:  The original query (and its gold answers) that the agent
        is playing.
      documents:  If set, does not actually query the environment.  This is used
        to avoid repeating requests for pre-training data for which we already
        computed the results at creation time.

    Returns:
      A `HistoryEntry` summarizing the environment's response for `query` for
      the agent.
    """
    if documents is None:
      try:
        response = self.nq_server.get_documents(
            query,
            original_query=original_query.query,
            num_documents=common_flags.NUM_DOCUMENTS_TO_RETRIEVE.value,
            num_ir_documents=common_flags.NUM_IR_DOCUMENTS_TO_RETRIEVE.value)
      except grpc.RpcError as rpc_exception:
        raise mzcore.RLEnvironmentError from rpc_exception

      docs = response.documents
    else:
      docs = documents
    ranked_docs = sorted(
        docs, key=lambda doc: doc.answer.mr_score, reverse=True)

    entry = types.HistoryEntry(
        query=query, original_query=original_query, documents=ranked_docs)

    return entry

  def _obs(self) -> Observations:

    def special_convert_tokens_to_ids(*args, **kwargs):
      """Maps all the added tokens to one of the unused tokens."""
      ids = self.tokenizer.convert_tokens_to_ids(*args, **kwargs)
      # size without added tokens
      original_vocab_size = self.tokenizer.vocab_size
      return [
          id_ if id_ < original_vocab_size else (id_ % original_vocab_size) + 1
          for id_ in ids
      ]  # +1 to skip the [PAD] token

    bert_state = make_bert_state(
        environment_state=self.state,
        tokenizer=self.tokenizer,
        idf_lookup=self.idf_lookup)

    token_ids, type_ids, float_values = utils.ObsFragment.combine_and_expand(
        fragments=bert_state,
        length=self.sequence_length,
        type_vocabs=self.descriptor.extras['type_vocabs'],
        float_names=self.descriptor.extras['float_names'],
        tokens_to_id_fn=special_convert_tokens_to_ids,
    )

    def _add_token(token: str, tfidf_score: float, valid_words: ValidWords):
      valid_words.words.append(token)

      # Corresponds to the rule corresponding to token under the policy.
      target_tok_id = self.grammar.terminal_to_action[
          state_tree.NQStateTree.clean_escape_characters(token)]
      valid_words.tfidf_tokens[target_tok_id] += tfidf_score

    def _convert_word_tokens_to_full_words(valid_words_list: List[ValidWords]):
      for valid_words in valid_words_list:
        for full_word in ' '.join(valid_words.words).replace(' ##', '').split():
          valid_words.full_words.add(full_word)

    def _convert_tokens_to_actions(valid_words_list: List[ValidWords]):
      for valid_words in valid_words_list:
        tfidf_tokens = valid_words.tfidf_tokens
        valid_words.word_piece_actions = sorted(
            tfidf_tokens, key=lambda x: tfidf_tokens[x], reverse=True)

    def _discard_full_word(full_word: str, valid_words_list: List[ValidWords]):
      for valid_words in valid_words_list:
        valid_words.full_words.discard(full_word)

    def _add_tokens(valid_words: ValidWords):
      for full_word in valid_words.full_words:
        tfidf_score = self.idf_lookup[full_word]
        tokens = self.tokenizer.tokenize(full_word)
        for token in tokens:
          _add_token(token, tfidf_score, valid_words)

    def _extract_vocabulary(bert_state, token_ids):
      all_valid_words = new_valid_words()
      question_valid_words = new_valid_words()
      answer_valid_words = new_valid_words()
      document_valid_words = new_valid_words()
      title_valid_words = new_valid_words()
      for obs in bert_state:
        for (token, label, tfidf_score) in zip(obs.token_list(),
                                               obs.type_values['state_part'],
                                               obs.float_values['idf_score']):
          if token == '[UNK]':
            continue
          # Corresponds to the id of the token in the input.
          source_tok_id = self.tokenizer.convert_tokens_to_ids(token)

          if source_tok_id in token_ids and tfidf_score > 0:
            if label == 'history_context':
              _add_token(token, tfidf_score, all_valid_words)
              _add_token(token, tfidf_score, document_valid_words)
            elif label == 'history_answer':
              _add_token(token, tfidf_score, answer_valid_words)
            elif label == 'history_title':
              _add_token(token, tfidf_score, all_valid_words)
              _add_token(token, tfidf_score, title_valid_words)
            elif label == 'original_query':
              _add_token(token, tfidf_score, question_valid_words)
      return ValidWordsByType(
          all_valid_words=all_valid_words,
          question_valid_words=question_valid_words,
          answer_valid_words=answer_valid_words,
          document_valid_words=document_valid_words,
          title_valid_words=title_valid_words,
          diff_valid_words=new_valid_words(),
          intersect_valid_words=new_valid_words())

    valid_words_by_type = _extract_vocabulary(bert_state, token_ids)
    valid_words_list = [
        valid_words_by_type.all_valid_words,
        valid_words_by_type.question_valid_words,
        valid_words_by_type.answer_valid_words,
        valid_words_by_type.document_valid_words,
        valid_words_by_type.title_valid_words
    ]

    target_valid_words_by_type = None
    if self.use_rf_restrict and self.state.target_documents:
      target_bert_state = make_bert_state(
          environment_state=self.state,
          tokenizer=self.tokenizer,
          idf_lookup=self.idf_lookup,
          use_target_documents=True)
      target_token_ids, _, _ = utils.ObsFragment.combine_and_expand(
          fragments=target_bert_state,
          length=self.sequence_length,
          type_vocabs=self.descriptor.extras['type_vocabs'],
          float_names=self.descriptor.extras['float_names'],
          tokens_to_id_fn=special_convert_tokens_to_ids,
      )
      target_valid_words_by_type = _extract_vocabulary(target_bert_state,
                                                       target_token_ids)
      valid_words_list.append(target_valid_words_by_type.all_valid_words)

    # Convert wordpiece tokens to full words.
    _convert_word_tokens_to_full_words(valid_words_list)

    # Remove refinement terms used in past queries from valid_words which
    # is used to create the local_trie.
    for history_entry in self.state.history:
      query_substr = history_entry.query[len(self.state.original_query.query):]
      _, adjustments = state_tree.from_lucene_str(query_substr)
      if adjustments:
        for adjustment, _, _ in adjustments:
          _discard_full_word(adjustment.term, valid_words_list)
      for word in query_substr.split():
        _discard_full_word(word, valid_words_list)

    if target_valid_words_by_type:
      v_t = target_valid_words_by_type.all_valid_words.full_words
      v_current = valid_words_by_type.all_valid_words.full_words
      # Update the diff_valid_words and intersect_valid_words accordingly.
      valid_words_by_type.diff_valid_words.full_words = v_current - v_t
      valid_words_by_type.intersect_valid_words.full_words = (
          v_current.intersection(v_t))

      _add_tokens(valid_words_by_type.diff_valid_words)
      _add_tokens(valid_words_by_type.intersect_valid_words)

      # Add the diff/intersect valid words to the list to be converted to word
      # piece actions.
      valid_words_list.append(valid_words_by_type.diff_valid_words)
      valid_words_list.append(valid_words_by_type.intersect_valid_words)

    # Sort the tfidf tokens as word piece actions.
    _convert_tokens_to_actions(valid_words_list)

    token_ids = np.array(token_ids, np.int32)
    type_ids = np.array(type_ids, np.int32).T
    float_values = np.array(float_values, np.float32).T

    return Observations(
        observation=(
            token_ids, type_ids, float_values,
            np.array(
                self.action_history[-common_flags.N_ACTIONS_ENCODING.value:]
                + [self.action_space + 1] *
                (common_flags.N_ACTIONS_ENCODING.value -
                 len(self.action_history)), np.int32)),
        valid_words=valid_words_by_type,
    )

  def seed(self, seed=None) -> None:
    self._rand = np.random.RandomState(seed)

  def get_final_document_list(self):
    """Returns the final documents for the current step.

    Returns:
      List of documents at the end of an episode.
    """
    # If we `stop after seeing results we don't like`, we use the _previous_
    # rather than the latest results for the final reward.
    if common_flags.USE_AGGREGATED_DOCUMENTS.value == 1:
      if self.stop_after_seeing_new_results:
        return get_aggregated_documents(self.state.history[:-1])
      else:
        return get_aggregated_documents(self.state.history)
    else:
      if self.stop_after_seeing_new_results and len(self.state.history) > 1:
        return self.state.history[-2].documents
      else:
        return self.state.history[-1].documents

  def special_episode_statistics_learner(self, return_as_dict=False):
    """Everything you want logged for episodes.

    These are only used for logging.  Actual rewards for learning need to be
    returned via `step`.

    Args:
      return_as_dict:  If true, returns a "human-readable" dict, rather than a
        tensorflow-compatible tuple.

    Returns:
      Metrics/Statistics either as a tuple of numpy.ndarrays, or a dictionary.
    """
    stats = {}
    documents_list = self.get_final_document_list()
    original_documents_list = self.state.history[0].documents

    stats['ndcg_score'] = self.state.score(
        'ndcg', documents_list=documents_list, k=int(common_flags.K.value))
    stats['ndcg_score_improvement'] = stats['ndcg_score'] - self.state.score(
        'ndcg',
        documents_list=original_documents_list,
        k=int(common_flags.K.value))

    stats['em_at_1'] = self.state.score(
        'em_at_k', documents_list=documents_list, k=1)
    stats['em_at_1_improvement'] = stats['em_at_1'] - self.state.score(
        'em_at_k', documents_list=original_documents_list, k=1)

    stats[f'em_at_{int(common_flags.K.value)}'] = self.state.score(
        'em_at_k',
        documents_list=documents_list,
        k=int(common_flags.K.value))
    stats[f'em_at_{int(common_flags.K.value)}_improvement'] = stats[
        f'em_at_{int(common_flags.K.value)}'] - self.state.score(
            'em_at_k',
            documents_list=original_documents_list,
            k=int(common_flags.K.value))

    stats[f'recall_at_{int(common_flags.K.value)}'] = self.state.score(
        'recall_at_k',
        documents_list=documents_list,
        k=int(common_flags.K.value))
    stats[f'recall_at_{int(common_flags.K.value)}_improvement'] = stats[
        f'recall_at_{int(common_flags.K.value)}'] - self.state.score(
            'recall_at_k',
            documents_list=original_documents_list,
            k=int(common_flags.K.value))

    stats['recall_at_1'] = self.state.score(
        'recall_at_k', documents_list=documents_list, k=1)
    stats['recall_at_1_improvement'] = stats['recall_at_1'] - self.state.score(
        'recall_at_k', documents_list=original_documents_list, k=1)

    stats['documents_explored'] = len(
        self.state.sorted_unique_documents(
            step=self.state.num_completed_requests))

    if return_as_dict:
      return stats

    return (
        np.float32(stats['ndcg_score']),
        np.float32(stats['ndcg_score_improvement']),
        np.float32(stats['em_at_1']),
        np.float32(stats['em_at_1_improvement']),
        np.float32(stats[f'em_at_{int(common_flags.K.value)}']),
        np.float32(stats[f'em_at_{int(common_flags.K.value)}_improvement']),
        np.float32(stats[f'recall_at_{int(common_flags.K.value)}']),
        np.float32(
            stats[f'recall_at_{int(common_flags.K.value)}_improvement']),
        np.float32(stats['recall_at_1']),
        np.float32(stats['recall_at_1_improvement']),
        np.float32(stats['documents_explored']),
    )

  def special_episode_statistics(self):
    stats = {}
    stats['num_queries'] = len(self.state.history)
    stats['queries'] = ('\n' + '-' * 20 + '\n').join(
        [str(entry) for entry in self.state.history])
    return stats

  def reset(
      self,
      index: Optional[Union[int, environment_pb2.GetQueryResponse]] = None,
      documents: Optional[Sequence[environment_pb2.Document]] = None
  ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Mapping[
      str, Any]]:
    """Resets the environment to play a query.

    Args:
      index:  If set as an integer, requests this specific query from the
        environment.  If set as a `GetQueryResponse`, resets for this query.  If
        `None`, a random query will be requested from the environment.
      documents:  If set, uses these results instead of actually querying the
        environment.  Useful during pre-training.

    Returns:
      This function changes the state of the environment, as the name suggests.
      It also returns a tuple of (Observation, InfoDict), where `Observation`
      is the observation for the agent at the beginning of the episode (after
      the original query has been issued).
    """

    if index is None or isinstance(index, int):
      query = self._get_query(index)
    else:
      query = index
    self._logging_info('Original Query [%d/%d]: %s | gold answers: %s.',
                       query.index, query.total, query.query, query.gold_answer)
    self.state = types.EnvState(original_query=query, k=common_flags.K.value)
    self.state.add_history_entry(
        self._get_env_output(
            query=utils.escape_for_lucene(query.query),
            original_query=query,
            documents=documents))

    if (common_flags.RELEVANCE_FEEDBACK_RESTRICT.value == 1 and
        self.training and query.gold_answer):
      target_query = (
          f'{utils.escape_for_lucene(query.query)} '
          f'+(contents:"{utils.escape_for_lucene(query.gold_answer[0])}")')
      self.state.target_documents = self._get_env_output(
          query=target_query, original_query=query).documents

      # Check that the target documents lead to a higher reward otherwise we
      # drop them.
      if self._target_reward() < self._initial_reward():
        self.state.target_documents = None

    # Signal we have not yet built the trie for these results.
    self.known_word_tries = None
    self.valid_word_actions = None
    self.use_rf_restrict = False

    documents = self.state.history[-1].documents
    # If we do not have documents at this point, we have to skip the episode.
    if not documents:
      raise mzcore.SkipEpisode(f'No documents for original query {query.query}')

    self.state.tree = state_tree.NQStateTree(grammar=self.grammar)

    self._logging_info('Initial result score: %s',
                       self._compute_reward(self.state.history[0].documents))

    self.action_history = []
    self.n_episode += 1
    obs = self._obs()
    info = self._info_dict(obs=obs)
    return obs.observation, info

  def _apply_action(self, action) -> None:
    self.state.tree.apply_action(action)

  def _episode_reward(self, metric: str) -> float:
    stats = self.special_episode_statistics_learner(return_as_dict=True)
    self._logging_info('Episode Statistics: %s',
                       '  '.join([f'{k}: {v}' for k, v in stats.items()]))
    if metric not in stats:
      raise NotImplementedError(
          f'Final episode reward for type {metric} is not implemented.')
    return stats[metric]

  def _get_current_documents_list(self,
                                  step: int) -> List[environment_pb2.Document]:
    if common_flags.USE_AGGREGATED_DOCUMENTS.value == 1:
      return get_aggregated_documents(self.state.history[:step])
    else:
      return self.state.history[step - 1].documents

  def _get_previous_documents_list(self,
                                   step: int) -> List[environment_pb2.Document]:
    if common_flags.USE_AGGREGATED_DOCUMENTS.value == 1:
      return get_aggregated_documents(self.state.history[:step - 1])
    else:
      return self.state.history[step - 2].documents

  def _intermediate_reward(self, step: int) -> float:
    """Computes an 'intermediate' reward after each step.

    r_t = S(d_t | q) - S(d_t-1 | q).
    Intermediate rewards (which  make up the bulk of our reward scheme) get
    computed after each step and, usually, quantify the "goodness" of local
    change.  For example, the improvement (or decrease) in a metric such as
    answer F1, relevant document recall, ... between the previous step and
    the current step.

    Args:
      step:  The step for which we want to compute the intermediate reward.

    Returns:
      The intermediate reward, as defined by the configuration.
    """

    assert step >= 2, (f'Intermediate reward computation requires at least 2 '
                       f'history entries. Requested was "{step}".')
    current_documents_list = self._get_current_documents_list(step)
    previous_documents_list = self._get_previous_documents_list(step)

    if common_flags.REWARD.value == 'curiosity+dcg':
      curiosity = len(
          set([d.content for d in current_documents_list]) -
          set([d.content for d in previous_documents_list])) / float(
              common_flags.NUM_DOCUMENTS_TO_RETRIEVE.value)
      dcg_current = self.state.score(
          identifier='dcg',
          documents_list=current_documents_list,
          k=common_flags.K.value)
      dcg_previous = self.state.score(
          identifier='dcg',
          documents_list=previous_documents_list,
          k=common_flags.K.value)
      ideal_dcg = utils.dcg_score(relevances=[1.] * common_flags.K.value)
      ndcg_improvement = (dcg_current - dcg_previous) / ideal_dcg
      return common_flags.REWARD_INTERPOLATION_VALUE.value * curiosity + (
          1 -
          common_flags.REWARD_INTERPOLATION_VALUE.value) * ndcg_improvement

    else:
      reward = common_flags.REWARD.value
      raise NotImplementedError(
          f'Intermediate episode reward for type {reward} is not implemented.')

  def _compute_reward(
      self, current_documents: List[environment_pb2.Document]) -> float:
    if common_flags.REWARD.value == 'curiosity+dcg':
      curiosity = len(
          set([d.content for d in current_documents]) -
          set([d.content for d in self.state.history[0].documents])) / float(
              common_flags.NUM_DOCUMENTS_TO_RETRIEVE.value)
      dcg_current = self.state.score(
          identifier='dcg',
          documents_list=current_documents,
          k=common_flags.K.value)
      ideal_dcg = utils.dcg_score(relevances=[1.] * common_flags.K.value)
      ndcg = dcg_current / ideal_dcg
      return common_flags.REWARD_INTERPOLATION_VALUE.value * curiosity + (
          1 - common_flags.REWARD_INTERPOLATION_VALUE.value) * ndcg

    else:
      reward = common_flags.REWARD.value
      raise NotImplementedError(
          f'Episode reward for type {reward} is not implemented.')

  def _final_reward(self) -> float:
    """Computes the final step reward, r_T = S(d_T | q) - S(d_0 | q).

    Returns:
      The final reward.
    """

    if common_flags.INACTION_PENALTY.value < 0:
      # We penalize the learner for not even issuing a query at all.
      if not len(self.state.history) > 1:
        return common_flags.INACTION_PENALTY.value

    current_documents_list = self.get_final_document_list()
    return self._compute_reward(
        current_documents=current_documents_list) - self._compute_reward(
            current_documents=self.state.history[0].documents)

  def _initial_reward(self):
    return self._compute_reward(
        current_documents=self.state.history[0].documents)

  def _target_reward(self):
    if self.state.target_documents:
      return self._compute_reward(current_documents=self.state.target_documents)
    else:
      return 0

  def _build_trie(self, valid_words: ValidWords) -> pygtrie.Trie:
    # NOTE:  Ensure that the trie only contains words that are fully
    #        constructible.
    return pygtrie.Trie.fromkeys((x for x in map(
        functools.partial(
            to_action_tuple,
            grammar=self.grammar,
            tokenizer=self.tokenizer,
            valid_actions=valid_words.word_piece_actions),
        valid_words.full_words) if x))

  def _info_dict(self, obs: Observations) -> Mapping[str, Any]:
    info = {}
    if common_flags.MASK_ILLEGAL_ACTIONS.value == 1:
      if common_flags.RESTRICT_VOCABULARY.value:
        # Only build the trie and valid words once per set of results.
        if self.valid_word_actions is None:
          self.valid_word_actions = state_tree.ValidWordActions(
              all_word_actions=set(
                  obs.valid_words.all_valid_words.word_piece_actions),
              question_word_actions=set(
                  obs.valid_words.question_valid_words.word_piece_actions),
              answer_word_actions=set(
                  obs.valid_words.answer_valid_words.word_piece_actions),
              document_word_actions=set(
                  obs.valid_words.document_valid_words.word_piece_actions),
              title_word_actions=set(
                  obs.valid_words.title_valid_words.word_piece_actions),
              diff_word_actions=set(
                  obs.valid_words.diff_valid_words.word_piece_actions),
              intersect_word_actions=set(
                  obs.valid_words.intersect_valid_words.word_piece_actions))
        if self.known_word_tries is None:
          self.known_word_tries = state_tree.KnownWordTries(
              local_trie=self._build_trie(obs.valid_words.all_valid_words),
              global_trie=self.global_trie,
              question_trie=self._build_trie(
                  obs.valid_words.question_valid_words),
              answer_trie=self._build_trie(obs.valid_words.answer_valid_words),
              document_trie=self._build_trie(
                  obs.valid_words.document_valid_words),
              title_trie=self._build_trie(obs.valid_words.title_valid_words))
      info['transition_model'] = state_tree.NQTransitionModel(
          full_action_space_size=len(self.grammar.productions()),
          actions=self.state.tree.actions,
          valid_word_actions=self.valid_word_actions,
          known_word_tries=self.known_word_tries,
          grammar=self.grammar,
          restriction_trie=None)
    return info

  def _current_lucene_query(self) -> str:
    """Flattens the `action tree` into a query for the environment."""

    adjustments = self.state.tree.to_adjustments()
    return state_tree.to_lucene_query(
        self.state.original_query.query,
        adjustments,
        escape_query=common_flags.RETRIEVAL_MODE.value.startswith('LUCENE'))

  def _use_relevance_feedback_restrict(self, reward: float) -> bool:
    """Returns true if restricts should be applied for the current step.

    Args:
      reward:  The reward used to determine if the restrict should be applied.
    Note: This uses a linear decay schedule for the moment but we may want to
      consider other strategies.
    """
    if (common_flags.RELEVANCE_FEEDBACK_RESTRICT.value == 1 and
        self.training and reward < 0):
      fraction_of_relevance_feedback_restricted_steps = max(
          0,
          common_flags.FRACTION_OF_RELEVANCE_FEEDBACK_RESTRICTED_STEPS.value
          -
          common_flags.FRACTION_OF_RELEVANCE_FEEDBACK_RESTRICTED_STEPS.value
          / common_flags.MAX_RELEVANCE_FEEDBACK_RESTRICTED_STEPS.value *
          self.training_steps)
      return random.random() < fraction_of_relevance_feedback_restricted_steps
    else:
      return False

  def step(self,
           action: int,
           training_steps: int = 0,
           results: Optional[Sequence[environment_pb2.Document]] = None):
    self.training_steps = training_steps
    if action not in self.state.tree.legal_actions():
      raise types.InvalidActionError

    self._apply_action(action)
    self.action_history.append(action)

    if common_flags.ADD_FINAL_REWARD.value == 1 and self.state.tree.is_complete(
    ):
      # Tree is completed --> [stop] properly generated. Collect the full reward
      # for the result set.
      # Do end computation.
      score, is_done = self._final_reward(), True
      obs = self._obs()
      self._logging_info('Episode end, final reward %s.', score)
      return obs.observation, score, is_done, self._info_dict(obs=obs)

    info, score, is_done = {}, 0.0, False

    if self.state.tree.finished_query():
      # New results, so the local trie / valid word actions are no longer valid.
      self.known_word_tries = None
      self.valid_word_actions = None
      self.use_rf_restrict = False

      current_query = self._current_lucene_query()

      self._logging_info(
          'Reformulation: %s \n with strategy: "%s" | action sequence: %s',
          current_query, ' '.join(self.state.tree.root.leaves()),
          ' '.join(str(a) for a in self.action_history))

      history_entry = self._get_env_output(
          query=current_query,
          original_query=self.state.original_query,
          documents=results)
      if not history_entry.documents:
        self._logging_info(
            'Terminating with negative reward %s due to no results.',
            common_flags.EMPTY_RESULTS_PENALTY.value)
        score, is_done = common_flags.EMPTY_RESULTS_PENALTY.value, True
        obs = self._obs()
        return obs.observation, score, is_done, self._info_dict(obs=obs)

      self.state.add_history_entry(history_entry)

      intermediate_reward = self._intermediate_reward(
          step=self.state.num_completed_requests)
      score = intermediate_reward

      if common_flags.RELEVANCE_FEEDBACK_RESTRICT.value == 1:
        self.use_rf_restrict = self._use_relevance_feedback_restrict(
            reward=intermediate_reward)
        if self.use_rf_restrict:
          # Replay the step with relevance feedback restrict.
          self.state.history.pop()
          self.state.tree.reset_to_previous_finished_query()

      self._logging_info('Intermediate reward: %s, use_rf_restrict=%s',
                         intermediate_reward, self.use_rf_restrict)

      is_done = self.state.tree.is_complete() or len(
          self.state.history) >= common_flags.MAX_NUM_REQUESTS.value

    if len(self.state.tree) > common_flags.MAX_NUM_ACTIONS.value:
      score = -common_flags.INCOMPLETE_TREE_PENALTY.value
      self._logging_info(
          'Maxed out tree construction steps. Terminating with '
          'penalty %s.', common_flags.INCOMPLETE_TREE_PENALTY.value)
      is_done = True

    # complete once 'stop' was issued (i.e. all non-terminals expanded)
    is_done = is_done or self.state.tree.is_complete()

    obs = self._obs()
    info = {**info, **self._info_dict(obs=obs)}
    return obs.observation, score, is_done, info

  def render(self, mode='human'):
    if mode == 'human':
      print(self.state)
      return
    elif mode == 'rgb_array':
      raise NotImplementedError()
    elif mode == 'ansi':
      return str(self.state)
    else:
      raise ValueError('mode not found: {}'.format(mode))

  def visualize_mcts(self, root: mzcore.Node,
                     history: Sequence[types.HistoryEntry]) -> str:
    """Visualize the MCTS corresponding to `root` and `history`.

    The returned string can be visualized using the `forest` LaTex package
    (https://ctan.org/pkg/forest?lang=en).

    Args:
      root:  Root node for the Monte-Carlo Tree search.
      history:  History from the Agent's state.

    Returns:
      A `forest`-package compatible string representing the Monte-Carlo Tree
      search tree.
    """

    # Only return a visualization for these history lengths.
    if (common_flags.VISUALIZE_MCTS.value == 0
        or len(history) not in [0, 3, 7, 12, 25]):
      return ''

    return types.visualize_mcts(
        root=root,
        history=history,
        grammar=self.grammar,
        min_visit_count=common_flags.VISUALIZE_MIN_VISIT_COUNT.value)


def create_environment(task: int,
                       training: bool = True,
                       stop_after_seeing_new_results: bool = False) -> NQEnv:
  """Creates an NQEnvironment.

  Args:
    task:  Used to set the random seed of the environment.  In SeedRL context,
      it makes sense to pass in the task number of the actor (hence the argument
      name), but really, this is just a random seed.
    training:  Whether this environment should serve training or development
      episodes.
    stop_after_seeing_new_results:  If False, [stop] means that you return the
      latest results (corresponding to your last query).  If True, [stop] means
      that you disregard the latest results and return the pen-ultimate results.

  Returns:
    An NQEnv instance.
  """
  logging.info('Creating environment: nq')
  env = NQEnv(
      training=training,
      nq_server=server.get_nq_server(),
      stop_after_seeing_new_results=stop_after_seeing_new_results)
  env.seed(task)
  return env


def get_aggregated_documents(history_list: List[types.HistoryEntry]):
  unique_documents = {}
  for history in history_list:
    for doc in history.documents:
      unique_documents[doc.content] = doc
  docs = list(unique_documents.values())
  ranked_docs = sorted(docs, key=lambda doc: doc.answer.mr_score, reverse=True)
  return ranked_docs[:common_flags.NUM_DOCUMENTS_TO_RETRIEVE.value]


def make_bert_state(
    environment_state: types.EnvState,
    tokenizer,
    idf_lookup: Mapping[str, float],
    use_target_documents: bool = False) -> List[utils.ObsFragment]:

  if use_target_documents:
    documents = environment_state.target_documents
  else:
    if common_flags.USE_AGGREGATED_DOCUMENTS.value == 1:
      documents = get_aggregated_documents(environment_state.history)
    else:
      documents = environment_state.history[-1].documents

  return bert_state_lib.make_bert_state_impl(
      query=environment_state.original_query.query,
      tree=environment_state.tree,
      documents=documents,
      idf_lookup=idf_lookup,
      context_size=common_flags.CONTEXT_WINDOW_SIZE.value,
      tokenize_fn=tokenizer.tokenize,
      max_length=common_flags.BERT_SEQUENCE_LENGTH.value,
      max_title_length=common_flags.CONTEXT_TITLE_SIZE.value)

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
"""Library for computing the bert state."""

import collections
import os.path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Sequence

from absl import logging
from language.search_agents import environment_pb2
from language.search_agents.muzero import common_flags
from language.search_agents.muzero import state_tree
from language.search_agents.muzero import types
from language.search_agents.muzero import utils as nqutils
import nltk
import numpy as np
import tensorflow as tf
import transformers

# Each type vocab is an additional BERT embedding.
TYPE_VOCABS = collections.OrderedDict(
    state_part=[
        '[PAD]',
        '[CLS]',
        '[SEP]',
        'original_query',
        'tree_stack',
        'tree_leaves',
        'history_query',
        'history_answer',
        'history_context',
        'history_title',
    ],)
# Each float_name is an additional BERT embedding
FLOAT_NAMES = ['mr_score', 'idf_score']

TokenizeFn = Callable[[str], List[str]]
TokensToIdsFn = Callable[[Sequence[str]], List[int]]
IdfLookupFn = Mapping[str, float]


def get_tokenizer() -> transformers.BertTokenizer:
  """Returns the BERT tokenizer."""
  do_lower_case = ('uncased' in common_flags.BERT_CONFIG.value or
                   'cased' not in common_flags.BERT_CONFIG.value)
  with tempfile.TemporaryDirectory() as tdir:
    vocab_fn = os.path.join(tdir, 'vocab.txt')
    tf.io.gfile.copy(common_flags.BERT_VOCAB.value, vocab_fn)

    # special symbols
    grammar_symbols = state_tree.NQStateTree.tree_node_symbols
    operator_symbols = [operator.value for operator in state_tree.Operator
                       ] + ['[stop]']
    # add field symbols
    operator_symbols += [field.value for field in state_tree.Field]
    special_symbols = grammar_symbols + operator_symbols
    assert len(special_symbols) < 99, 'Too many special symbols.'

    tokenizer = transformers.BertTokenizer(
        vocab_fn, do_lower_case=do_lower_case)
    tokenizer.add_tokens(special_symbols)
  return tokenizer


def compute_bert_state(query: str,
                       documents: Sequence[environment_pb2.Document],
                       idf_lookup: IdfLookupFn, context_size: int,
                       title_size: int, seq_length: int,
                       tokenize_fn: TokenizeFn,
                       tokens_to_ids_fn: TokensToIdsFn) -> Dict[str, Any]:
  """Computes the BERT state as in the agent."""

  obs_fragments = make_bert_state_impl(
      query=query,
      tree=state_tree.NQStateTree(nltk.CFG.fromstring("Q -> 'unused'")),
      documents=documents,
      idf_lookup=idf_lookup,
      tokenize_fn=tokenize_fn,
      context_size=context_size,
      max_length=seq_length,
      max_title_length=title_size)

  token_ids, type_ids, float_values = nqutils.ObsFragment.combine_and_expand(
      fragments=obs_fragments,
      length=seq_length,
      type_vocabs=TYPE_VOCABS,
      float_names=FLOAT_NAMES,
      tokens_to_id_fn=tokens_to_ids_fn,
  )
  token_ids = np.array(token_ids, np.int32)
  type_ids = np.array(type_ids, np.int32).T
  float_values = np.array(float_values, np.float32).T

  return {
      'obs_fragments': obs_fragments,
      'token_ids': token_ids,
      'type_ids': type_ids,
      'float_values': float_values,
  }


def original_query_state_part(query: str, tokenize_fn: TokenizeFn,
                              idf_lookup: IdfLookupFn) -> nqutils.ObsFragment:
  """Computes the original query part of the BERT state."""
  original_tokens = tokenize_fn(query)

  return nqutils.ObsFragment(
      text=nqutils.Text(tokens=[
          '[CLS]',
          *original_tokens,
          '[SEP]',
      ]),
      type_values={
          'state_part': ['[CLS]'] + ['original_query'] * len(original_tokens) +
                        ['[SEP]'],
      },
      float_values={
          'mr_score': [0.0] * (len(original_tokens) + 2),
          'idf_score': [0.] + [
              idf_lookup[token]
              for token in nqutils.bert_tokens_to_words(original_tokens)
          ] + [0.]
      },
  )


def state_tree_state_part(tree: state_tree.NQStateTree,
                          idf_lookup: IdfLookupFn) -> nqutils.ObsFragment:
  """Computes the state tree part of the BERT state."""
  leave_tokens = tree.root.leaves()

  return nqutils.ObsFragment(
      text=nqutils.Text(tokens=[
          *leave_tokens,
          '[SEP]',
      ]),
      type_values={
          'state_part': ['tree_leaves'] * len(leave_tokens) + ['[SEP]'],
      },
      float_values={
          'mr_score': [0.0] * (len(leave_tokens) + 1),
          'idf_score': [
              idf_lookup[token]
              for token in nqutils.bert_tokens_to_words(leave_tokens)
          ] + [0.]
      },
  )


def history_state_part(documents: Sequence[environment_pb2.Document],
                       tokenize_fn: TokenizeFn, idf_lookup: IdfLookupFn,
                       context_size: int, max_length: int,
                       max_title_length: int) -> List[nqutils.ObsFragment]:
  """Computes the history part of the BERT state."""
  history_fragments = []

  for doc in documents:
    answer_tokens = tokenize_fn(doc.answer.answer if doc.answer else '')
    answer_tokens_words = nqutils.bert_tokens_to_words(answer_tokens)
    _, context_window = types.HistoryEntry.get_window_around_substr(
        doc.content, doc.answer.answer, context_size)
    context_tokens = tokenize_fn(context_window)
    context_tokens_words = nqutils.bert_tokens_to_words(context_tokens)
    title_tokens = tokenize_fn(doc.title if doc.title else '')
    title_tokens = title_tokens[:max_title_length]
    title_tokens_words = nqutils.bert_tokens_to_words(title_tokens)
    state_part = [
        p for tokens, part in
        zip([answer_tokens, context_tokens, title_tokens],
            ['history_answer', 'history_context', 'history_title'])
        for p in [part] * len(tokens) + ['[SEP]']
    ]

    length = len(context_tokens) + len(answer_tokens) + len(title_tokens) + 3
    history_fragments.append(
        nqutils.ObsFragment(
            text=nqutils.Text(tokens=[
                *answer_tokens,
                '[SEP]',
                *context_tokens,
                '[SEP]',
                *title_tokens,
                '[SEP]',
            ]),
            type_values={
                'state_part': state_part,
            },
            float_values={
                'mr_score': [doc.answer.mr_score] * length,
                'idf_score':
                    [idf_lookup[word] for word in answer_tokens_words] + [0.] +
                    [idf_lookup[word] for word in context_tokens_words] + [0.] +
                    [idf_lookup[word] for word in title_tokens_words] + [0.]
            },
        ))
    # stop early if we already have enough history to fill up the whole state
    if sum([len(fragment.text.tokens) for fragment in history_fragments
           ]) >= max_length:
      logging.info('BERT state reached max_length: %d', max_length)
      break

  return history_fragments


def make_bert_state_impl(query: str, tree: state_tree.NQStateTree,
                         documents: Sequence[environment_pb2.Document],
                         idf_lookup: IdfLookupFn, context_size: int,
                         tokenize_fn: TokenizeFn, max_length: int,
                         max_title_length: int) -> List[nqutils.ObsFragment]:
  """Computes the BERT state."""
  query_fragment = original_query_state_part(
      query=query, tokenize_fn=tokenize_fn, idf_lookup=idf_lookup)

  tree_fragment = state_tree_state_part(tree=tree, idf_lookup=idf_lookup)

  history_fragments = history_state_part(
      documents=documents,
      tokenize_fn=tokenize_fn,
      idf_lookup=idf_lookup,
      context_size=context_size,
      max_length=max_length,
      max_title_length=max_title_length)

  return [query_fragment, tree_fragment] + history_fragments

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
"""Defines grammars for the search agent."""

import enum
from typing import Collection, List

from absl import logging
import dataclasses
from language.search_agents.muzero import common_flags
from language.search_agents.muzero import state_tree


class GrammarType(enum.Enum):
  """Support grammar types."""
  # Relevance feedback with both + action and - action at each step, e.g.
  # +(title:<term>) -(contents:<term>).
  BERT = 1
  # Only + action OR - action at each step, e.g. +(title:<term>).
  ONE_TERM_AT_A_TIME = 2
  # Only add terms at each step, e.g. <term>.
  # This is equivalent to using an OR operator.
  ADD_TERM_ONLY = 3
  # Pick either + action, - action, OR term only at each step.
  # This combines the ONE_TERM_AT_A_TIME and ADD_TERM_ONLY grammars above.
  ONE_TERM_AT_A_TIME_WITH_ADD_TERM_ONLY = 4


@dataclasses.dataclass
class GrammarConfig:
  """Configures a grammar to govern reformulations."""

  grammar_type: GrammarType
  split_vocabulary_by_type: bool


def grammar_config_from_flags() -> GrammarConfig:
  return GrammarConfig(
      grammar_type={
          'bert':
              GrammarType.BERT,
          'one_term_at_a_time':
              GrammarType.ONE_TERM_AT_A_TIME,
          'add_term_only':
              GrammarType.ADD_TERM_ONLY,
          'one_term_at_a_time_with_add_term_only':
              GrammarType.ONE_TERM_AT_A_TIME_WITH_ADD_TERM_ONLY,
      }[common_flags.GRAMMAR_TYPE.value],
      split_vocabulary_by_type= \
        common_flags.SPLIT_VOCABULARY_BY_TYPE.value == 1)


def get_term_types():
  """Returns the vocabulary types for a `Word` non-terminal (_W_)."""
  term_types = ['_Wq_', '_Wa_', '_Wd_']
  if common_flags.USE_DOCUMENT_TITLE.value == 1:
    term_types.append('_Wt_')
  return term_types


def _make_bert_vocab_productions(vocab: Collection[str]) -> List[str]:
  """Generates productions for the `unconstrained` vocabulary setting.

  To ensure that non-initial word-pieces never begin a word, we use a grammar
  corresponding to the following rule:

    Word -->  InitialWordPiece (NonInitialWordPiece)*

  We use the following non-terminals:

    _W_:   corresponds to `Word`
    _Vsh_: pre-terminal corresponidng to `NonInitialWordPiece`
    _Vw_:  pre-terminal corresponding to `InitialWordPiece`
    _W-_:  helper non-terminal, implementing the generation of
           `NonInitialWordPiece`s.


  Args:
    vocab:  Iterable comprising all valid terminals.  This spans both wordpieces
      and `full` words.

  Returns:
    A list of productions in textual form which jointly define the "vocabulary"
    part of a grammar that expands the _W_(ord) non-terminal to terminal symbols
    in `vocab`.
  """

  productions = []

  productions.append('_W_ -> _Vw_ _W-_')
  productions.append('_W_ -> _Vw_')
  productions.append('_W-_ -> _Vsh_')
  productions.append('_W-_ -> _Vsh_ _W-_')

  for word in vocab:
    word = state_tree.NQStateTree.clean_escape_characters(word)
    if word.startswith('##'):
      productions.append("_Vsh_ -> '{}'".format(word))
    else:
      # No point in having the agent generate these "fake" tokens.
      if word.startswith('[unused') or word in ('[pos]', '[neg]', '[contents]',
                                                '[title]', '[UNK]', '[PAD]',
                                                '[SEP]', '[CLS]', '[MASK]'):
        continue
      productions.append("_Vw_ -> '{}'".format(word))

  return productions


def _expand_vocab_type_grammar(grammar_productions: List[str]) -> List[str]:
  """Expands the `Word` non-terminal to question|answer|document subtypes.

  Args:
    grammar_productions: Current grammar with the basic `Word` non-terminal.

  Returns:
    Rules where each `Word` non-terminal (_W_) is expanded into question (_Wq_),
    answer (_Wa_), document (_Wd_), and document title (_Wt_) subtypes.
  """
  productions = []
  for production in grammar_productions:
    if '_W_' in production:
      for term_type in get_term_types():
        productions.append(production.replace('_W_', term_type))
    else:
      productions.append(production)
  return productions


def construct_grammar(grammar_config: GrammarConfig,
                      vocab: Collection[str]) -> state_tree.NQCFG:
  """Builds the grammar according to `grammar_config`."""

  productions = []

  # Lexical rules.
  if grammar_config.grammar_type in (
      GrammarType.BERT, GrammarType.ONE_TERM_AT_A_TIME,
      GrammarType.ADD_TERM_ONLY,
      GrammarType.ONE_TERM_AT_A_TIME_WITH_ADD_TERM_ONLY):
    productions.extend(_make_bert_vocab_productions(vocab=vocab))

  # "Internal" rules.
  if grammar_config.grammar_type == GrammarType.BERT:
    for field_add in ('[title]', '[contents]'):
      for field_sub in ('[title]', '[contents]'):
        productions.append(
            f"_Q_ -> '[pos]' '{field_add}' _W_ '[neg]' '{field_sub}' _W_ _Q_")
  elif grammar_config.grammar_type == GrammarType.ONE_TERM_AT_A_TIME:
    for field in ('[title]', '[contents]'):
      productions.append(f"_Q_ -> '[pos]' '{field}' _W_ _Q_")
      productions.append(f"_Q_ -> '[neg]' '{field}' _W_ _Q_")
  elif grammar_config.grammar_type == GrammarType.ADD_TERM_ONLY:
    productions.append("_Q_ -> '[or]' _W_ _Q_")
  elif grammar_config.grammar_type == GrammarType.ONE_TERM_AT_A_TIME_WITH_ADD_TERM_ONLY:
    for field in ('[title]', '[contents]'):
      productions.append(f"_Q_ -> '[pos]' '{field}' _W_ _Q_")
      productions.append(f"_Q_ -> '[neg]' '{field}' _W_ _Q_")
    productions.append("_Q_ -> '[or]' _W_ _Q_")
  else:
    raise NotImplementedError(
        'The grammar vocabulary type {} is not implemented.'.format(
            grammar_config.grammar_type))

  # Always add the stop action.
  productions.append("_Q_ -> '[stop]'")

  if grammar_config.split_vocabulary_by_type:
    productions = _expand_vocab_type_grammar(grammar_productions=productions)

  grammar_str = ' \n '.join(productions)
  grammar = state_tree.NQCFG(grammar_str)
  grammar.set_start(grammar.productions()[-1].lhs())

  logging.info('Grammar: %s', grammar.productions())

  return grammar

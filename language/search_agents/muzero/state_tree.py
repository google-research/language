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
# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension
# pylint: disable=g-long-lambda
"""NQ State Tree."""

import collections
import copy
import enum
import functools
import re
from typing import Collection, List, Dict, Optional, Sequence, Tuple, Callable

from absl import logging
import dataclasses
from language.search_agents.muzero import utils
import nltk
import pygtrie

from muzero import core


class Operator(enum.Enum):
  """The lucene operator."""
  MINUS = '[neg]'
  PLUS = '[pos]'
  APPEND = '[or]'


class Field(enum.Enum):
  """The lucene field."""
  TITLE = '[title]'
  CONTENTS = '[contents]'
  ALL = '[all]'


@dataclasses.dataclass
class QueryAdjustment:
  """One single adjustment to the query."""
  operator: Operator
  field: Field
  term: str
  term_type: str = '_W_'


@dataclasses.dataclass
class KnownWordTries:
  local_trie: pygtrie.Trie
  global_trie: pygtrie.Trie
  question_trie: pygtrie.Trie
  answer_trie: pygtrie.Trie
  document_trie: pygtrie.Trie
  title_trie: pygtrie.Trie


@dataclasses.dataclass
class ValidWordActions:
  all_word_actions: Collection[int]
  question_word_actions: Collection[int]
  answer_word_actions: Collection[int]
  document_word_actions: Collection[int]
  title_word_actions: Collection[int]
  diff_word_actions: Collection[int]
  intersect_word_actions: Collection[int]


class NQCFG(nltk.CFG):
  """Context-free grammar for NQ environment."""

  def __init__(self,
               grammar_str: str,
               restrictions: Optional[Dict[str, List[str]]] = None):
    r"""Context Free Grammar for the NQ environment.

    Args:
      grammar_str: A grammar string in the form required by nltk.CFG.fromstring.
        E.g. "A -> B \n B -> 'c'"
      restrictions: A dictionary that maps from the string representation of the
        last two terminals to a list of identifiers for the allowed
        vocabularies. E.g. {'[pos][contents]': ['Vap', 'Vat'],}

    Returns:
      Context Free Grammar for the NQ environment.
    """

    self.nonterminals = set(
        [production.lhs() for production in self.productions()])
    self.production_lookup = collections.defaultdict(list)
    # Maps a rule to the corresponding action.
    self.rule_to_action = {}
    # Maps an action back to the rule.
    self.action_to_rule = {}
    # Maps a terminal (word piece) to its (corresponding rule's) action.
    self.terminal_to_action = {}

    # Associate each production with an integer which corresponds to the id of
    # an agent action.
    for i, production in enumerate(self.productions()):
      self.production_lookup[str(production.lhs())].append((i, production))
      self.rule_to_action[str(production)] = i
      self.action_to_rule[str(i)] = str(production)
      if production.is_lexical() and len(production.rhs()) == 1:
        self.terminal_to_action[production.rhs()[0]] = i

    self.min_steps_to_terminal = {
        non_terminal: self._min_steps_to_terminal(non_terminal)
        for non_terminal in self.nonterminals
    }
    self.restrictions = restrictions

  @property
  def stop_action(self):
    return self.rule_to_action["_Q_ -> '[stop]'"]

  def _min_steps_to_terminal(self, node):
    if isinstance(node, str):
      return 0
    productions = filter(lambda x: node not in x.rhs(),
                         self.productions(lhs=node))
    min_steps = float('inf')
    for prod in productions:
      steps = sum(
          [self._min_steps_to_terminal(child) + 1 for child in prod.rhs()])
      min_steps = min(min_steps, steps)
      if min_steps == 1:
        break
    return min_steps

  def __new__(cls, grammar_str, *args, **kwargs):
    grammar = nltk.CFG.fromstring(grammar_str)
    grammar.__class__ = cls
    return grammar

  def set_start(self, start):
    self._start = start

  def terminals(self):
    return [
        term for p in self.productions() for term in p.rhs()
        if isinstance(term, str)
    ]

  @classmethod
  def fromdict(cls, grammar_dict: Dict[str, List[str]], *args, **kwargs):
    # Build grammar string from grammar_dict
    grammar_str = '\n'.join([
        '{} -> {}'.format(lhs, ' | '.join(rhs_list))
        for lhs, rhs_list in grammar_dict.items()
    ])
    return cls(grammar_str, *args, **kwargs)

  def production_to_action(self, production: nltk.Production) -> int:
    return self.rule_to_action[str(production)]

  def query_adjustment_to_action_sequence(
      self,
      query_adjustment,
      answer_words: Sequence[str],
      title_words: Sequence[str],
      question_words: Sequence[str],
      document_words: Sequence[str],
      tokenize_fn: Callable[[str], Sequence[str]],
      fallback_term_type: Optional[str] = None) -> Tuple[bool, Sequence[int]]:
    """Get the action sequence from a single QueryAdjustment."""

    def _term_type_from_action_sequence(action_sequence):
      """Get the term_type from a step action sequence."""
      for action in action_sequence:
        lhs = str(self.productions()[action].lhs())
        if lhs in ['_Wa_', '_Wt_', '_Wq_', '_Wd_']:
          return lhs
      return None

    query_addon = []
    query_addon.extend([query_adjustment.operator.value])
    if query_adjustment.operator != Operator.APPEND:
      query_addon.extend([query_adjustment.field.value])
    query_addon.extend(tokenize_fn(query_adjustment.term))

    # Find the vocabulary the word came from.
    if query_adjustment.term in answer_words:
      term_type = '_Wa_'
    elif query_adjustment.term in title_words:
      term_type = '_Wt_'
    elif query_adjustment.term in question_words:
      term_type = '_Wq_'
    elif query_adjustment.term in document_words:
      term_type = '_Wd_'
    else:
      # The term is not in one of the legal vocabularies.
      if fallback_term_type:
        logging.warning(
            "Couldn't find term type for term '%s'. Using fallback term type '%s'",
            query_adjustment.term, fallback_term_type)
        term_type = fallback_term_type
      else:
        return False, []

    parser = nltk.parse.BottomUpChartParser(grammar=self)
    for parsed_step in parser.parse(query_addon + ['[stop]']):
      action_sequence = [
          self.production_to_action(production)
          for production in parsed_step.productions()
      ]
      if term_type == _term_type_from_action_sequence(action_sequence):
        return True, action_sequence[:-1]

    # Couldn't find a parse
    return False, []


class NQStateTree:
  """State Tree for the NQ environment.

  The tree is constructed using rules from a context-free grammar. It can be
  exported to a Lucene or a BERT query.
  """
  tree_node_symbols = [
      # Pre-terminals.
      '_Vw_',  # Initial WordPiece.
      '_Vsh_',  # Non-initial WordPiece.

      # Non-terminals.
      '_Q_',  # Query - start symbol.
      '_W_',  # Word.
      '_Wq_',  # Word from the question.
      '_Wa_',  # Word from answers in the state.
      '_Wd_',  # Word from documents in the state.
      '_Wt_',  # Word from document titles in the state.
      '_W-_',  # Controls word-piece recursion.
  ]

  escape_characters = {
      '\'': '_sq_',
  }

  def __init__(self, grammar: nltk.CFG):
    self.grammar = grammar
    self.reset()

  def _find_node_to_expand(self):
    """Finds the next node in the tree that is not yet fully expanded."""
    return self.stack[-1] if self.stack else None

  def is_complete(self):
    return self._find_node_to_expand() is None

  def is_stop_action(self, action: int) -> bool:
    return self.grammar.productions()[action].rhs()[0] == '[stop]'

  def finished_query(self):
    return bool(len(self.stack) == 1 and self.root.leaves())

  def start_adjustment(self):
    return len(self.stack) == 1

  def _min_remaining_steps_from_stack(self, stack):
    if not stack:
      return 0
    return sum(
        self.grammar.min_steps_to_terminal[node.label()] for node in stack)

  def min_remaining_steps(self):
    return self._min_remaining_steps_from_stack(self.stack)

  def legal_actions(self):
    """Return only rules having the to-be-expanded non-terminal as lhs."""
    current_node = self._find_node_to_expand()
    if current_node is None:
      return []
    legal_actions = self.grammar.production_lookup[str(current_node.label())]
    return [idx for idx, legal_action in legal_actions]

  def legal_actions_mask(self):
    legal_actions = self.legal_actions()
    return [
        1 if i in legal_actions else 0
        for i in range(len(self.grammar.productions()))
    ]

  def apply_production(self, production: nltk.Production):
    if production not in self.grammar.productions():
      raise ValueError(f'Production {production} not in grammar of the tree.')
    return self._expand_node(production)

  def apply_action(self, action: int):
    production = self.grammar.productions()[action]
    return self._expand_node(production)

  def reset(self):
    self.root = nltk.Tree(self.grammar.start(), [])
    self.stack = [self.root]
    self.actions = []

  def tree_str(self):
    stack_str = ' '.join([str(node.label()) for node in self.stack])
    tree_str = ' [SEP] '.join([stack_str, ' '.join(self.root.leaves())])
    return tree_str

  def to_adjustments(self) -> List[QueryAdjustment]:
    """Group the yield into `QueryAdjustments`."""
    leaves = self.root.leaves()
    adjustments = []

    while leaves:
      leaf = leaves.pop(0)
      if leaf not in [operator.value for operator in Operator] or not leaves:
        continue
      if leaves[0] in ['[title]', '[contents]']:
        field = leaves.pop(0)
        term = leaves.pop(0)
      else:
        term = leaves.pop(0)
        field = '[all]'
      # in case it is a subword token
      while leaves and leaves[0].startswith('##'):
        term += leaves.pop(0).replace('##', '')
      adjustments.append(QueryAdjustment(Operator(leaf), Field(field), term))
    return adjustments

  def reset_to_previous_finished_query(self):
    end = None
    for i, production in reversed(list(enumerate(self.actions))):
      if production.lhs() == nltk.Nonterminal('_Q_'):
        end = i
        break
    productions = self.actions[:end]
    self.reset()
    for production in productions:
      self.apply_production(production)

  def _expand_node(self, production: nltk.Production):
    current_node = self.stack.pop()
    if production.lhs() == current_node.label():
      self._append(current_node, production.rhs())
      self.actions.append(production)
    else:
      self.stack.append(current_node)
      raise ValueError(
          f'Rule is not applicable: {production}, stack: {self.stack}.')

  def _append(self, node: nltk.Tree, children):
    add_to_stack = []
    for child in children:
      if nltk.grammar.is_nonterminal(child):
        new_node = nltk.Tree(child, [])
        node.append(new_node)
        add_to_stack.append(new_node)
      else:
        node.append(child)
    if add_to_stack:
      self.stack.extend(add_to_stack[::-1])

  def __repr__(self):
    return self.tree_str()

  def __len__(self):
    return len(self.actions)

  @staticmethod
  def clean_escape_characters(word):
    for k, v in NQStateTree.escape_characters.items():
      word = word.replace(k, v)
    return word


class NQTransitionModel(core.TransitionModel):
  """Specialized transition model for NQ used inside the MCTS.

  This model provides information about the legal actions after a sequence of
  actions. This model is given to the agent and used inside the MCTS.
  """

  def __init__(self,
               full_action_space_size: int,
               actions: List[nltk.grammar.Production],
               grammar: NQCFG,
               valid_word_actions: Optional[ValidWordActions] = None,
               known_word_tries: Optional[KnownWordTries] = None,
               restriction_trie: Optional[pygtrie.Trie] = None):
    super().__init__(full_action_space_size)

    self.grammar = grammar

    self.actions = tuple(self.grammar.production_to_action(p) for p in actions)
    self.valid_word_actions = valid_word_actions
    self.known_word_tries = known_word_tries
    self.restriction_trie = restriction_trie

    self.initial_stack = self.apply_actions_to_stack(
        stack=(self.grammar.start(),), actions=self.actions)

    # Wrap methods with lru cache for each instance separately.
    # This is important so the memory used by the cache gets freed correctly.
    self.apply_actions_to_stack = functools.lru_cache(maxsize=512)(
        self.apply_actions_to_stack)
    self.legal_actions_after_sequence = functools.lru_cache(maxsize=512)(
        self.legal_actions_after_sequence)
    self.filter_legal_actions_for_vw_node_crossed_search_boundary = (
        functools.lru_cache(maxsize=8)(
            self.filter_legal_actions_for_vw_node_crossed_search_boundary))
    self.filter_legal_actions_for_vsh_node_crossed_search_boundary = (
        functools.lru_cache(maxsize=512)(
            self.filter_legal_actions_for_vsh_node_crossed_search_boundary))

  @staticmethod
  def filter_legal_actions_for_vw_node(legal_actions: Tuple[int],
                                       is_recursive: bool,
                                       current_trie: pygtrie.Trie):
    if is_recursive:
      # We need to check that only pieces which can be extended are valid.
      legal_actions = [
          action for action in legal_actions
          if current_trie.has_subtrie((action,))
      ]
    else:
      # We need to check that only pieces which yield a valid word are
      # valid.
      legal_actions = [
          action for action in legal_actions if current_trie.has_key((action,))
      ]
    return legal_actions

  def filter_legal_actions_for_vw_node_crossed_search_boundary(
      self, legal_actions: Tuple[int], is_recursive: bool):
    return self.filter_legal_actions_for_vw_node(
        legal_actions, is_recursive, self.known_word_tries.global_trie)

  @staticmethod
  def filter_legal_actions_for_vsh_node(legal_actions: Tuple[int],
                                        is_recursive: bool,
                                        word_prefix: Tuple[int],
                                        current_trie: pygtrie.Trie):
    if is_recursive:
      # We need to check that only pieces which can be extended are valid.
      legal_actions = [
          action for action in legal_actions
          if current_trie.has_subtrie(word_prefix + (action,))
      ]
    else:
      # We need to check that only pieces which yield a valid word are
      # valid.
      legal_actions = [
          action for action in legal_actions
          if current_trie.has_key(word_prefix + (action,))
      ]
    return legal_actions

  def filter_legal_actions_for_vsh_node_crossed_search_boundary(
      self, legal_actions: Tuple[int], is_recursive: bool,
      word_prefix: Tuple[int]):
    return self.filter_legal_actions_for_vsh_node(
        legal_actions, is_recursive, word_prefix,
        self.known_word_tries.global_trie)

  def apply_action_to_stack(self, stack: Tuple[nltk.Nonterminal],
                            action: int) -> Tuple[nltk.Nonterminal, ...]:
    """Expand `stack` with the rule corresponding to `action`.

    Args:
      stack:  A sequence of `nltk.Nonterminal`s representing the current stack.
      action:  An action corresponding to a rule.

    Returns:
      A _new_ stack representing the result of applying the production
      corresponding to `action` to the top of `stack`.  If this is invalid
      (because the top of `stack` != the lhs of production), raises an error.
    """
    non_terminal_on_top_of_stack = stack[-1]
    stack_remainder = tuple(stack[:-1])

    production = self.grammar.productions()[action]
    if production.lhs() != non_terminal_on_top_of_stack:
      raise ValueError(
          f'Expected rule that expands {non_terminal_on_top_of_stack}, got: '
          f'{production}, stack={stack}')

    # Return the stack that results by replacing the top symbol with all the
    # non-terminals on the rhs of the applied rule, _in reverse order_.
    return stack_remainder + tuple(
        rhs for rhs in reversed(production.rhs())
        if nltk.grammar.is_nonterminal(rhs))

  def apply_actions_to_stack(
      self, stack: Tuple[nltk.Nonterminal],
      actions: Sequence[int]) -> Tuple[nltk.Nonterminal, ...]:
    """Expand `stack` with the rules corresponding to `actions`.

    Args:
      stack:  A sequence of `nltk.Nonterminal`s representing the current stack.
      actions:  A sequence of actions.

    Returns:
      A _new_ stack representing the result of applying all the productions
      corresponding to `actions` to the top of `stack` in order.  If this is
      invalid (because the top of `stack` != the lhs of production) at any
      point, raises an error.
    """
    for action in actions:
      stack = self.apply_action_to_stack(stack, action)
      # Completed a tree.  Need to seed the stack with a start symbol.
      if not stack:
        stack = (self.grammar.start(),)
    return stack

  def legal_actions_for_stack(self,
                              stack: Sequence[nltk.Nonterminal]) -> Tuple[int]:
    legal_actions = self.grammar.production_lookup[str(stack[-1])]
    return tuple(idx for idx, legal_action in legal_actions)

  def legal_actions_after_sequence(self,
                                   actions_sequence: Optional[Tuple[int]]):
    """Returns the legal action space after a sequence of actions."""
    stack = self.apply_actions_to_stack(self.initial_stack, actions_sequence)
    full_tree_action_history = self.actions + actions_sequence

    original_legal_actions = self.legal_actions_for_stack(stack=stack)
    legal_actions = original_legal_actions

    def _find_word_prefix():

      # Find the latest `_W_ -> *` rule, if any.
      w_pos = None
      for pos in range(len(full_tree_action_history) - 1, -1, -1):
        rule = self.grammar.action_to_rule[str(full_tree_action_history[pos])]
        if (rule.startswith('_W_') or rule.startswith('_Wq_') or
            rule.startswith('_Wa_') or rule.startswith('_Wd_') or
            rule.startswith('_Wt_')):
          w_pos = pos
          break
      if w_pos is None:
        return ()

      # Collect all the terminals generated so far.
      word_prefix = []
      for pos in range(w_pos + 1, len(full_tree_action_history)):
        if self.grammar.productions()[
            full_tree_action_history[pos]].is_lexical():
          word_prefix.append(full_tree_action_history[pos])
      return tuple(word_prefix)

    def _compute_crossed_search_boundary():
      # We crossed a search boundary if at any point during the actions sequence
      # we have another grammar start on top of the stack.
      for i in range(len(actions_sequence)):
        intermediate_stack = self.apply_actions_to_stack(
            self.initial_stack, actions_sequence[:i + 1])
        if intermediate_stack[-1] == self.grammar.start():
          return True
      return False

    def _find_word_action():
      # Find the latest rule that indicates the type of `Word` to use.
      for pos in range(len(full_tree_action_history) - 1, -1, -1):
        rule = self.grammar.action_to_rule[str(full_tree_action_history[pos])]
        if '_W_' in rule:
          return '_W_'
        if '_Wq_' in rule:
          return '_Wq_'
        if '_Wa_' in rule:
          return '_Wa_'
        if '_Wd_' in rule:
          return '_Wd_'
        if '_Wt_' in rule:
          return '_Wt_'
      return ''

    def _find_operator():
      # Find the latest rule that indicates the type of operator to use.
      for pos in range(len(full_tree_action_history) - 1, -1, -1):
        rule = self.grammar.action_to_rule[str(full_tree_action_history[pos])]
        if '[pos]' in rule:
          return '[pos]'
        if '[neg]' in rule:
          return '[neg]'
        if '[or]' in rule:
          return '[or]'
      return ''

    def _can_construct(pieces, valid_actions) -> bool:
      if crossed_search_boundary:
        return True
      return all(piece in valid_actions for piece in pieces)

    def _get_restricted_trie(rule: str, trie: pygtrie.Trie) -> pygtrie.Trie:
      if trie:
        if '[pos]' in rule or '[or]' in rule:
          # For [pos] or [or] operators, remove words from input trie that are
          # not valid intersect_word_actions.
          valid_word_actions = self.valid_word_actions.intersect_word_actions
        elif '[neg]' in rule:
          # For [neg] rules, remove words from the input trie that are not
          # valid diff_word_actions.
          valid_word_actions = self.valid_word_actions.diff_word_actions
        else:
          valid_word_actions = None
        if valid_word_actions:
          result = copy.deepcopy(trie)
          for word in trie.iterkeys(()):
            if not _can_construct(word, valid_word_actions):
              del result[word]
          return result
      return trie

    def _log_error(error: str):
      logging.error(error)
      logging.error('  action_sequence: %s', actions_sequence)
      logging.error('  full_action_sequence: %s', full_tree_action_history)
      for action in full_tree_action_history:
        logging.error('    action: %d = [%s]', action,
                      self.grammar.action_to_rule[str(action)])
      logging.error('  word_prefix: %s', word_prefix)
      logging.error('  word_action: %s', word_action)
      logging.error('  stack: %s', stack)
      logging.error('  original_legal_actions: %s', original_legal_actions)
      for action in original_legal_actions:
        logging.error('    action: %d = [%s]', action,
                      self.grammar.action_to_rule[str(action)])
      logging.error('  known_word_tries: %s', self.known_word_tries)
      logging.error('  current_trie: %s', current_trie)
      logging.error('  valid_word_actions: %s', self.valid_word_actions)
      logging.error('  current_valid_word_actions: %s',
                    current_valid_word_actions)
      logging.error('  crossed_search: %s', crossed_search_boundary)

    def _actions_since_last_query(taken_actions, simulated_actions):
      """Sequence of actions since last issued query."""
      if self.initial_stack and self.initial_stack[-1] == self.grammar.start():
        return simulated_actions
      actions = []
      for action in taken_actions[::-1]:
        actions.append(action)
        if self.grammar.productions()[action].lhs() == self.grammar.start():
          break
      return tuple(actions[::-1] + list(simulated_actions))

    current_node = f'{stack[-1].symbol()}'
    crossed_search_boundary = _compute_crossed_search_boundary()
    word_prefix = _find_word_prefix()
    word_action = _find_word_action()

    actions_since_last_query = _actions_since_last_query(
        self.actions, actions_sequence)
    if self.restriction_trie and self.restriction_trie.has_subtrie(
        actions_since_last_query):
      restricted_legal_actions = []
      for sequence in self.restriction_trie.iterkeys(actions_since_last_query):
        if len(sequence) > len(actions_since_last_query):
          restricted_legal_actions.append(
              sequence[len(actions_since_last_query)])
      if restricted_legal_actions:
        return tuple(set(restricted_legal_actions))

    # When we use a trie, especially when splitting the `Word` types into
    # _Wq_, _Wa_, _Wd_ and _Wt_, it's possible that the associated trie is
    # empty. We need to check that it's possible to generate a term from a
    # given trie.
    if (current_node == '_Q_' and not crossed_search_boundary and
        self.known_word_tries is not None):
      legal_actions = []
      for action in original_legal_actions:
        rule = self.grammar.action_to_rule[str(action)]
        if '_Wq_' in rule:
          if self.known_word_tries.question_trie:
            legal_actions.append(action)
        elif '_Wa_' in rule:
          if _get_restricted_trie(
              rule=rule, trie=self.known_word_tries.answer_trie):
            legal_actions.append(action)
        elif '_Wd_' in rule:
          if _get_restricted_trie(
              rule=rule, trie=self.known_word_tries.document_trie):
            legal_actions.append(action)
        elif '_Wt_' in rule:
          if _get_restricted_trie(
              rule=rule, trie=self.known_word_tries.title_trie):
            legal_actions.append(action)
        elif '_W_' in rule:
          if _get_restricted_trie(
              rule=rule, trie=self.known_word_tries.local_trie):
            legal_actions.append(action)
        else:
          legal_actions.append(action)
      return tuple(legal_actions)

    current_trie = None
    if self.known_word_tries is not None:
      if crossed_search_boundary:
        current_trie = self.known_word_tries.global_trie
      else:
        operator = _find_operator()
        if word_action.startswith('_Wq_'):
          current_trie = self.known_word_tries.question_trie
        elif word_action.startswith('_Wa_'):
          current_trie = _get_restricted_trie(
              rule=operator, trie=self.known_word_tries.answer_trie)
        elif word_action.startswith('_Wd_'):
          current_trie = _get_restricted_trie(
              rule=operator, trie=self.known_word_tries.document_trie)
        elif word_action.startswith('_Wt_'):
          current_trie = _get_restricted_trie(
              rule=operator, trie=self.known_word_tries.title_trie)
        else:
          current_trie = _get_restricted_trie(
              rule=operator, trie=self.known_word_tries.local_trie)

    current_valid_word_actions = None
    if self.valid_word_actions is not None:
      if word_action.startswith('_Wq_'):
        current_valid_word_actions = self.valid_word_actions.question_word_actions
      elif word_action.startswith('_Wa_'):
        current_valid_word_actions = self.valid_word_actions.answer_word_actions
      elif word_action.startswith('_Wd_'):
        current_valid_word_actions = self.valid_word_actions.document_word_actions
      elif word_action.startswith('_Wt_'):
        current_valid_word_actions = self.valid_word_actions.title_word_actions
      else:
        current_valid_word_actions = self.valid_word_actions.all_word_actions

    if (current_node in ['_Vw_', '_Vsh_'] and
        current_valid_word_actions is not None and not crossed_search_boundary):
      # For lexical rules, we apply restrictions if we were given the set of
      # available word action as per BERT state.
      # Additionally, if a known-trie has been passed, we will further restrict
      # to only those actions that yield valid words.

      # We have not yet crossed a "query" boundary, so we will restrict words to
      # be expanded to the word actions valid for the known BERT state.
      legal_actions = [
          action for action in legal_actions
          if action in current_valid_word_actions
      ]

    if current_trie is not None:
      if current_node in ['_W_', '_Wq_', '_Wa_', '_Wd_', '_Wt_']:
        # If we use a trie, we need to control the recursion of word
        # derivations. In particular, we need to check that
        #   1)  the _W_ -> _Vw_ _W-_ is only available if _any_ multi-word
        #       sequence could be generated;
        #   2)  the _W_ -> _Vw_ is only available if at least one initial word
        #       sequence can be generated.
        has_length_1_words = False
        has_length_2_words = False
        for word in current_trie.iterkeys(()):
          if has_length_1_words and has_length_2_words:
            break
          if len(word) == 1 and _can_construct(word,
                                               current_valid_word_actions):
            has_length_1_words = True
          elif not has_length_2_words and _can_construct(
              word, current_valid_word_actions):
            has_length_2_words = True

        valid_rules = []
        if has_length_1_words:
          valid_rules.append(
              self.grammar.rule_to_action[f'{current_node} -> _Vw_'])
        if has_length_2_words:
          valid_rules.append(
              self.grammar.rule_to_action[f'{current_node} -> _Vw_ _W-_'])
        legal_actions = [
            action for action in legal_actions if action in valid_rules
        ]
      elif current_node == '_W-_':
        # If we use a trie, we need to control the continuation of recursion.
        # In particular, we need to check that
        #   1) the _W-_ -> _Vsh_ is only available if at least one valid word
        #      can be generated by terminating and not recursing;
        #   2) the _W-_ -> _Vsh_ _W-_ is only available if at least one word
        #      can be generated if another recursion happens.
        has_immediate_word = False
        has_continuing_word = False
        try:
          for word in current_trie.iterkeys(word_prefix):
            if has_immediate_word and has_continuing_word:
              break
            if not has_immediate_word and len(
                word) == len(word_prefix) + 1 and _can_construct(
                    word[-1:], current_valid_word_actions):
              has_immediate_word = True
            elif not has_continuing_word and len(
                word) > len(word_prefix) + 1 and _can_construct(
                    word[len(word_prefix):], current_valid_word_actions):
              has_continuing_word = True
        except KeyError:
          _log_error('INVALID WORD PREFIX')

        valid_rules = []
        if has_immediate_word:
          valid_rules.append(self.grammar.rule_to_action['_W-_ -> _Vsh_'])
        if has_continuing_word:
          valid_rules.append(self.grammar.rule_to_action['_W-_ -> _Vsh_ _W-_'])
        legal_actions = [
            action for action in legal_actions if action in valid_rules
        ]
      elif current_node == '_Vw_':
        # Check if the last rule ends with `_W-_` to determine if we recursed
        # or not.
        rule = self.grammar.action_to_rule[str(full_tree_action_history[-1])]
        is_recursive = rule.endswith('_W-_')
        if crossed_search_boundary:
          legal_actions = (
              self.filter_legal_actions_for_vw_node_crossed_search_boundary(
                  legal_actions, is_recursive))
        else:
          legal_actions = self.filter_legal_actions_for_vw_node(
              legal_actions, is_recursive, current_trie)
      elif current_node == '_Vsh_':
        # Check if the last rule ends with `_W-_` to determine if we recursed
        # or not.
        rule = self.grammar.action_to_rule[str(full_tree_action_history[-1])]
        is_recursive = rule.endswith('_W-_')
        if crossed_search_boundary:
          legal_actions = (
              self.filter_legal_actions_for_vsh_node_crossed_search_boundary(
                  legal_actions, is_recursive, word_prefix))
        else:
          legal_actions = self.filter_legal_actions_for_vsh_node(
              legal_actions, is_recursive, word_prefix, current_trie)

    # These should not happen, but the Trie-based constraining surfaces the odd
    # error case every once in a while, so try to dump some info whenever this
    # happens.
    if not legal_actions:
      _log_error('INVALID TRANSITION')

    return tuple(legal_actions)


def to_lucene_query(base_query: str,
                    adjustments: Optional[List[QueryAdjustment]] = None,
                    escape_query: bool = True) -> str:
  """Builds a valid lucene query from a base query, query adjustments, and
  vocabulary mappings.

  Args:
    base_query: str, The base query which is prepended to the new query.
    adjustments: List[QueryAdjustment], List of all the adjustmenst
      (=reformulations) applied to the base query. The terms in the adjustments
      can be templates or words directly.
    escape_query:  If True, escapes `base_query`.  Needed for Lucene, not needed
      for running against WebSearch.

  Returns:
    A valid and escaped lucene query.
  """
  if not adjustments:
    adjustments = []

  escape_fn = {
      True: utils.escape_for_lucene,
      False: lambda x: x,
  }[escape_query]

  lucene_query = escape_fn(base_query)

  for adjustment in adjustments:
    term = adjustment.term
    # `term` needs to be properly escaped if it contains any special characters.
    term = utils.escape_for_lucene(term)

    field = adjustment.field.value.strip('[]')
    if adjustment.operator == Operator.MINUS:
      lucene_query += f' -({field}:"{term}")'
    elif adjustment.operator == Operator.PLUS:
      lucene_query += f' +({field}:"{term}")'
    elif adjustment.operator == Operator.APPEND:
      if adjustment.field == Field.ALL:
        lucene_query += f' {term}'
      else:
        lucene_query += f' ({field}:"{term}")'

  return lucene_query


def from_lucene_str(
    lucene_str: str
) -> Tuple[str, List[Tuple[QueryAdjustment, str, Tuple[int, int]]]]:
  """Returns an abstract string from a lucene string.

  Args:
    lucene_str: str, A valid lucene string.

  Returns:
    A tuple of string and adjustments.
  """
  adjustments = []

  re_patterns = {
      r' \(contents\:"(.*?)"\)': [Field.CONTENTS, Operator.APPEND],
      r' \(title\:"(.*?)"\)': [Field.TITLE, Operator.APPEND],
      r' \-\(contents\:"(.*?)"\)': [Field.CONTENTS, Operator.MINUS],
      r' \-\(title\:"(.*?)"\)': [Field.TITLE, Operator.MINUS],
      r' \+\(contents\:"(.*?)"\)': [Field.CONTENTS, Operator.PLUS],
      r' \+\(title\:"(.*?)"\)': [Field.TITLE, Operator.PLUS],
  }

  matches = [
      match for re_pattern in re_patterns.keys()
      for match in re.finditer(re_pattern, lucene_str)
  ]
  for match in matches:
    span = match.span()
    matched_expression = match.group()
    # unescape
    term = match.groups()[0].replace('\\', '').strip()  # pytype: disable=attribute-error
    adjustment = QueryAdjustment(
        field=re_patterns[match.re.pattern][0],
        operator=re_patterns[match.re.pattern][1],
        term=term)
    adjustments.append((adjustment, matched_expression, span))
  adjustments = sorted(adjustments, key=lambda x: x[-1])

  base_query = lucene_str
  for (adjustment, matched_expression, span) in adjustments:
    base_query = base_query.replace(matched_expression, '', 1)
  # unescape
  base_query = base_query.replace('\\', '').strip()

  final_query = base_query
  for (adjustment, _, _) in adjustments:
    final_query += (f' {adjustment.field.value} {adjustment.operator.value} '
                    f'{adjustment.term}')

  return final_query, adjustments

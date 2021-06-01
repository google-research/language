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
"""Tests for language.search_agents.muzero.state_tree.py."""

from typing import Collection, List, Sequence, Tuple

from absl.testing import parameterized
import nltk
import pygtrie
import tensorflow as tf

from language.search_agents.muzero import grammar_lib
from language.search_agents.muzero import state_tree

# pylint: disable=g-complex-comprehension
# pylint: disable=anomalous-backslash-in-string


def apply_productions(tree: state_tree.NQStateTree, productions: List[str]):

  def get_production_by_str(production_str):
    productions = {str(p): p for p in tree.grammar.productions()}
    return productions[production_str]

  for p in productions:
    tree.apply_production(get_production_by_str(p))


def make_trie(grammar: state_tree.NQCFG, words: Sequence[str]) -> pygtrie.Trie:
  word_keys = set()
  for word in words:
    word_keys.add(
        tuple(map(grammar.terminal_to_action.__getitem__, word.split())))
  return pygtrie.Trie.fromkeys(word_keys)


def make_valid_word_actions(
    word_actions: Collection[int]) -> state_tree.ValidWordActions:
  return state_tree.ValidWordActions(
      all_word_actions=word_actions,
      question_word_actions=word_actions,
      answer_word_actions=word_actions,
      document_word_actions=word_actions,
      title_word_actions=word_actions,
      diff_word_actions=set(),
      intersect_word_actions=set())


def rules_to_actions(grammar: state_tree.NQCFG, rules: List[str]) -> Tuple[int]:
  return tuple(grammar.rule_to_action[rule] for rule in rules)


class NQCFGTest(parameterized.TestCase, tf.test.TestCase):
  """Test for the NQCFG."""

  def setUp(self):
    super().setUp()
    # setup a grammar
    grammar = [
        "_Q_ -> '[pos]' '[title]' _Wa_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _Wd_ _Q_",
        "_Q_ -> '[stop]'",
        "_Wa_ -> 'car'",
        "_Wd_ -> 'dog'",
    ]
    self.grammar = state_tree.NQCFG(" \n ".join(grammar))
    self.grammar.set_start(self.grammar.productions()[0].lhs())

  @parameterized.named_parameters(
      dict(
          testcase_name="adjustment_to_action_sequence_valid",
          query_adjustment=state_tree.QueryAdjustment(
              operator=state_tree.Operator.PLUS,
              field=state_tree.Field.TITLE,
              term="car"),
          answer_words=["car"],
          title_words=[],
          question_words=[],
          document_words=["dog"],
          tokenize_fn=lambda x: x.split(),
          fallback_term_type="_Wa_",
          expected=(True, [0, 3])),
      dict(
          testcase_name="adjustment_to_action_sequence_fallback",
          query_adjustment=state_tree.QueryAdjustment(
              operator=state_tree.Operator.PLUS,
              field=state_tree.Field.TITLE,
              term="car"),
          answer_words=[],
          title_words=[],
          question_words=[],
          document_words=[],
          tokenize_fn=lambda x: x.split(),
          fallback_term_type="_Wa_",
          expected=(True, [0, 3])),
      dict(
          testcase_name="adjustment_to_action_sequence_invalid",
          query_adjustment=state_tree.QueryAdjustment(
              operator=state_tree.Operator.MINUS,
              field=state_tree.Field.TITLE,
              term="car"),
          answer_words=["car"],
          title_words=[],
          question_words=[],
          document_words=["dog"],
          tokenize_fn=lambda x: x.split(),
          fallback_term_type="_Wa_",
          expected=(False, [])))
  def test_query_adjustment_to_action_sequence(self, query_adjustment,
                                               answer_words, title_words,
                                               question_words, document_words,
                                               tokenize_fn, fallback_term_type,
                                               expected):
    self.assertEqual(
        self.grammar.query_adjustment_to_action_sequence(
            query_adjustment, answer_words, title_words, question_words,
            document_words, tokenize_fn, fallback_term_type), expected)


class NQStateTreeTest(tf.test.TestCase):
  """Test for the state tree."""

  def setUp(self):
    super().setUp()
    # setup a grammar
    grammar = [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[stop]'",
        "_W_ -> 'car'",
        "_W_ -> 'dog'",
    ]
    self.grammar = state_tree.NQCFG(" \n ".join(grammar))
    self.grammar.set_start(self.grammar.productions()[0].lhs())
    self.state_tree = state_tree.NQStateTree(self.grammar)

  def test_legal_actions(self):
    # Only _Q_-expansions to start with.
    legal_actions_target = {
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[stop]'",
    }
    self.assertEqual(
        legal_actions_target, {
            str(self.grammar.productions()[idx])
            for idx in self.state_tree.legal_actions()
        })
    self.assertEmpty(self.state_tree.root.leaves())

    # Expand the `_Q_`, so now you must expand the `_W_`.
    apply_productions(self.state_tree, [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
    ])
    legal_actions_target = {
        "_W_ -> 'car'",
        "_W_ -> 'dog'",
    }
    self.assertEqual(
        legal_actions_target, {
            str(self.grammar.productions()[idx])
            for idx in self.state_tree.legal_actions()
        })
    self.assertEqual(self.state_tree.root.leaves(),
                     ["[pos]", "[title]", "[neg]", "[contents]"])

    # Expand a _W_ node,, so you'll move to the next _W_.
    apply_productions(self.state_tree, [
        "_W_ -> 'car'",
    ])
    legal_actions_target = {
        "_W_ -> 'car'",
        "_W_ -> 'dog'",
    }
    self.assertEqual(
        legal_actions_target, {
            str(self.grammar.productions()[idx])
            for idx in self.state_tree.legal_actions()
        })
    self.assertEqual(self.state_tree.root.leaves(),
                     ["[pos]", "[title]", "car", "[neg]", "[contents]"])

    # Expand another _W_ node, so we're back to _Q_.
    apply_productions(self.state_tree, [
        "_W_ -> 'dog'",
    ])

    legal_actions_target = {
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[stop]'",
    }
    self.assertEqual(
        legal_actions_target, {
            str(self.grammar.productions()[idx])
            for idx in self.state_tree.legal_actions()
        })
    self.assertEqual(self.state_tree.root.leaves(),
                     ["[pos]", "[title]", "car", "[neg]", "[contents]", "dog"])

  def test_finished_query(self):
    apply_productions(self.state_tree, [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
    ])
    self.assertFalse(self.state_tree.finished_query())

    apply_productions(self.state_tree, [
        "_W_ -> 'dog'",
    ])
    self.assertFalse(self.state_tree.finished_query())
    self.assertFalse(self.state_tree.is_complete())

    apply_productions(self.state_tree, [
        "_W_ -> 'car'",
    ])
    self.assertTrue(self.state_tree.finished_query())
    # Query is completed, but we now recurse for a further adjustment.
    self.assertFalse(self.state_tree.is_complete())

  def test_is_complete(self):
    apply_productions(self.state_tree, [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_W_ -> 'dog'",
        "_W_ -> 'dog'",
    ])
    self.assertFalse(self.state_tree.is_complete())
    apply_productions(self.state_tree, [
        "_Q_ -> '[stop]'",
    ])
    self.assertTrue(self.state_tree.is_complete())

  def test_reset_to_previous_finished_query(self):
    grammar = [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[neg]' '[title]' _W_ '[pos]' '[contents]' _W_ _Q_",
        "_Q_ -> '[stop]'",
        "_W_ -> 'car'",
        "_W_ -> 'dog'",
    ]
    self.grammar = state_tree.NQCFG(" \n ".join(grammar))
    self.grammar.set_start(self.grammar.productions()[0].lhs())
    self.state_tree = state_tree.NQStateTree(self.grammar)

    apply_productions(self.state_tree, [
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_W_ -> 'dog'",
        "_W_ -> 'car'",
    ])
    self.assertTrue(self.state_tree.finished_query())

    adjustments = self.state_tree.to_adjustments()
    self.assertLen(adjustments, 2)
    self.assertEqual(adjustments[0].operator, state_tree.Operator.PLUS)
    self.assertEqual(adjustments[0].field, state_tree.Field.TITLE)
    self.assertEqual(adjustments[0].term, "dog")
    self.assertEqual(adjustments[1].operator, state_tree.Operator.MINUS)
    self.assertEqual(adjustments[1].field, state_tree.Field.CONTENTS)
    self.assertEqual(adjustments[1].term, "car")

    apply_productions(self.state_tree, [
        "_Q_ -> '[neg]' '[title]' _W_ '[pos]' '[contents]' _W_ _Q_",
        "_W_ -> 'car'",
        "_W_ -> 'dog'",
    ])
    self.assertTrue(self.state_tree.finished_query())

    adjustments = self.state_tree.to_adjustments()
    self.assertLen(adjustments, 4)
    self.assertEqual(adjustments[2].operator, state_tree.Operator.MINUS)
    self.assertEqual(adjustments[2].field, state_tree.Field.TITLE)
    self.assertEqual(adjustments[2].term, "car")
    self.assertEqual(adjustments[3].operator, state_tree.Operator.PLUS)
    self.assertEqual(adjustments[3].field, state_tree.Field.CONTENTS)
    self.assertEqual(adjustments[3].term, "dog")

    self.state_tree.reset_to_previous_finished_query()
    self.assertTrue(self.state_tree.finished_query())

    adjustments = self.state_tree.to_adjustments()
    self.assertLen(adjustments, 2)
    self.assertEqual(adjustments[0].operator, state_tree.Operator.PLUS)
    self.assertEqual(adjustments[0].field, state_tree.Field.TITLE)
    self.assertEqual(adjustments[0].term, "dog")
    self.assertEqual(adjustments[1].operator, state_tree.Operator.MINUS)
    self.assertEqual(adjustments[1].field, state_tree.Field.CONTENTS)
    self.assertEqual(adjustments[1].term, "car")

  def test_to_lucene_query(self):
    base_query = "What is 7 + 5 - 2"
    adjustments = [
        state_tree.QueryAdjustment(state_tree.Operator.MINUS,
                                   state_tree.Field.TITLE, "some-term to add")
    ]
    lucene_query = state_tree.to_lucene_query(base_query, adjustments)
    self.assertEqual(lucene_query,
                     'What is 7 \+ 5 \- 2 -(title:"some\-term to add")')

  def test_from_lucene_query(self):
    lucene_str = 'What is 7 \+ 5 \- 2 -(title:"some\-term to add")'
    query_str, adjustments = state_tree.from_lucene_str(lucene_str)
    self.assertEqual(query_str,
                     "What is 7 + 5 - 2 [title] [neg] some-term to add")
    self.assertEqual(adjustments, [(state_tree.QueryAdjustment(
        state_tree.Operator.MINUS, state_tree.Field.TITLE,
        "some-term to add"), R' -(title:"some\-term to add")', (19, 48))])

  def test_transition_model_from_empty(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##d", "##s", "bear"})

    tree = state_tree.NQStateTree(grammar=grammar)

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal] for terminal in {"car", "##s"}
        }),
        grammar=grammar)

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()), [
            grammar.production_to_action(p)
            for p in grammar.productions(lhs=nltk.Nonterminal("_Q_"))
        ])

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_"
                ])), [
                    grammar.production_to_action(p)
                    for p in grammar.productions(lhs=nltk.Nonterminal("_W_"))
                ])

    # Note, only `car` is in current bert state as per the initialization of
    # the transition model.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_"
                ])), [grammar.terminal_to_action["car"]])

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                ])), [grammar.terminal_to_action["car"]])

    # Again, the available word pieces are constrained to how the transition
    # model was initialized.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_",
                ])), [grammar.terminal_to_action["##s"]])

    # However, now that we crossed a `query boundary`, we lose the restrictions.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    # Crossing a query boundary.  This means we now would have
                    # a new set of results, and therefore, don't know how we
                    # should constrain the lexical rules.
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                ])),
        [grammar.terminal_to_action[term] for term in {"car", "bear"}])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    # Crossing a query boundary.  This means we now would have
                    # a new set of results, and therefore, don't know how we
                    # should constrain the lexical rules.
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                ])),
        [grammar.terminal_to_action[term] for term in {"##s", "##d"}])

    # Exercise the caching.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    # Crossing a query boundary.  This means we now would have
                    # a new set of results, and therefore, don't know how we
                    # should constrain the lexical rules.
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                ])),
        [grammar.terminal_to_action[term] for term in {"car", "bear"}])

  def test_transition_model_from_partial_tree(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##d", "##s", "bear"})

    # Make a tree that already has some state.
    tree = state_tree.NQStateTree(grammar=grammar)
    apply_productions(tree, [
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
        "_W_ -> _Vw_",
        "_Vw_ -> 'car'",
        "_W_ -> _Vw_ _W-_",
        "_Vw_ -> 'car'",
        "_W-_ -> _Vsh_",
        "_Vsh_ -> '##s'",
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
        "_W_ -> _Vw_ _W-_",
        "_Vw_ -> 'bear'",
    ])

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal] for terminal in {"car", "##s"}
        }),
        grammar=grammar)

    # Check that previous history is properly used.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()), [
            grammar.production_to_action(p)
            for p in grammar.productions(lhs=nltk.Nonterminal("_W-_"))
        ])

    # Constrain to what you initialized with.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(grammar=grammar, rules=["_W-_ -> _Vsh_"])),
        [grammar.terminal_to_action["##s"]])

    # Checking the query boundary crossing..
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    # Cross the boundary.
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_"
                ])),
        [grammar.terminal_to_action[term] for term in {"car", "bear"}])

  def test_transition_model_no_lexical_masking(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##d", "##s", "bear"})

    # Make a tree that already has some state.
    tree = state_tree.NQStateTree(grammar=grammar)
    apply_productions(tree, [
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
        "_W_ -> _Vw_",
        "_Vw_ -> 'car'",
        "_W_ -> _Vw_ _W-_",
        "_Vw_ -> 'car'",
        "_W-_ -> _Vsh_",
        "_Vsh_ -> '##s'",
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
        "_W_ -> _Vw_ _W-_",
        "_Vw_ -> 'bear'",
    ])

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=None,
        grammar=grammar)

    # Check that previous history is properly used.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()), [
            grammar.production_to_action(p)
            for p in grammar.productions(lhs=nltk.Nonterminal("_W-_"))
        ])

    # Constrain to what you initialized with.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(grammar=grammar, rules=["_W-_ -> _Vsh_"])),
        [grammar.terminal_to_action[term] for term in {"##s", "##d"}])

  def test_transition_model_with_trie(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##d", "##s", "bear", "##t"})

    local_word_pieces = {"bear", "##s", "##d", "##t", "car"}
    known_words_trie = make_trie(
        grammar=grammar,
        words={"bear", "bear ##s", "car ##d ##s", "car ##t ##s"})
    global_words_trie = make_trie(
        grammar=grammar,
        words={
            "car",
            "car ##d",
            "car ##s",
            "car ##t ##s",
            "bear ##s",
            "bear ##d",
            "bear ##d ##s",
        })

    tree = state_tree.NQStateTree(grammar=grammar)

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal]
            for terminal in local_word_pieces
        }),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=known_words_trie,
            global_trie=global_words_trie,
            question_trie=known_words_trie,
            answer_trie=known_words_trie,
            document_trie=known_words_trie,
            title_trie=known_words_trie),
        grammar=grammar)

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()), [
            grammar.production_to_action(p)
            for p in grammar.productions(lhs=nltk.Nonterminal("_Q_"))
        ])

    # Given `known_words_trie`, we could recurse or not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_"
                ])), [
                    grammar.rule_to_action["_W_ -> _Vw_ _W-_"],
                    grammar.rule_to_action["_W_ -> _Vw_"]
                ])

    # Only `bear` is valid.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_"
                ])),
        [grammar.terminal_to_action["bear"], grammar.terminal_to_action["car"]])

    # Given `known_words_trie`, recursing more on prefix `bear` will not yield
    # valid words, so only the non-recursive version of _W-_ is available.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])

    # Only `##s` can complete `bear`, given `known_words_trie`.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                ])), [grammar.terminal_to_action["##s"]])

    # Given `known_words_trie`, we need to recurse  more on prefix `car` to
    # yield the valid word `car ##{d|t} ##s`;  without recursion, in this case,
    # no valid word could be dervied.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_ _W-_"]])

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                ])),
        [grammar.terminal_to_action["##d"], grammar.terminal_to_action["##t"]])

    # Now we must not recurse further, as we will not be able to extend into
    # an even longer valid word.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])

    # Crossing the query boundary - this relaxes assumptions to the global trie.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                ])),
        rules_to_actions(
            grammar=grammar, rules=["_W_ -> _Vw_", "_W_ -> _Vw_ _W-_"]))
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vw_node_crossed_search_boundary.cache_info(
        ).currsize, 0)
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vsh_node_crossed_search_boundary.cache_info(
        ).currsize, 0)

    # Only `car` can be generated without recursing.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[stop]'",  # Exercise the stack reset.
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                ])),
        [grammar.terminal_to_action["car"]])
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vw_node_crossed_search_boundary.cache_info(
        ).currsize, 1)
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vsh_node_crossed_search_boundary.cache_info(
        ).currsize, 0)

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                ])),
        [grammar.terminal_to_action["car"], grammar.terminal_to_action["bear"]])
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vw_node_crossed_search_boundary.cache_info(
        ).currsize, 2)
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vsh_node_crossed_search_boundary.cache_info(
        ).currsize, 0)

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                    "_W_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_W_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                ])), [grammar.terminal_to_action["##t"]])
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vw_node_crossed_search_boundary.cache_info(
        ).currsize, 2)
    self.assertEqual(
        transition_model
        .filter_legal_actions_for_vsh_node_crossed_search_boundary.cache_info(
        ).currsize, 1)

  def test_transition_model_with_trie_from_non_empty_tree(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##d", "##s", "bear", "##t"})

    local_word_pieces = {"bear", "##s", "##d", "##t", "car"}
    known_words_trie = make_trie(
        grammar=grammar, words={"bear", "bear ##s", "car ##t ##s"})
    global_words_trie = make_trie(
        grammar=grammar,
        words={
            "car",
            "car ##d",
            "car ##s",
            "car ##t ##s",
            "bear ##s",
            "bear ##d",
            "bear ##d ##s",
        })

    tree = state_tree.NQStateTree(grammar=grammar)
    apply_productions(
        tree,
        [
            "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
            "_W_ -> _Vw_ _W-_",  # Expands the first _W_
            "_Vw_ -> 'car'",
            "_W-_ -> _Vsh_ _W-_"
        ])

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal]
            for terminal in local_word_pieces
        }),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=known_words_trie,
            global_trie=global_words_trie,
            question_trie=known_words_trie,
            answer_trie=known_words_trie,
            document_trie=known_words_trie,
            title_trie=known_words_trie),
        grammar=grammar)

    # Note that because we already decided to generate at least two more
    # word-pieces in the pre-constructed tree, we only can generate a `##t`,
    # because only this would yield a valid word (car ##t ##s).
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()),
        [grammar.terminal_to_action["##t"]])

    # Checks the expansion of the second _W_ and crossing the search boundary.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Vsh_ -> '##t'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_W_ -> _Vw_",  # Expands the second _W_
                    "_Vw_ -> 'car'",
                    "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
                ])),
        [
            grammar.rule_to_action["_W_ -> _Vw_ _W-_"],
            grammar.rule_to_action["_W_ -> _Vw_"]
        ])

  def test_transition_model_with_trie_and_word_types(self):
    local_word_pieces = {
        "bear", "##s", "##d", "##t", "car", "cycle", "tri", "##cycle", "bi",
        "bar", "##k", "jump", "##er", "for", "##est"
    }
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=True),
        vocab=local_word_pieces)

    known_words_trie = make_trie(
        grammar=grammar,
        words={"bear", "bear ##s", "car ##d ##s", "car ##t ##s"})
    global_words_trie = make_trie(
        grammar=grammar,
        words={
            "car",
            "car ##d",
            "car ##s",
            "car ##t ##s",
            "bear ##s",
            "bear ##d",
            "bear ##d ##s",
        })
    question_trie = make_trie(
        grammar=grammar,
        words={"cycle", "tri ##cycle", "bi ##cycle", "bi ##cycle ##s"})
    answer_trie = make_trie(
        grammar=grammar, words={"bar", "bar ##s", "bar ##k", "for ##est ##s"})
    document_trie = make_trie(
        grammar=grammar,
        words={
            "jump",
            "jump ##s",
            "jump ##er",
            "bear ##d ##s",
        })

    tree = state_tree.NQStateTree(grammar=grammar)

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal]
            for terminal in local_word_pieces
        }),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=known_words_trie,
            global_trie=global_words_trie,
            question_trie=question_trie,
            answer_trie=answer_trie,
            document_trie=document_trie,
            title_trie=known_words_trie),
        grammar=grammar)

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(()), [
            grammar.production_to_action(p)
            for p in grammar.productions(lhs=nltk.Nonterminal("_Q_"))
        ])

    # Given `known_words_trie`, we could recurse or not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_"
                ])), [
                    grammar.rule_to_action["_Wq_ -> _Vw_ _W-_"],
                    grammar.rule_to_action["_Wq_ -> _Vw_"]
                ])

    # Recurse, yield terminals from the correct trie.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_ _W-_"
                ])),
        [grammar.terminal_to_action["tri"], grammar.terminal_to_action["bi"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_"
                ])),
        [grammar.terminal_to_action["bar"], grammar.terminal_to_action["for"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_"
                ])), [
                    grammar.terminal_to_action["jump"],
                    grammar.terminal_to_action["bear"]
                ])

    # Given `known_words_trie`, recursing more on prefix `tri` will not yield
    # valid words, so only the non-recursive version of _W-_ is available.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_ _W-_",
                    "_Vw_ -> 'tri'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])

    # Only `##cycle` can complete `tri`, given `known_words_trie`.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_ _W-_",
                    "_Vw_ -> 'tri'",
                    "_W-_ -> _Vsh_",
                ])), [grammar.terminal_to_action["##cycle"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                    "_W-_ -> _Vsh_",
                ])),
        [grammar.terminal_to_action["##s"], grammar.terminal_to_action["##k"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                ])),
        [grammar.terminal_to_action["##s"], grammar.terminal_to_action["##er"]])

    # Given `known_words_trie`, we need to recurse more on prefix `bi` to
    # yield the valid word `bi ##cycle ##s`; without recursion, in this case,
    # no valid word could be derived.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_ _W-_",
                    "_Vw_ -> 'tri'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##cycle'",
                    "_Wq_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bi'",
                ])),
        [
            grammar.rule_to_action["_W-_ -> _Vsh_"],  # `bi ##cycle`
            grammar.rule_to_action["_W-_ -> _Vsh_ _W-_"]  # bi ##cycle ##s`
        ])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'for'",
                ])),
        [
            grammar.rule_to_action["_W-_ -> _Vsh_ _W-_"]  # `for ##est ##s`
        ])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                ])),
        [
            grammar.rule_to_action["_W-_ -> _Vsh_ _W-_"]  # `bear ##d ##s`
        ])

    # Generate the intermediate wordpieces.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'for'",
                    "_W-_ -> _Vsh_ _W-_",
                ])), [
                    grammar.terminal_to_action["##est"],
                ])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_ _W-_",
                ])), [grammar.terminal_to_action["##d"]])

    # Now we must not recurse further, as we will not be able to extend into
    # an even longer valid word.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'for'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##est'",
                ])), [
                    grammar.rule_to_action["_W-_ -> _Vsh_"],
                ])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])

    # Crossing the query boundary - this relaxes assumptions to the global
    # trie.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bar'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wa_ -> _Vw_ _W-_",
                    "_Vw_ -> 'for'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##est'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _Wa_ '[neg]' '[title]' _Wa_ _Q_",
                ])),
        rules_to_actions(
            grammar=grammar, rules=["_Wa_ -> _Vw_", "_Wa_ -> _Vw_ _W-_"]))
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                ])),
        rules_to_actions(
            grammar=grammar, rules=["_Wq_ -> _Vw_", "_Wq_ -> _Vw_ _W-_"]))

    # Only `car` can be generated without recursing.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[stop]'",  # Exercise the stack reset.
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_",
                ])),
        [grammar.terminal_to_action["car"]])

    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wd_ '[neg]' '[title]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'jump'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'bear'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                    "_W-_ -> _Vsh_",
                    "_Vsh_ -> '##s'",
                    "_Q_ -> '[pos]' '[contents]' _Wq_ '[neg]' '[title]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_",
                    "_Vw_ -> 'car'",
                    "_Wq_ -> _Vw_ _W-_",
                ])),
        [grammar.terminal_to_action["car"], grammar.terminal_to_action["bear"]])

  def test_transition_model_with_valid_word_actions(self):
    all_word_pieces = {
        "bear", "##s", "##d", "##t", "car", "cycle", "tri", "##cycle", "bi",
        "bar", "##k", "jump", "##er", "for", "##est"
    }
    question_word_pieces = {"bear", "car"}
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ONE_TERM_AT_A_TIME,
            split_vocabulary_by_type=True),
        vocab=all_word_pieces)
    all_trie = make_trie(
        grammar=grammar, words={"bear ##s", "car ##d ##s", "car"})
    question_trie = make_trie(grammar=grammar, words={"bear", "car"})
    all_word_actions = {
        grammar.terminal_to_action[terminal] for terminal in all_word_pieces
    }
    question_word_actions = {
        grammar.terminal_to_action[terminal]
        for terminal in question_word_pieces
    }

    tree = state_tree.NQStateTree(grammar=grammar)

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=state_tree.ValidWordActions(
            all_word_actions=all_word_actions,
            question_word_actions=question_word_actions,
            answer_word_actions=all_word_actions,
            document_word_actions=all_word_actions,
            title_word_actions=all_word_actions,
            diff_word_actions=set(),
            intersect_word_actions=set()),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=all_trie,
            global_trie=all_trie,
            question_trie=question_trie,
            answer_trie=all_trie,
            document_trie=all_trie,
            title_trie=all_trie),
        grammar=grammar)

    # Given `valid_word_actions` for the word type, we could not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=["_Q_ -> '[pos]' '[contents]' _Wq_ _Q_"])),
        [grammar.rule_to_action["_Wq_ -> _Vw_"]])

    # Given `valid_word_actions` for the word type, we could recurse or not
    # recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=["_Q_ -> '[pos]' '[contents]' _Wd_ _Q_"])), [
                    grammar.rule_to_action["_Wd_ -> _Vw_ _W-_"],
                    grammar.rule_to_action["_Wd_ -> _Vw_"]
                ])

    # Cross the search boundary and choose a different word source.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ _Q_", "_Wq_ -> _Vw_",
                    "_Vw_ -> 'bear'", "_Q_ -> '[pos]' '[contents]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_", "_Vw_ -> 'car'", "_W-_ -> _Vsh_ _W-_"
                ])), [grammar.terminal_to_action["##d"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ _Q_",
                    "_Wq_ -> _Vw_",
                    "_Vw_ -> 'bear'",
                    "_Q_ -> '[pos]' '[contents]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_",
                    "_Vw_ -> 'car'",
                    "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'",
                ])), [grammar.rule_to_action["_W-_ -> _Vsh_"]])
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=[
                    "_Q_ -> '[pos]' '[contents]' _Wq_ _Q_", "_Wq_ -> _Vw_",
                    "_Vw_ -> 'bear'", "_Q_ -> '[pos]' '[contents]' _Wd_ _Q_",
                    "_Wd_ -> _Vw_ _W-_", "_Vw_ -> 'car'", "_W-_ -> _Vsh_ _W-_",
                    "_Vsh_ -> '##d'", "_W-_ -> _Vsh_"
                ])), [grammar.terminal_to_action["##s"]])

  def test_transition_model_start_actions_with_empty_trie(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ADD_TERM_ONLY,
            split_vocabulary_by_type=True),
        vocab={"car", "##d", "##s", "bear", "##t"})

    local_word_pieces = {"bear", "##s", "##d", "##t", "car"}
    known_words_trie = make_trie(
        grammar=grammar, words={"bear", "bear ##s", "car ##t ##s"})
    global_words_trie = make_trie(
        grammar=grammar,
        words={
            "car",
            "car ##d",
            "car ##s",
            "car ##t ##s",
            "bear ##s",
            "bear ##d",
            "bear ##d ##s",
        })

    tree = state_tree.NQStateTree(grammar=grammar)
    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal]
            for terminal in local_word_pieces
        }),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=known_words_trie,
            global_trie=global_words_trie,
            question_trie=known_words_trie,
            answer_trie=known_words_trie,
            document_trie=known_words_trie,
            title_trie=known_words_trie),
        grammar=grammar)

    # Checks the start of available actions.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(grammar=grammar, rules=[])), [
                grammar.rule_to_action["_Q_ -> '[stop]'"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wq_ _Q_"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wa_ _Q_"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wd_ _Q_"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wt_ _Q_"]
            ])

    # If the answer_trie is empty, the _Wa_ is removed from legal actions.
    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=make_valid_word_actions({
            grammar.terminal_to_action[terminal]
            for terminal in local_word_pieces
        }),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=known_words_trie,
            global_trie=global_words_trie,
            question_trie=known_words_trie,
            answer_trie=make_trie(grammar=grammar, words={}),
            document_trie=known_words_trie,
            title_trie=known_words_trie),
        grammar=grammar)
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(grammar=grammar, rules=[])), [
                grammar.rule_to_action["_Q_ -> '[stop]'"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wq_ _Q_"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wd_ _Q_"],
                grammar.rule_to_action["_Q_ -> '[or]' _Wt_ _Q_"]
            ])

  def test_transition_model_with_diff_word_actions(self):
    all_word_pieces = {
        "bear", "##s", "##d", "##t", "car", "cycle", "tri", "##cycle", "bi",
        "bar", "##k", "jump", "##er", "for", "##est"
    }
    intersect_word_pieces = {"bear", "car"}

    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ONE_TERM_AT_A_TIME,
            split_vocabulary_by_type=True),
        vocab=all_word_pieces)
    all_trie = make_trie(
        grammar=grammar, words={"bear ##s", "car ##d ##s", "car"})

    all_word_actions = {
        grammar.terminal_to_action[terminal] for terminal in all_word_pieces
    }
    intersect_word_actions = {
        grammar.terminal_to_action[terminal]
        for terminal in intersect_word_pieces
    }

    tree = state_tree.NQStateTree(grammar=grammar)

    transition_model = state_tree.NQTransitionModel(
        full_action_space_size=len(grammar.productions()),
        actions=tree.actions,
        valid_word_actions=state_tree.ValidWordActions(
            all_word_actions=all_word_actions,
            question_word_actions=all_word_actions,
            answer_word_actions=all_word_actions,
            document_word_actions=all_word_actions,
            title_word_actions=all_word_actions,
            diff_word_actions=set(),
            intersect_word_actions=intersect_word_actions),
        known_word_tries=state_tree.KnownWordTries(
            local_trie=all_trie,
            global_trie=all_trie,
            question_trie=all_trie,
            answer_trie=all_trie,
            document_trie=all_trie,
            title_trie=all_trie),
        grammar=grammar)

    # The word type _Wq_ is not restricted to intersect_word_actions,
    # we could both recurse or not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=["_Q_ -> '[pos]' '[contents]' _Wq_ _Q_"])), [
                    grammar.rule_to_action["_Wq_ -> _Vw_ _W-_"],
                    grammar.rule_to_action["_Wq_ -> _Vw_"]
                ])

    # The word type _Wd_ with [pos] operator is restricted to
    # intersect_word_actions, we can only not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=["_Q_ -> '[pos]' '[contents]' _Wd_ _Q_"])),
        [grammar.rule_to_action["_Wd_ -> _Vw_"]])

    # The word type _Wd_ with [neg] operator is not restricted to
    # intersect_word_actions, we could both recurse or not recurse.
    self.assertSameElements(
        transition_model.legal_actions_after_sequence(
            rules_to_actions(
                grammar=grammar,
                rules=["_Q_ -> '[neg]' '[contents]' _Wd_ _Q_"])), [
                    grammar.rule_to_action["_Wd_ -> _Vw_ _W-_"],
                    grammar.rule_to_action["_Wd_ -> _Vw_"]
                ], f"{grammar.action_to_rule}")


if __name__ == "__main__":
  tf.test.main()

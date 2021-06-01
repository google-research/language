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
"""Tests for language.search_agents.muzero.grammar_lib.py."""

from language.search_agents.muzero import grammar_lib
import tensorflow as tf


class GrammarLibTest(tf.test.TestCase):

  def test_construct_grammar_bert(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.BERT,
            split_vocabulary_by_type=False),
        vocab={"car", "##ry", "in", "##ner"})
    grammar = [str(production) for production in grammar.productions()]

    grammar_target = [
        # Word recursion.
        "_W_ -> _Vw_",
        "_W_ -> _Vw_ _W-_",
        "_W-_ -> _Vsh_",
        "_W-_ -> _Vsh_ _W-_",
        # Terminal rules.
        "_Vw_ -> 'car'",
        "_Vw_ -> 'in'",
        "_Vsh_ -> '##ry'",
        "_Vsh_ -> '##ner'",
        # Q-rules.
        "_Q_ -> '[stop]'",  # No more adjustments.
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[pos]' '[contents]' _W_ '[neg]' '[title]' _W_ _Q_",
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[pos]' '[title]' _W_ '[neg]' '[title]' _W_ _Q_",
    ]
    self.assertEqual(set(grammar), set(grammar_target))

  def test_construct_grammar_one_term_at_a_time(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ONE_TERM_AT_A_TIME,
            split_vocabulary_by_type=False),
        vocab={"car", "##ry", "in", "##ner"})
    grammar = [str(production) for production in grammar.productions()]

    grammar_target = [
        # Word recursion.
        "_W_ -> _Vw_",
        "_W_ -> _Vw_ _W-_",
        "_W-_ -> _Vsh_",
        "_W-_ -> _Vsh_ _W-_",
        # Terminal rules.
        "_Vw_ -> 'car'",
        "_Vw_ -> 'in'",
        "_Vsh_ -> '##ry'",
        "_Vsh_ -> '##ner'",
        # Q-rules.
        "_Q_ -> '[stop]'",  # No more adjustments.
        "_Q_ -> '[pos]' '[contents]' _W_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[pos]' '[title]' _W_ _Q_",
        "_Q_ -> '[neg]' '[title]' _W_ _Q_",
    ]
    self.assertEqual(set(grammar), set(grammar_target))

  def test_construct_grammar_add_term_only(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ADD_TERM_ONLY,
            split_vocabulary_by_type=False),
        vocab={"car", "##ry", "in", "##ner"})
    grammar = [str(production) for production in grammar.productions()]

    grammar_target = [
        # Word recursion.
        "_W_ -> _Vw_",
        "_W_ -> _Vw_ _W-_",
        "_W-_ -> _Vsh_",
        "_W-_ -> _Vsh_ _W-_",
        # Terminal rules.
        "_Vw_ -> 'car'",
        "_Vw_ -> 'in'",
        "_Vsh_ -> '##ry'",
        "_Vsh_ -> '##ner'",
        # Q-rules.
        "_Q_ -> '[stop]'",  # No more adjustments.
        "_Q_ -> '[or]' _W_ _Q_",
    ]
    self.assertEqual(set(grammar), set(grammar_target))

  def test_construct_grammar_one_term_at_a_time_with_add_term_only(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType
            .ONE_TERM_AT_A_TIME_WITH_ADD_TERM_ONLY,
            split_vocabulary_by_type=False),
        vocab={"car", "##ry", "in", "##ner"})
    grammar = [str(production) for production in grammar.productions()]

    grammar_target = [
        # Word recursion.
        "_W_ -> _Vw_",
        "_W_ -> _Vw_ _W-_",
        "_W-_ -> _Vsh_",
        "_W-_ -> _Vsh_ _W-_",
        # Terminal rules.
        "_Vw_ -> 'car'",
        "_Vw_ -> 'in'",
        "_Vsh_ -> '##ry'",
        "_Vsh_ -> '##ner'",
        # Q-rules.
        "_Q_ -> '[stop]'",  # No more adjustments.
        "_Q_ -> '[pos]' '[contents]' _W_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _W_ _Q_",
        "_Q_ -> '[pos]' '[title]' _W_ _Q_",
        "_Q_ -> '[neg]' '[title]' _W_ _Q_",
        "_Q_ -> '[or]' _W_ _Q_",
    ]
    self.assertEqual(set(grammar), set(grammar_target))

  def test_construct_grammar_with_split_vocabulary_by_type(self):
    grammar = grammar_lib.construct_grammar(
        grammar_config=grammar_lib.GrammarConfig(
            grammar_type=grammar_lib.GrammarType.ONE_TERM_AT_A_TIME,
            split_vocabulary_by_type=True),
        vocab={"car", "##ry", "in", "##ner"})
    grammar = [str(production) for production in grammar.productions()]

    grammar_target = [
        # Word recursion.
        "_Wq_ -> _Vw_",
        "_Wq_ -> _Vw_ _W-_",
        "_Wa_ -> _Vw_",
        "_Wa_ -> _Vw_ _W-_",
        "_Wd_ -> _Vw_",
        "_Wd_ -> _Vw_ _W-_",
        "_Wt_ -> _Vw_",
        "_Wt_ -> _Vw_ _W-_",
        "_W-_ -> _Vsh_",
        "_W-_ -> _Vsh_ _W-_",
        # Terminal rules.
        "_Vw_ -> 'car'",
        "_Vw_ -> 'in'",
        "_Vsh_ -> '##ry'",
        "_Vsh_ -> '##ner'",
        # Q-rules.
        "_Q_ -> '[stop]'",  # No more adjustments.
        "_Q_ -> '[pos]' '[contents]' _Wq_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _Wq_ _Q_",
        "_Q_ -> '[pos]' '[title]' _Wq_ _Q_",
        "_Q_ -> '[neg]' '[title]' _Wq_ _Q_",
        "_Q_ -> '[pos]' '[contents]' _Wa_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _Wa_ _Q_",
        "_Q_ -> '[pos]' '[title]' _Wa_ _Q_",
        "_Q_ -> '[neg]' '[title]' _Wa_ _Q_",
        "_Q_ -> '[pos]' '[contents]' _Wd_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _Wd_ _Q_",
        "_Q_ -> '[pos]' '[title]' _Wd_ _Q_",
        "_Q_ -> '[neg]' '[title]' _Wd_ _Q_",
        "_Q_ -> '[pos]' '[contents]' _Wt_ _Q_",
        "_Q_ -> '[neg]' '[contents]' _Wt_ _Q_",
        "_Q_ -> '[pos]' '[title]' _Wt_ _Q_",
        "_Q_ -> '[neg]' '[title]' _Wt_ _Q_",
    ]
    self.assertEqual(set(grammar), set(grammar_target))


if __name__ == "__main__":
  tf.test.main()

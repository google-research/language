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
"""Tests for language.search_agents.muzero.bert_state_lib.py."""

import collections
from typing import List

from absl.testing import flagsaver
from language.search_agents import environment_pb2
from language.search_agents.muzero import bert_state_lib
from language.search_agents.muzero import state_tree
from language.search_agents.muzero import utils as nqutils
import tensorflow as tf

# pylint: disable=g-complex-comprehension

flags_default = dict(
    bert_sequence_length=128,
    num_documents_to_retrieve=5,
    context_window_size=20,
)


def get_updated_default_flags(**kwargs):
  new_flags = dict(**kwargs)
  for k, v in flags_default.items():
    if k not in new_flags:
      new_flags[k] = v
  return new_flags


def apply_productions(tree: state_tree.NQStateTree, productions: List[str]):

  def get_production_by_str(production_str):
    productions = {str(p): p for p in tree.grammar.productions()}
    return productions[production_str]

  for p in productions:
    tree.apply_production(get_production_by_str(p))


class BertStateLibTest(tf.test.TestCase):
  """Tests for the NQ environment."""

  @flagsaver.flagsaver(**flags_default)
  def setUp(self):
    super().setUp()

    self.tokenizer = bert_state_lib.get_tokenizer()
    self.idf_lookup = collections.defaultdict(float, [
        ('cover', 5.25),
        ('ufc', 8.5),
        ('2', 3.5),
    ])

  def test_original_query_state_template(self):
    self.assertEqual(
        bert_state_lib.original_query_state_part(
            query='who is on the cover of ufc 2',
            tokenize_fn=self.tokenizer.tokenize,
            idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[CLS] who is on the cover of ufc 2 [SEP]'.split()),
            type_values={
                'state_part': ['[CLS]'] + ['original_query'] * 8 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * (8 + 2),
                'idf_score': [
                    0.0,  # [CLS]
                    0.0,  # who
                    0.0,  # is
                    0.0,  # on
                    0.0,  # the
                    5.25,  # cover
                    0.0,  # of
                    8.5,  # UFC
                    3.5,  # 2
                    0.0  # [SEP]
                ]
            }))

  def test_state_tree_state_part(self):
    tree = state_tree.NQStateTree(
        grammar=state_tree.NQCFG(' \n '.join([
            "_Q_ -> '[neg]' '[title]' _W_ '[pos]' '[contents]' _W_",
            '_W_ -> _Vw_',
            '_W_ -> _Vw_ _Vsh_',
            "_Vw_ -> 'cov'",
            "_Vw_ -> 'out'",
            "_Vsh_ -> '##er'",
        ])))
    tree.grammar.set_start(tree.grammar.productions()[0].lhs())

    apply_productions(tree,
                      ["_Q_ -> '[neg]' '[title]' _W_ '[pos]' '[contents]' _W_"])

    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] [pos] [contents] [SEP]'.split()),
            type_values={
                'state_part': ['tree_leaves'] * 4 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 5,
                'idf_score': [0.0] * 5,
            }))

    # Does not change the leaves.
    apply_productions(tree, ['_W_ -> _Vw_'])
    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] [pos] [contents] [SEP]'.split()),
            type_values={
                'state_part': ['tree_leaves'] * 4 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 5,
                'idf_score': [0.0] * 5,
            }))

    apply_productions(tree, ["_Vw_ -> 'out'"])
    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] out [pos] [contents] [SEP]'.split()),
            type_values={
                'state_part': ['tree_leaves'] * 5 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 6,
                'idf_score': [0.0] * 6,
            }))

    # Does not change the leaves.
    apply_productions(tree, ['_W_ -> _Vw_ _Vsh_'])
    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] out [pos] [contents] [SEP]'.split()),
            type_values={
                'state_part': ['tree_leaves'] * 5 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 6,
                'idf_score': [0.0] * 6,
            }))

    apply_productions(tree, ["_Vw_ -> 'cov'"])
    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] out [pos] [contents] cov [SEP]'.split()),
            type_values={
                'state_part': ['tree_leaves'] * 6 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 7,
                'idf_score': [0.0] * 7,
            }))

    apply_productions(tree, ["_Vsh_ -> '##er'"])
    self.assertEqual(
        bert_state_lib.state_tree_state_part(
            tree=tree, idf_lookup=self.idf_lookup),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='[neg] [title] out [pos] [contents] cov ##er [SEP]'
                .split()),
            type_values={
                'state_part': ['tree_leaves'] * 7 + ['[SEP]'],
            },
            float_values={
                'mr_score': [0.0] * 8,
                #                         cov ##er
                'idf_score': [0.0] * 5 + [5.25, 5.25, 0],
            }))

  def test_history_state_template(self):
    new_flags = get_updated_default_flags(
        context_window_size=3, context_title_size=1)
    with flagsaver.flagsaver(**new_flags):
      history_state_part = bert_state_lib.history_state_part(
          documents=[
              environment_pb2.Document(
                  content='this is a high term',
                  answer=environment_pb2.Answer(answer='high', mr_score=42.0),
                  title='high title'),
              environment_pb2.Document(
                  content='this is a low term instead',
                  answer=environment_pb2.Answer(answer='low', mr_score=1.0),
                  title='low title'),
          ],
          tokenize_fn=self.tokenizer.tokenize,
          idf_lookup=collections.defaultdict(float, (
              ('high', 10.0),
              ('low', 5.0),
              ('term', 7.5),
              ('title', 8.0),
          )),
          context_size=3,
          max_length=128,
          max_title_length=1)
    self.assertEqual(history_state_part, [
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='high [SEP] a high term [SEP] high [SEP]'.split()),
            type_values={
                'state_part': ['history_answer', '[SEP]'] +
                              ['history_context'] * 3 + ['[SEP]'] +
                              ['history_title'] + ['[SEP]']
            },
            float_values={
                'mr_score': [42.0] * 8,
                'idf_score': [10.0, 0.0, 0.0, 10.0, 7.5, 0.0, 10.0, 0.0],
            }),
        nqutils.ObsFragment(
            text=nqutils.Text(
                tokens='low [SEP] a low term [SEP] low [SEP]'.split()),
            type_values={
                'state_part': ['history_answer', '[SEP]'] +
                              ['history_context'] * 3 + ['[SEP]'] +
                              ['history_title'] + ['[SEP]']
            },
            float_values={
                'mr_score': [1.0] * 8,
                'idf_score': [5.0, 0.0, 0.0, 5.0, 7.5, 0.0, 5.0, 0.0],
            }),
    ])


if __name__ == '__main__':
  tf.test.main()

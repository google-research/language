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
"""Tests for language.search_agents.muzero.env.py."""

import collections

from absl import flags
from absl.testing import flagsaver
from language.search_agents import environment_pb2
from language.search_agents.muzero import env
from language.search_agents.muzero import types
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


def create_document(content: str, mr_score: float, answer: str = None):
  doc = environment_pb2.Document()
  doc.content = content
  doc.answer.answer = content if answer is None else answer
  doc.answer.mr_score = mr_score
  return doc


def create_env(max_history_entries: int):
  new_flags = get_updated_default_flags(
  )
  with flagsaver.flagsaver(**new_flags):
    nqenv = env.NQEnv(
        nq_server=None,
        state=types.EnvState(original_query=environment_pb2.GetQueryResponse()),
        training=False,
        stop_after_seeing_new_results=True)
    history_entries = [
        types.HistoryEntry(
            query='query',
            original_query='original_query',
            documents=[
                create_document('1', 1),
                create_document('2', 2),
                create_document('3', 3),
            ]),
        types.HistoryEntry(
            query='query',
            original_query='original_query',
            documents=[
                create_document('3', 3),
                create_document('4', 4),
                create_document('0', 0),
            ]),
        types.HistoryEntry(
            query='query',
            original_query='original_query',
            documents=[
                create_document('5', 5),
                create_document('3', 3),
                create_document('-1', -1),
            ])
    ]
    nqenv.state.history = history_entries[:max_history_entries]
    return nqenv


FLAGS = flags.FLAGS

Data = collections.namedtuple('Data', ['answer', 'context', 'score'])


class NQEnvTest(tf.test.TestCase):
  """Tests for the NQ environment."""

  @flagsaver.flagsaver(**flags_default)
  def setUp(self):
    super().setUp()

  def test_get_aggregated_documents(self):
    nqenv = create_env(max_history_entries=3)

    new_flags = get_updated_default_flags(num_documents_to_retrieve=3,)
    with flagsaver.flagsaver(**new_flags):
      aggregated_documents = env.get_aggregated_documents(nqenv.state.history)
      self.assertLen(aggregated_documents, 3)
      self.assertEqual(aggregated_documents[0].content, '5')
      self.assertEqual(aggregated_documents[1].content, '4')
      self.assertEqual(aggregated_documents[2].content, '3')

  def test_get_final_documents(self):
    new_flags = get_updated_default_flags(
        num_documents_to_retrieve=3, use_aggregated_documents=0)
    with flagsaver.flagsaver(**new_flags):
      # stop_after_seeing_new_results with 1 step in the history
      nqenv = create_env(max_history_entries=1)
      nqenv.stop_after_seeing_new_results = True

      final_documents = nqenv.get_final_document_list()
      self.assertLen(final_documents, 3)
      self.assertEqual(final_documents[0].content, '1')
      self.assertEqual(final_documents[1].content, '2')
      self.assertEqual(final_documents[2].content, '3')

      # stop_after_seeing_new_results with multiple steps in the history
      nqenv = create_env(max_history_entries=3)
      nqenv.stop_after_seeing_new_results = True

      final_documents = nqenv.get_final_document_list()
      self.assertLen(final_documents, 3)
      self.assertEqual(final_documents[0].content, '3')
      self.assertEqual(final_documents[1].content, '4')
      self.assertEqual(final_documents[2].content, '0')

      # stop_after_seeing_new_results=False with multiple steps in the history
      nqenv.stop_after_seeing_new_results = False
      final_documents = nqenv.get_final_document_list()
      self.assertLen(final_documents, 3)
      self.assertEqual(final_documents[0].content, '5')
      self.assertEqual(final_documents[1].content, '3')
      self.assertEqual(final_documents[2].content, '-1')

  def test_get_final_aggregated_documents(self):
    new_flags = get_updated_default_flags(
        num_documents_to_retrieve=3, use_aggregated_documents=1)
    with flagsaver.flagsaver(**new_flags):
      # stop_after_seeing_new_results with multiple steps in the history
      nqenv = create_env(max_history_entries=3)
      nqenv.stop_after_seeing_new_results = True

      final_documents = nqenv.get_final_document_list()
      self.assertLen(final_documents, 3)
      self.assertEqual(final_documents[0].content, '4')
      self.assertEqual(final_documents[1].content, '3')
      self.assertEqual(final_documents[2].content, '2')

      # stop_after_seeing_new_results=False with multiple steps in the history
      nqenv.stop_after_seeing_new_results = False
      final_documents = nqenv.get_final_document_list()
      self.assertLen(final_documents, 3)
      self.assertEqual(final_documents[0].content, '5')
      self.assertEqual(final_documents[1].content, '4')
      self.assertEqual(final_documents[2].content, '3')

  def test_get_current_and_previous_documents_list(self):
    new_flags = get_updated_default_flags(
        num_documents_to_retrieve=3, use_aggregated_documents=0)
    with flagsaver.flagsaver(**new_flags):
      # stop_after_seeing_new_results with multiple steps in the history
      nqenv = create_env(max_history_entries=3)

      current_documents = nqenv._get_current_documents_list(2)
      self.assertLen(current_documents, 3)
      self.assertEqual(current_documents[0].content, '3')
      self.assertEqual(current_documents[1].content, '4')
      self.assertEqual(current_documents[2].content, '0')

      previous_documents = nqenv._get_previous_documents_list(2)
      self.assertLen(previous_documents, 3)
      self.assertEqual(previous_documents[0].content, '1')
      self.assertEqual(previous_documents[1].content, '2')
      self.assertEqual(previous_documents[2].content, '3')

      current_documents = nqenv._get_current_documents_list(3)
      self.assertLen(current_documents, 3)
      self.assertEqual(current_documents[0].content, '5')
      self.assertEqual(current_documents[1].content, '3')
      self.assertEqual(current_documents[2].content, '-1')

      previous_documents = nqenv._get_previous_documents_list(3)
      self.assertLen(previous_documents, 3)
      self.assertEqual(previous_documents[0].content, '3')
      self.assertEqual(previous_documents[1].content, '4')
      self.assertEqual(previous_documents[2].content, '0')

  def test_get_current_and_previous_aggregated_documents_list(self):
    new_flags = get_updated_default_flags(
        num_documents_to_retrieve=3, use_aggregated_documents=1)
    with flagsaver.flagsaver(**new_flags):
      # stop_after_seeing_new_results with multiple steps in the history
      nqenv = create_env(max_history_entries=3)

      current_documents = nqenv._get_current_documents_list(2)
      self.assertLen(current_documents, 3)
      self.assertEqual(current_documents[0].content, '4')
      self.assertEqual(current_documents[1].content, '3')
      self.assertEqual(current_documents[2].content, '2')

      previous_documents = nqenv._get_previous_documents_list(2)
      self.assertLen(previous_documents, 3)
      self.assertEqual(previous_documents[0].content, '3')
      self.assertEqual(previous_documents[1].content, '2')
      self.assertEqual(previous_documents[2].content, '1')

      current_documents = nqenv._get_current_documents_list(3)
      self.assertLen(current_documents, 3)
      self.assertEqual(current_documents[0].content, '5')
      self.assertEqual(current_documents[1].content, '4')
      self.assertEqual(current_documents[2].content, '3')

      previous_documents = nqenv._get_previous_documents_list(3)
      self.assertLen(previous_documents, 3)
      self.assertEqual(previous_documents[0].content, '4')
      self.assertEqual(previous_documents[1].content, '3')
      self.assertEqual(previous_documents[2].content, '2')

  def test_valid_full_words_in_obs(self):
    new_flags = get_updated_default_flags(
        num_documents_to_retrieve=3,
    )
    with flagsaver.flagsaver(**new_flags):
      original_query = environment_pb2.GetQueryResponse()
      original_query.query = 'who was the first emperor of ancient china'
      nqenv = env.NQEnv(
          nq_server=None,
          state=types.EnvState(original_query=original_query),
          training=False,
          stop_after_seeing_new_results=True)
      # Only the original query is in the history.
      nqenv.state.history = [
          types.HistoryEntry(
              query=original_query.query,
              original_query=original_query.query,
              documents=[
                  create_document(
                      content='Chuanqi Huangdi', mr_score=5, answer='Huangdi'),
                  create_document(
                      content='Yuan Dynasty Provinces',
                      mr_score=3,
                      answer='Yuan'),
                  create_document(
                      content='Huangdi Yinfujing Yellow Emperor',
                      mr_score=-1,
                      answer='Yinfujing'),
              ])
      ]
      obs = nqenv._obs()
      self.assertEqual(
          set(obs.valid_words.all_valid_words.full_words), {
              'dynasty', 'yinfujing', 'yellow', 'yuan', 'chuanqi', 'huangdi',
              'provinces', 'emperor'
          })
      self.assertEqual(
          set(obs.valid_words.question_valid_words.full_words),
          {'first', 'emperor', 'ancient', 'china'})
      self.assertEqual(
          set(obs.valid_words.answer_valid_words.full_words),
          {'huangdi', 'yuan', 'yinfujing'})
      self.assertEqual(
          set(obs.valid_words.document_valid_words.full_words),
          set(obs.valid_words.all_valid_words.full_words))

      # query includes terms with +/- contents/titles
      nqenv.state.history[-1].query = (
          'who was the first emperor of ancient china '
          '+(contents:"emperor") -(contents:"provinces") '
          '+(title:"huangdi") -(title:"chinese")')
      obs = nqenv._obs()
      self.assertEqual(
          set(obs.valid_words.all_valid_words.full_words),
          {'dynasty', 'yinfujing', 'yellow', 'yuan', 'chuanqi'})
      self.assertEqual(
          set(obs.valid_words.question_valid_words.full_words),
          {'first', 'ancient', 'china'})
      self.assertEqual(
          set(obs.valid_words.answer_valid_words.full_words),
          {'yuan', 'yinfujing'})
      self.assertEqual(
          set(obs.valid_words.document_valid_words.full_words),
          set(obs.valid_words.all_valid_words.full_words))

      # query includes only added terms
      nqenv.state.history[-1].query = (
          'who was the first emperor of ancient china emperor '
          'huangdi')
      obs = nqenv._obs()
      self.assertEqual(
          set(obs.valid_words.all_valid_words.full_words),
          {'dynasty', 'yinfujing', 'yellow', 'yuan', 'chuanqi', 'provinces'})
      self.assertEqual(
          set(obs.valid_words.question_valid_words.full_words),
          {'first', 'ancient', 'china'})
      self.assertEqual(
          set(obs.valid_words.answer_valid_words.full_words),
          {'yuan', 'yinfujing'})
      self.assertEqual(
          set(obs.valid_words.document_valid_words.full_words),
          set(obs.valid_words.all_valid_words.full_words))

      # query includes both terms with +/- contents/titles and added terms
      nqenv.state.history[-1].query = (
          'who was the first emperor of ancient china emperor '
          'huangdi +(contents:"emperor") provinces '
          '-(title:"chuanqi") chinese')
      obs = nqenv._obs()
      self.assertEqual(
          set(obs.valid_words.all_valid_words.full_words),
          {'dynasty', 'yinfujing', 'yellow', 'yuan'})
      self.assertEqual(
          set(obs.valid_words.question_valid_words.full_words),
          {'first', 'ancient', 'china'})
      self.assertEqual(
          set(obs.valid_words.answer_valid_words.full_words),
          {'yuan', 'yinfujing'})
      self.assertEqual(
          set(obs.valid_words.document_valid_words.full_words),
          set(obs.valid_words.all_valid_words.full_words))


if __name__ == '__main__':
  tf.test.main()

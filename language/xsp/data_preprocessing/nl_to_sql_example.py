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
"""Standard format for an example mapping from NL to SQL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.xsp.data_preprocessing.language_utils import get_wordpieces
from language.xsp.data_preprocessing.language_utils import Wordpiece
from language.xsp.data_preprocessing.schema_utils import DatabaseTable
from language.xsp.data_preprocessing.schema_utils import get_schema_entities
from language.xsp.data_preprocessing.schema_utils import process_tables
from language.xsp.data_preprocessing.sql_utils import SQLQuery


class NLToSQLInput(object):
  """Contains information about the input to a NL to SQL model."""

  def __init__(self):
    self.original_utterance = None
    self.utterance_wordpieces = list()
    self.tables = list()

  def to_json(self):
    """Returns the JSON form of this class."""
    return {
        'utterance_wordpieces': [
            wordpiece.to_json() for wordpiece in self.utterance_wordpieces
        ],
        'tables': [table.to_json() for table in self.tables],
        'original_utterance': self.original_utterance
    }

  def from_json(self, dictionary):
    """Loads the NLToSQLInput attributes from a dictionary."""
    self.original_utterance = dictionary['original_utterance']
    self.tables = [
        DatabaseTable().from_json(table) for table in dictionary['tables']
    ]
    self.utterance_wordpieces = [
        Wordpiece().from_json(wordpiece)
        for wordpiece in dictionary['utterance_wordpieces']
    ]

    return self


class NLToSQLExample(object):
  """Contains both inputs and outputs for a NL to SQL example."""

  def __init__(self):
    self.model_input = NLToSQLInput()
    self.gold_sql_query = SQLQuery()

  def to_json(self):
    return {
        'model_input': self.model_input.to_json(),
        'gold_sql_query': self.gold_sql_query.to_json()
    }

  def from_json(self, dictionary):
    self.model_input = self.model_input.from_json(dictionary['model_input'])
    self.gold_sql_query = self.gold_sql_query.from_json(
        dictionary['gold_sql_query'])

    return self

  def gold_query_string(self):
    """Generates a query string from the decoder actions for an example."""
    gold_query = list()
    for action in self.gold_sql_query.actions:
      if action.symbol:
        gold_query.append(action.symbol)
      elif action.entity_copy:
        copy_action = action.entity_copy
        if copy_action.copied_table:
          gold_query.append(copy_action.copied_table.original_table_name)
        else:
          gold_query.append(copy_action.copied_column.original_column_name)
      else:
        gold_query.append(action.utterance_copy.wordpiece)
    return ' '.join(gold_query)


def populate_utterance(example, utterance, schema, tokenizer):
  """Sets the model input for a NLToSQLExample."""
  example.model_input.original_utterance = utterance

  schema_entities = get_schema_entities(schema)

  # Set the utterance wordpieces
  try:
    wordpieces, aligned_schema_entities = get_wordpieces(
        example.model_input.original_utterance, tokenizer, schema_entities)
    example.model_input.utterance_wordpieces.extend(wordpieces)

    # Set the table information
    example.model_input.tables.extend(
        process_tables(schema, tokenizer, aligned_schema_entities))
  except UnicodeDecodeError as e:
    print(unicode(e))
    return None
  return example

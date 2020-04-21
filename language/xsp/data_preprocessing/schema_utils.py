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
"""Contains utilities for processing database schemas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from language.xsp.data_preprocessing.language_utils import get_wordpieces
from language.xsp.data_preprocessing.language_utils import Wordpiece

ACCEPTABLE_COL_TYPES = {'text', 'number', 'others', 'time', 'boolean'}


class TableColumn(object):
  """Contains information about column in a database table."""

  def __init__(self):
    self.original_column_name = None
    self.column_name_wordpieces = list()
    self.column_type = None
    self.table_name = None
    self.is_foreign_key = None
    self.matches_to_utterance = None

  def to_json(self):
    assert self.column_type in ACCEPTABLE_COL_TYPES, (
        ('Column type not '
         'recognized: %r; name: %r') %
        (self.column_type, self.original_column_name))
    return {
        'original_column_name': self.original_column_name,
        'column_name_wordpieces': [
            wordpiece.to_json() for wordpiece in self.column_name_wordpieces
        ],
        'column_type': self.column_type,
        'table_name': self.table_name,
        'is_foreign_key': self.is_foreign_key,
        'matches_to_utterance': self.matches_to_utterance
    }

  def from_json(self, dictionary):
    """Sets the properties of the column from a dictionary representation."""
    self.original_column_name = dictionary['original_column_name']
    self.column_name_wordpieces = [
        Wordpiece().from_json(wordpiece)
        for wordpiece in dictionary['column_name_wordpieces']
    ]
    self.column_type = dictionary['column_type']
    self.table_name = dictionary['table_name']
    self.is_foreign_key = dictionary['is_foreign_key']
    self.matches_to_utterance = dictionary['matches_to_utterance']
    assert self.column_type in ACCEPTABLE_COL_TYPES, (
        ('Column type not '
         'recognized: %r; name: %r') %
        (self.column_type, self.original_column_name))

    return self


class DatabaseTable(object):
  """Contains information about a table in a database."""

  def __init__(self):
    self.original_table_name = None
    self.table_name_wordpieces = list()
    self.table_columns = list()
    self.matches_to_utterance = None

  def to_json(self):
    return {
        'original_table_name': self.original_table_name,
        'table_name_wordpieces': [
            wordpiece.to_json() for wordpiece in self.table_name_wordpieces
        ],
        'table_columns': [column.to_json() for column in self.table_columns],
        'matches_to_utterance': self.matches_to_utterance
    }

  def from_json(self, dictionary):
    """Converts from a JSON dictionary to a DatabaseTable object."""
    self.original_table_name = dictionary['original_table_name']
    self.table_name_wordpieces = [
        Wordpiece().from_json(wordpiece)
        for wordpiece in dictionary['table_name_wordpieces']
    ]
    self.table_columns = [
        TableColumn().from_json(column)
        for column in dictionary['table_columns']
    ]
    self.matches_to_utterance = dictionary['matches_to_utterance']

    return self

  def __str__(self):
    return json.dumps(self.to_json())


def column_is_primary_key(column):
  """Returns whether a column object is marked as a primary key."""
  primary_key = column['is primary key']
  if isinstance(primary_key, str):
    primary_key = primary_key.lower()
    if primary_key in {'y', 'n', 'yes', 'no', '-'}:
      primary_key = primary_key in {'y', 'yes'}
    else:
      raise ValueError('primary key should be a boolean: ' + primary_key)
  return primary_key


def column_is_foreign_key(column):
  """Returns whether a column object is marked as a foreign key."""
  foreign_key = column['is foreign key']
  if isinstance(foreign_key, str):
    foreign_key = foreign_key.lower()

    if foreign_key in {'y', 'n', 'yes', 'no', '-'}:
      foreign_key = foreign_key in {'y', 'yes'}
    else:
      raise ValueError('Foreign key should be a boolean: ' + foreign_key)

  return foreign_key


def process_columns(columns, tokenizer, table_name, aligned_schema_entities):
  """Processes a column in a table to a TableColumn object."""
  column_obj_list = list()
  for column in columns:
    column_obj = TableColumn()
    column_obj.original_column_name = column['field name']
    column_obj.column_name_wordpieces.extend(
        get_wordpieces(
            column_obj.original_column_name.replace('_', ' '), tokenizer)[0])
    col_type = column['type'].lower()
    if 'int' in col_type or 'float' in col_type or 'double' in col_type or 'decimal' in col_type:
      col_type = 'number'
    if 'varchar' in col_type or 'longtext' in col_type:
      col_type = 'text'
    column_obj.column_type = col_type
    column_obj.table_name = table_name

    column_obj.matches_to_utterance = column_obj.original_column_name.lower(
    ).replace('_', ' ') in aligned_schema_entities

    column_obj.is_foreign_key = column_is_foreign_key(column)
    column_obj_list.append(column_obj)
  return column_obj_list


def process_table(table_name, columns, tokenizer, aligned_schema_entities):
  """Processes a schema table into a DatabaseTable object."""
  table_obj = DatabaseTable()
  table_obj.original_table_name = table_name

  table_obj.matches_to_utterance = table_obj.original_table_name.lower(
  ).replace('_', ' ') in aligned_schema_entities

  # Name wordpieces. Remove underscores then tokenize.
  table_obj.table_name_wordpieces.extend(
      get_wordpieces(table_name.replace('_', ' '), tokenizer)[0])

  table_obj.table_columns.extend(
      process_columns(columns, tokenizer, table_obj.original_table_name,
                      aligned_schema_entities))

  return table_obj


def process_tables(schema, tokenizer, aligned_schema_entities):
  """Processes each table in a schema."""
  return [
      process_table(table_name, columns, tokenizer, aligned_schema_entities)
      for table_name, columns in schema.items()
  ]


def get_schema_entities(schema):
  """Gets the schema entities (column and table names) for a schema."""
  names = set()
  for table_name, cols in schema.items():
    names.add(table_name.lower().replace('_', ' '))
    for col in cols:
      names.add(col['field name'].lower().replace('_', ' '))
  return names

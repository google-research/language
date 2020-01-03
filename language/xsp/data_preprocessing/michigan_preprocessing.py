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
"""Loads the Michigan datasets from a file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json

from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample
from language.xsp.data_preprocessing.nl_to_sql_example import populate_utterance

import tensorflow.compat.v1.gfile as gfile


def get_nl_sql_pairs(filepath, splits, with_dbs=False):
  """Gets pairs of natural language and corresponding gold SQL for Michigan."""
  with gfile.Open(filepath) as infile:
    data = json.load(infile)

  pairs = list()

  tag = '[' + filepath.split('/')[-1].split('.')[0] + ']'
  print('Getting examples with tag ' + tag)

  # The UMichigan data is split by anonymized queries, where values are
  # anonymized but table/column names are not. However, our experiments are
  # performed on the original splits of the data.
  for query in data:
    # Take the first SQL query only. From their Github documentation:
    # "Note - we only use the first query, but retain the variants for
    #  completeness"
    anonymized_sql = query['sql'][0]

    # It's also associated with a number of natural language examples, which
    # also contain anonymous tokens. Save the de-anonymized utterance and query.
    for example in query['sentences']:
      if example['question-split'] not in splits:
        continue

      nl = example['text']
      sql = anonymized_sql

      # Go through the anonymized values and replace them in both the natural
      # language and the SQL.
      #
      # It's very important to sort these in descending order. If one is a
      # substring of the other, it shouldn't be replaced first lest it ruin the
      # replacement of the superstring.
      for variable_name, value in sorted(
          example['variables'].items(), key=lambda x: len(x[0]), reverse=True):
        if not value:
          # TODO(alanesuhr) While the Michigan repo says to use a - here, the
          # thing that works is using a % and replacing = with LIKE.
          #
          # It's possible that I should remove such clauses from the SQL, as
          # long as they lead to the same table result. They don't align well
          # to the natural language at least.
          #
          # See: https://github.com/jkkummerfeld/text2sql-data/tree/master/data
          value = '%'

        nl = nl.replace(variable_name, value)
        sql = sql.replace(variable_name, value)

      # In the case that we replaced an empty anonymized value with %, make it
      # compilable new allowing equality with any string.
      sql = sql.replace('= "%"', 'LIKE "%"')

      if with_dbs:
        pairs.append((nl, sql, example['table-id']))
      else:
        pairs.append((nl, sql))

  return pairs


def read_schema(schema_csv):
  """Loads a database schema from a CSV representation."""
  tables = {}
  with gfile.Open(schema_csv) as infile:
    for column in csv.DictReader(
        infile,
        quotechar='"',
        delimiter=',',
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True):
      column = {
          key.lower().strip(): value for key, value in column.items() if key
      }

      table_name = column['table name']
      if table_name != '-':
        if table_name not in tables:
          tables[table_name] = list()
        column.pop('table name')
        tables[table_name].append(column)
  return tables


def convert_michigan(nl, sql, schema, tokenizer, generate_sql):
  """Converts a Michigan example to a NLToSQLExample."""
  example = NLToSQLExample()
  populate_utterance(example, nl, schema, tokenizer)

  if generate_sql:
    raise ValueError(
        'Generating annotated SQL is not yet supported for Michigan datasets. '
        'Tried to annotate: ' + sql)
  return example

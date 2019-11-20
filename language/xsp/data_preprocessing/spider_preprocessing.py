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
"""Contains functions for loading and preprocessing the Spider data."""
import json

from language.xsp.data_preprocessing import abstract_sql_converters
from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample
from language.xsp.data_preprocessing.nl_to_sql_example import populate_utterance
from language.xsp.data_preprocessing.sql_parsing import populate_sql
from language.xsp.data_preprocessing.sql_utils import preprocess_sql
import sqlparse
import tensorflow.gfile as gfile

WRONG_TRAINING_EXAMPLES = {
    # In this query the SQL query mentions a ref_company_types table that is not
    # present in the schema of the assets_maintenance database.
    'What is the description of the type of the company who concluded its '
    'contracts most recently?',
}


def process_dbs(raw_dbs):
  """Converts database specification directly from Spider to our format."""
  dbs = dict()

  # Should return a list of databases, each which is a dictionary whose keys are
  # table names, and values are columns; columns have 'field name',
  # 'is foreign key', 'is primary key', and 'type' annotations.
  for db in raw_dbs:
    db_dict = dict()
    for table in db['table_names_original']:
      db_dict[table] = list()
    all_foreign_indices = \
      set([k[0] for k in db['foreign_keys']]
          + [k[1] for k in db['foreign_keys']])
    for global_column_idx, (table_idx, column_name) in \
        enumerate(db['column_names_original']):
      if table_idx >= 0:
        is_primary_key = global_column_idx in db['primary_keys']
        is_foreign_key = global_column_idx in all_foreign_indices
        column_dict = {
            'field name': column_name,
            'is primary key': is_primary_key,
            'is foreign key': is_foreign_key,
            'type': db['column_types'][global_column_idx]
        }
        db_dict[db['table_names_original'][table_idx]].append(column_dict)
    dbs[db['db_id']] = db_dict
  return dbs


def load_spider_tables(filenames):
  """Loads database schemas from the specified filenames."""
  examples = dict()
  for filename in filenames.split(','):
    with gfile.GFile(filename) as training_file:
      examples.update(process_dbs(json.load(training_file)))
  return examples


def load_spider_examples(filenames):
  """Loads examples from the Spider dataset from the specified files."""
  examples = []
  for filename in filenames.split(','):
    with gfile.GFile(filename) as training_file:
      examples += json.load(training_file)
  return examples


def convert_spider(spider_example,
                   schema,
                   wordpiece_tokenizer,
                   generate_sql,
                   anonymize_values,
                   abstract_sql=False,
                   table_schemas=None,
                   allow_value_generation=False):
  """Converts a Spider example to the standard format.

  Args:
    spider_example: JSON object for SPIDER example in original format.
    schema: JSON object for SPIDER schema in converted format.
    wordpiece_tokenizer: language.bert.tokenization.FullTokenizer instance.
    generate_sql: If True, will populate SQL.
    anonymize_values: If True, anonymizes values in SQL.
    abstract_sql: If True, use under-specified FROM clause.
    table_schemas: required if abstract_sql, list of TableSchema tuples.

  Returns:
    NLToSQLExample instance.
  """
  if spider_example['question'] in WRONG_TRAINING_EXAMPLES:
    return None

  sql_query = spider_example['query'].rstrip('; ')
  sql_query = sqlparse.parse(preprocess_sql(sql_query.lower()))[0]

  example = NLToSQLExample()

  # Set the input
  populate_utterance(example, ' '.join(spider_example['question_toks']), schema,
                     wordpiece_tokenizer)

  # Set the output
  successful_copy = True
  if generate_sql:
    if abstract_sql:
      successful_copy = abstract_sql_converters.populate_abstract_sql(example,
                                                    spider_example['query'],
                                                    table_schemas)
    else:
      successful_copy = populate_sql(sql_query, example, anonymize_values)

  # If the example contained an unsuccessful copy action, and values should not be
  # generated, then return an empty example.
  if not successful_copy and not allow_value_generation:
    return None

  return example

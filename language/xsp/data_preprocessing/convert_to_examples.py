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
"""Used to convert raw data to the standard JSON format for examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from bert.tokenization import FullTokenizer
from language.xsp.data_preprocessing import abstract_sql
from language.xsp.data_preprocessing import abstract_sql_converters
from language.xsp.data_preprocessing.michigan_preprocessing import convert_michigan
from language.xsp.data_preprocessing.michigan_preprocessing import get_nl_sql_pairs
from language.xsp.data_preprocessing.michigan_preprocessing import read_schema
from language.xsp.data_preprocessing.spider_preprocessing import convert_spider
from language.xsp.data_preprocessing.spider_preprocessing import load_spider_examples
from language.xsp.data_preprocessing.spider_preprocessing import load_spider_tables
from language.xsp.data_preprocessing.wikisql_preprocessing import convert_wikisql
from language.xsp.data_preprocessing.wikisql_preprocessing import load_wikisql_tables
import tensorflow.gfile as gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset (required for handling '
    'different formats differently.)')

flags.DEFINE_string('input_filepath', None, 'File to read examples from.')

flags.DEFINE_list('splits', None, 'The splits to create examples for.')

flags.DEFINE_string('output_filepath', None, 'File to output examples to.')

flags.DEFINE_string('tokenizer_vocabulary', '',
                    'Filepath to the tokenizer vocabulary.')

flags.DEFINE_bool('generate_sql', False,
                  'Whether to provide SQL labels in the proto.')

flags.DEFINE_bool('anonymize_values', False, 'Whether to anonymize values.')

flags.DEFINE_bool(
    'abstract_sql', True,
    'Whether to provide SQL labels using under-specified FROM clauses.')


def _load_json_from_file(filename):
  with gfile.GFile(filename) as json_file:
    return json.load(json_file)


def process_spider(output_file, debugging_file, tokenizer):
  """Loads, converts, and writes Spider examples to the standard format."""
  if len(FLAGS.splits) > 1:
    raise ValueError('Not expecting more than one split for Spider.')
  split = FLAGS.splits[0]

  table_definitions = load_spider_tables(
      os.path.join(FLAGS.input_filepath, 'tables.json'))
  print('Loaded %d table definitions.' % len(table_definitions))

  spider_examples = \
    load_spider_examples(os.path.join(FLAGS.input_filepath,
                                      split + '.json'))

  num_examples_created = 0
  num_examples_failed = 0

  # TODO(petershaw): Reduce duplication with other code path for schema
  # pre-processing.
  tables_json = _load_json_from_file(
      os.path.join(FLAGS.input_filepath, 'tables.json'))
  spider_table_schemas_map = abstract_sql_converters.spider_table_schemas_map(
      tables_json)

  for spider_example in spider_examples:
    # Make sure the DB specified exists.
    example_db = spider_example['db_id']
    try:
      example = convert_spider(
          spider_example,
          table_definitions[example_db],
          tokenizer,
          FLAGS.generate_sql,
          FLAGS.anonymize_values,
          abstract_sql=FLAGS.abstract_sql,
          table_schemas=spider_table_schemas_map[example_db])
    except abstract_sql.UnsupportedSqlError as e:
      print(e)
      example = None
    if example:
      output_file.write(json.dumps(example.to_json()) + '\n')
      num_examples_created += 1

      debugging_file.write(example.model_input.original_utterance + '\n')
      if FLAGS.generate_sql:
        debugging_file.write(example.gold_query_string() + '\n\n')
    else:
      num_examples_failed += 1
  return num_examples_created, num_examples_failed


def process_wikisql(output_file, debugging_file, tokenizer):
  """Loads, converts, and writes WikiSQL examples to the standard format."""
  # TODO(alanesuhr,petershaw): Support asql for this dataset.
  if FLAGS.generate_sql and FLAGS.abstract_sql:
    raise NotImplementedError(
        'Abstract SQL currently only supported for SPIDER.')

  if len(FLAGS.splits) > 1:
    raise ValueError('Not expecting more than one split for WikiSQL.')
  split = FLAGS.splits[0]

  num_examples_created = 0
  num_examples_failed = 0

  data_filepath = os.path.join(FLAGS.input_filepath,
                               FLAGS.dataset_name + '.json')

  paired_data = get_nl_sql_pairs(
      data_filepath, set(FLAGS.splits), with_dbs=True)

  table_definitions = \
    load_wikisql_tables(os.path.join(FLAGS.input_filepath,
                                     split + '.tables.jsonl'))

  for input_example in paired_data:
    example = \
      convert_wikisql(input_example, table_definitions[input_example[2]],
                      tokenizer,
                      FLAGS.generate_sql,
                      FLAGS.anonymize_values)
    if example:
      output_file.write(json.dumps(example.to_json()) + '\n')
      num_examples_created += 1

      debugging_file.write(example.model_input.original_utterance + '\n')
      if FLAGS.generate_sql:
        debugging_file.write(example.gold_query_string() + '\n\n')
    else:
      num_examples_failed += 1
  return num_examples_created, num_examples_failed


def process_michigan_datasets(output_file, debugging_file, tokenizer):
  """Loads, converts, and writes Michigan examples to the standard format."""
  # TODO(alanesuhr,petershaw): Support asql for this dataset.
  if FLAGS.generate_sql and FLAGS.abstract_sql:
    raise NotImplementedError(
        'Abstract SQL currently only supported for SPIDER.')

  schema_csv = os.path.join(FLAGS.input_filepath,
                            FLAGS.dataset_name + '_schema.csv')
  data_filepath = os.path.join(FLAGS.input_filepath,
                               FLAGS.dataset_name + '.json')

  # Don't actually provide table entities.
  num_examples_created = 0
  num_examples_failed = 0

  print('Loading from ' + data_filepath)
  paired_data = get_nl_sql_pairs(data_filepath, set(FLAGS.splits))
  print('Loaded %d examples.' % len(paired_data))

  schema = read_schema(schema_csv)

  for nl, sql in paired_data:
    example = convert_michigan(nl, sql, schema, tokenizer, FLAGS.generate_sql)
    if example is not None:
      output_file.write(json.dumps(example.to_json()) + '\n')
      num_examples_created += 1

      debugging_file.write(example.model_input.original_utterance + '\n')
      if FLAGS.generate_sql:
        debugging_file.write(example.gold_query_string() + '\n\n')
    else:
      num_examples_failed += 1

  return num_examples_created, num_examples_failed


def main(unused_argv):
  tokenizer = FullTokenizer(FLAGS.tokenizer_vocabulary)

  print('Loading ' + str(FLAGS.dataset_name) + ' dataset from ' +
        FLAGS.input_filepath)

  # The debugging file saves all of the processed SQL queries.
  debugging_file = gfile.Open(
      os.path.join('/'.join(FLAGS.output_filepath.split('/')[:-1]),
                   FLAGS.dataset_name + '_'.join(FLAGS.splits) + '_gold.txt'),
      'w')

  # The output file will save a sequence of string-serialized JSON objects, one
  # line per object.
  output_file = gfile.Open(os.path.join(FLAGS.output_filepath), 'w')

  if FLAGS.dataset_name.lower() == 'spider':
    num_examples_created, num_examples_failed = process_spider(
        output_file, debugging_file, tokenizer)
  elif FLAGS.dataset_name.lower() == 'wikisql':
    num_examples_created, num_examples_failed = process_wikisql(
        output_file, debugging_file, tokenizer)
  else:
    num_examples_created, num_examples_failed = process_michigan_datasets(
        output_file, debugging_file, tokenizer)

  print('Wrote %s examples, could not annotate %s examples.' %
        (num_examples_created, num_examples_failed))
  debugging_file.write('Wrote %s examples, could not annotate %s examples.' %
                       (num_examples_created, num_examples_failed))
  debugging_file.close()
  output_file.close()


if __name__ == '__main__':
  app.run(main)

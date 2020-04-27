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
r"""Binary for testing abstract SQL on SPIDER.

Example usage:


${PATH_TO_BINARY} \
  --input=${SPIDER_DIR}/spider/train_spider.json \
  --tables=${SPIDER_DIR}/spider/tables.json \
  --gold_sql_output=${SPIDER_DIR}/absql/gold.sql \
  --abstract_sql_output=${SPIDER_DIR}/absql/absql.sql

python ${SPIDER_DIR}/spider-master/evaluation.py \
  --gold "${SPIDER_DIR}/absql/gold.sql" \
  --pred "${SPIDER_DIR}/absql/absql.sql" \
  --etype match \
  --db "${SPIDER_DIR}/spider/database" \
  --table "${SPIDER_DIR}/spider/tables.json"
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json

from absl import app
from absl import flags

from language.xsp.data_preprocessing import abstract_sql
from language.xsp.data_preprocessing import abstract_sql_converters

FLAGS = flags.FLAGS

flags.DEFINE_string('input', '', 'Path to SPIDER json examples.')
flags.DEFINE_string('tables', '', 'Path to SPIDER json tables.')
flags.DEFINE_string('gold_sql_output', '', 'Path to output gold SQL.')
flags.DEFINE_string('abstract_sql_output', '',
                    'Path to output reconstructed abstract SQL.')
flags.DEFINE_bool('restore_from_clause', True,
                  'Whether to remove and restore the FROM clause.')
flags.DEFINE_bool('keep_going', False,
                  'Whether to keep going after a ParseError.')


def _load_json(filename):
  with open(filename) as json_file:
    return json.load(json_file)


def _get_abstract_sql(gold_sql, foreign_keys, table_schema,
                      restore_from_clause):
  """Returns string using abstract SQL transformations."""
  print('Processing query:\n%s' % gold_sql)
  sql_spans = abstract_sql.sql_to_sql_spans(gold_sql, table_schema)
  if restore_from_clause:
    sql_spans = abstract_sql.replace_from_clause(sql_spans)
    print('Replaced clause query:\n%s' %
          abstract_sql.sql_spans_to_string(sql_spans))
    sql_spans = abstract_sql.restore_from_clause(sql_spans, foreign_keys)
  return abstract_sql.sql_spans_to_string(sql_spans)


def main(unused_argv):
  table_json = _load_json(FLAGS.tables)
  # Map of database id to a list of ForiegnKeyRelation tuples.
  foreign_key_map = abstract_sql_converters.spider_foreign_keys_map(table_json)
  table_schema_map = abstract_sql_converters.spider_table_schemas_map(
      table_json)

  examples = _load_json(FLAGS.input)
  num_failures = 0
  num_examples = 0
  with open(FLAGS.gold_sql_output, 'w') as gold_sql_file:
    with open(FLAGS.abstract_sql_output, 'w') as abstract_sql_file:
      for example in examples:
        num_examples += 1
        print('Parsing example number %s.' % num_examples)
        gold_sql_query = example['query']
        foreign_keys = foreign_key_map[example['db_id']]
        table_schema = table_schema_map[example['db_id']]
        try:
          abstract_sql_query = _get_abstract_sql(gold_sql_query, foreign_keys,
                                                 table_schema,
                                                 FLAGS.restore_from_clause)
        except abstract_sql.UnsupportedSqlError as e:
          print('Error for query:\n%s' % gold_sql_query)
          num_failures += 1
          if not FLAGS.keep_going:
            raise e
          else:
            continue
        else:
          # Write SQL to output files.
          gold_sql_query = gold_sql_query.replace('\t', ' ')
          gold_sql_file.write('%s\t%s\n' % (gold_sql_query, example['db_id']))
          abstract_sql_file.write('%s\t%s\n' %
                                  (abstract_sql_query, example['db_id']))
  print('Examples: %s' % num_examples)
  print('Failed parses: %s' % num_failures)


if __name__ == '__main__':
  app.run(main)

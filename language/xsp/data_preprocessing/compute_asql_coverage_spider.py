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
r"""Compute coverage for Abstract SQL for Spider.

Example usage:


${PATH_TO_BINARY} \
  --spider_examples_json=${SPIDER_DIR}/train_spider.json \
  --spider_tables_json=${SPIDER_DIR}/tables.json \
  --alsologtostderr
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

flags.DEFINE_string('spider_examples_json', '', 'Path to Spider json examples')
flags.DEFINE_string('spider_tables_json', '', 'Path to Spider json tables')


def _load_json(filename):
  with open(filename) as json_file:
    return json.load(json_file)


def compute_spider_coverage(spider_examples_json, spider_tables_json):
  """Prints out statistics for asql conversions."""
  table_json = _load_json(spider_tables_json)
  # Map of database id to a list of ForiegnKeyRelation tuples.
  foreign_key_map = abstract_sql_converters.spider_foreign_keys_map(table_json)
  table_schema_map = abstract_sql_converters.spider_table_schemas_map(
      table_json)
  examples = _load_json(spider_examples_json)
  num_examples = 0
  num_conversion_failures = 0
  num_reconstruction_failtures = 0
  for example in examples:
    num_examples += 1
    print('Parsing example number %s.' % num_examples)
    gold_sql_query = example['query']
    foreign_keys = foreign_key_map[example['db_id']]
    table_schema = table_schema_map[example['db_id']]
    try:
      sql_spans = abstract_sql.sql_to_sql_spans(gold_sql_query, table_schema)
      sql_spans = abstract_sql.replace_from_clause(sql_spans)
    except abstract_sql.UnsupportedSqlError as e:
      print('Error converting:\n%s\n%s' % (gold_sql_query, e))
      num_conversion_failures += 1
    else:
      try:
        sql_spans = abstract_sql.restore_from_clause(sql_spans, foreign_keys)
      except abstract_sql.UnsupportedSqlError as e:
        print('Error recontructing:\n%s\n%s' % (gold_sql_query, e))
        num_reconstruction_failtures += 1
  print('Examples: %s' % num_examples)
  print('Failed conversions: %s' % num_conversion_failures)
  print('Failed reconstructions: %s' % num_reconstruction_failtures)


def main(unused_argv):
  compute_spider_coverage(FLAGS.spider_examples_json, FLAGS.spider_tables_json)


if __name__ == '__main__':
  app.run(main)

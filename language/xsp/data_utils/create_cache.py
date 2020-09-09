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
"""Creates a cache for the specified dataset by executing the gold queries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sqlite3


def main(dataset_name, cache_path, errors_filepath, splits):
  if dataset_name == 'spider':
    pass
  else:
    db = sqlite3.connect(os.path.join('databases', dataset_name + '.db'))
    c = db.cursor()

    cache = dict()

    if os.path.exists(cache_path):
      print('Reading existing cache from %s' % cache_path)
      with open(cache_path) as infile:
        cache = json.loads(infile.read())

    num_empty = 0
    num_queries = 0

    with open(os.path.join(dataset_name, dataset_name + '.json')) as infile:
      data = json.load(infile)

    for query in data:
      for example in query['sentences']:
        if example['question-split'] not in splits:
          continue

        anon_sql = query['sql'][0]
        nl = example['text']

        for variable, value in sorted(
            example['variables'].items(), key=lambda x: len(x[0]),
            reverse=True):
          if not value:
            value = '%'
          nl = nl.replace(variable, value)
          print('%s\t%s' % (variable, value))
          anon_sql = anon_sql.replace(variable, value)
        anon_sql = anon_sql.replace('= "%"', 'LIKE "%"')
        anon_sql = anon_sql.replace('= %', 'LIKE "%"')

        if 'scholar' in dataset_name.lower():
          new_pred = ''
          last_quote = ''
          for char in anon_sql:
            new_pred += char
            if char in {'"', '\''} and not last_quote:
              last_quote = char
            elif char == last_quote:
              last_quote = ''
              new_pred += ' COLLATE NOCASE'
            anon_sql = new_pred

        if 'advising' in dataset_name.lower():
          # Fix so that it's selecting a concat of columns instead.
          if 'count' in anon_sql.lower():
            # Find range of count thing
            count_start_idx = anon_sql.lower().index('count')
            count_end_idx = (
                count_start_idx + anon_sql.lower()[count_start_idx:].index(')'))

            if ',' in anon_sql[count_start_idx:count_end_idx]:
              problem_segment = anon_sql[count_start_idx:count_end_idx]
              problem_segment = problem_segment.replace(',', '||')
              anon_sql = (
                  anon_sql[:count_start_idx] + problem_segment +
                  anon_sql[count_end_idx:])
          prev_token = ''
          bad_tokens = set()
          for token in anon_sql.split():
            if prev_token == '=':
              if (token[0] in {'"', '\''} and token[-1] in {'"', '\''} and
                  token[-2].isnumeric() and not token[1].isnumeric()):
                bad_tokens.add(token)
              elif token[-1].isnumeric() and not token[0].isnumeric():
                bad_tokens.add(token)
              prev_token = token
          for token in bad_tokens:
            anon_sql = anon_sql.replace('= ' + token, 'LIKE "%"')
          if bad_tokens:
            print(bad_tokens)

        print(nl)

        # Two specific exceptions on utterances that need correction or take a
        # long time to process.
        if nl == ('What is the number of businesses user Michelle reviews per '
                  'month ?'):
          anon_sql = ('select count(distinct(review.text)), review.month from '
                      'review where review.user_id in (select user_id from '
                      'user where user.name = \'Michelle\') group by '
                      'review.month;')

        if nl == ('return me the number of papers in " University of '
                  'Michigan " in Databases area .'):
          results = '121572'
          cache[anon_sql] = results
        else:
          if anon_sql not in cache:
            # Update the cache to include this SQL query.
            print(anon_sql)
            try:
              c.execute(anon_sql)
              results = c.fetchall()
            except sqlite3.OperationalError as e:
              with open(errors_filepath) as f:
                f.write(nl + '\n')
                f.write(anon_sql + '\n')
                f.write(str(e) + '\n\n')

              results = list()
            cache[anon_sql] = results
          else:
            results = cache[anon_sql]

          if not results:
            num_empty += 1

          if ('advising' not in dataset_name and nl in cache and
              cache[nl] != anon_sql):
            print(nl)
            print(anon_sql)
            print(cache[nl])
            keep_going = input('Allow this to happen? This utterance will be '
                               'mapped to the second query.').lower() == 'y'
            if not keep_going:
              raise ValueError('NL is the same but anonymized SQL is not.')
        cache[nl] = anon_sql
        num_queries += 1

  print('Num empty: %s' % num_empty)
  print('Total num queries: %s' % num_queries)
  print('Prop empty: %2f' % (100. * num_empty / num_queries))
  db.close()

  print('Writing cache')
  with open(cache_path + '.tmp', 'w') as ofile:
    json.dump(cache, ofile)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_name',
      type=str,
      help='The name of the dataset to create a cache for.')
  parser.add_argument(
      '--splits',
      type=str,
      help='Comma-separated list of split names to create a cache for.')

  args = parser.parse_args()

  main(args.dataset_name,
       os.path.join(args.dataset_name, args.dataset_name + '_cache.json'),
       os.path.join(args.dataset_name, 'exec_errors.txt'),
       args.splits.split(','))

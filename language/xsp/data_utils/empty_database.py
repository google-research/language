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
"""Empties the database contents while retaining the database structure."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import os
import sqlite3


def empty_db(name):
  """Empty DB."""
  db = sqlite3.connect(name)
  c = db.cursor()

  q = ("select name from sqlite_master where type='table' and "
       "name not like 'sqlite_%'")
  c.execute(q)

  table_names = [name[0] for name in c.fetchall()]
  for name in table_names:
    print('emptying ' + name)
    q = 'delete from ' + name
    c.execute(q)
    db.commit()

  db.close()


def main(db_to_del):
  if 'spider_databases' in db_to_del:
    for filename in os.listdir(db_to_del):
      if filename.endswith('.sqlite'):
        empty_db(os.path.join(db_to_del, filename))

  else:
    empty_db(db_to_del)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--db_to_empty',
      type=str,
      help='The name of the database which should be emptied.')
  args = parser.parse_args()
  main(args.db_to_empty)


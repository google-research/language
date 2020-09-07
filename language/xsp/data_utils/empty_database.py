"""Empties the database contents while retaining the database structure."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import os
import sqlite3
import sys


def empty_db(name):
  db = sqlite3.connect(name)
  c = db.cursor()

  q = "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
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
  parser.add_argument('--db_to_empty', 
                      type=str, 
                      help='The name of the database which should be emptied.')
  args = parser.parse_args()
  main(args.db_to_empty) 

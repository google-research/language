"""Adds indices to databases which require them."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import sqlite3
import sys


def main(db_name):
  with open('data_utils/extra_' + db_name + '_indices.txt') as infile:
    indices = infile.read().split('\n')

  db = sqlite3.connect('databases/' + db_name + '.db')
  c = db.cursor()

  for index in indices:
    print('Adding index:')
    print(index)
    q = index
    c.execute(q)
    db.commit()

  db.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--database_name', 
                      type=str, 
                      help='The database to add indices to.')
  args = parser.parse_args()
  main(args.database_name)


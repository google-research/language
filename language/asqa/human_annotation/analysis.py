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
# pylint: disable=g-no-space-after-docstring-summary,g-short-docstring-punctuation,g-explicit-length-test,dangerous-default-value,redefined-outer-name,g-explicit-bool-comparison, missing-function-docstring,missing-module-docstring,raise-missing-from,g-complex-comprehension
"""Example usage.

python3 analysis.py --setup ./setup.tsv --comparisons ./AnnotationInterface.xlsx
--dst ./resuts_of_comparisons.tsv
"""
import argparse

import pandas as pd


def process_eval2(idx, pair, df):
  """Determine the result of each pairwise comparison in Evaluation 2

  ("tie" vs "left" vs "right"). If comparison is not made or made incorrecntly
  (more than one options checked),
  the result is forced to be a "tie" and the warning message is printed.

  Args:
    idx: current row in the raw comparison dataframe
    pair: index of the pairwise comparison
    df: raw comparison dataframe

  Returns:
    result of the comparison
  """

  values = [df.loc[idx, 1], df.loc[idx, 3], df.loc[idx, 5]]

  if sum(values) != 1:
    print(f'Eval 2 for pair {pair} has an issue (zero or more than one checkbox'
          'selected) -- I substitute it with \"tie\", but please investigate!')
    return 'tie'

  return 'tie' if values[0] else 'left' if values[1] else 'right'


def process_pair(idx, df):
  """Parse the result of a given pairwise comparison.

  Args:
    idx: a starting row in the raw comparison dataframe
    df: raw comparison dataframe

  Returns:
    wide representation of the comparison
  """

  pair = int(df.loc[idx, 0].split(' ')[1])
  indicators_left, indicators_right, row = [], [], []

  idx = idx + 6

  while df.loc[idx, 0] != 'Evaluation 2':
    val_left = df.loc[idx, 1]
    val_right = df.loc[idx, 4]

    indicators_left.append(val_left)
    indicators_right.append(val_right)

    idx += 1

  num_question = len(indicators_left)
  acc_left = sum(indicators_left)
  acc_right = sum(indicators_right)

  q1 = process_eval2(idx + 2, pair, df)
  q2 = process_eval2(idx + 4, pair, df)
  q3 = process_eval2(idx + 6, pair, df)

  row = [pair, num_question, acc_left, acc_right, q1, q2, q3]

  return row, idx + 7


def process_comparisons(setup, comparisons):
  """Main function that parses the xlsx representation of the conducted

  pairwise comparisons and returns a convinient representation.

  Args:
    setup: setup dataframe (see readme)
    comparisons: raw comparison dataframe

  Returns:
    Convinient representation of pairwise comparisons.
  """

  pair = 0
  rows = []
  idx = 0

  num_pair = len(setup)

  while pair < num_pair:
    tmp_res = process_pair(idx, comparisons)
    left_model = setup.loc[pair][1]
    right_model = setup.loc[pair][2]
    key = setup.loc[pair][0]

    rows += [tmp_res[0] + [left_model, right_model, key]]
    idx = tmp_res[1]
    pair = tmp_res[0][0]

  df = pd.DataFrame(
      rows,
      columns=[
          'pair', 'numQA', 'accLeft', 'accRight', 'Ambiguity', 'Fluency',
          'Overall', 'leftModel', 'rightModel', 'key'
      ])

  df['accLeft'] = df['accLeft'] / df['numQA']
  df['accRight'] = df['accRight'] / df['numQA']

  return df


def parse_args(argv=None):
  """Argument parser."""

  parser = argparse.ArgumentParser()

  parser.add_argument('--setup', type=str, help='Path to the full setup file')

  parser.add_argument(
      '--comparisons',
      type=str,
      help='Path to the file with results of pairwise comparisons')

  parser.add_argument('--dst', type=str, help='Path to the resulting file')

  # parse the arguments
  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_args()
  setup = pd.read_csv(args.setup, sep='\t', dtype=str)
  comparisons = pd.read_excel(args.comparisons, header=None)
  df = process_comparisons(setup, comparisons)
  df.to_csv(args.dst, sep='\t', index=False)

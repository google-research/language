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
# pylint: disable=g-no-space-after-docstring-summary,g-space-before-docstring-summary,g-short-docstring-punctuation,g-explicit-length-test,dangerous-default-value,redefined-outer-name,g-explicit-bool-comparison, missing-function-docstring,missing-module-docstring,raise-missing-from,g-complex-comprehension
"""python3 preparation.py --asqa ../Data/ASQA.json --setup ./setup.tsv --dst ./ready_for_drive.tsv"""
import argparse
import json

import pandas as pd


def wide_representation(key):
  """Create a wide representation of QA pairs used to construct the ASQA instance

  Args:
    key: Key of the ASQA instance

  Returns:
    Wide representation of QA pairs used to construct the ASQA instance
  """

  instance = asqa[key]
  questions, answers = [], []
  ambiguous_question = instance['ambiguous_question']

  for qa_pair in instance['qa_pairs']:
    questions.append(qa_pair['question'])
    answers.append(' | '.join(qa_pair['short_answers']))

  for _ in range(len(questions), 6):
    questions.append('NA')
    answers.append('NA')

  row = [ambiguous_question] + questions + answers

  return row


def enhance_data(raw_df):
  """Enhance the setup.tsv file with additional information necessary to build the annotation interfance.

  Args:
    raw_df: Dataframe representing the setup.tsv file (see readme)

  Returns:
    Dataframe that is used in Stage 2
  """

  df = pd.concat([
      raw_df,
      raw_df.apply(
          lambda row: wide_representation(row[0]), result_type='expand', axis=1)
  ],
                 axis=1)

  return df


def parse_args(argv=None):
  """Argument parser."""

  parser = argparse.ArgumentParser()

  parser.add_argument('--asqa', type=str, help='Path to the ASQA data.')
  parser.add_argument('--setup', type=str, help='Path to the setup file.')
  parser.add_argument('--dst', type=str, help='Path to the resulting file.')

  return parser.parse_args(argv)


if __name__ == '__main__':

  args = parse_args()

  df = pd.read_csv(args.setup, sep='\t', dtype=str)

  with open(args.asqa, 'r') as handler:
    asqa_tmp = json.load(handler)
    asqa = {}
    for split in asqa_tmp:
      for key in asqa_tmp[split]:
        asqa[key] = asqa_tmp[split][key]

  enhance_data(df).to_csv(args.dst, sep='\t', index=False)

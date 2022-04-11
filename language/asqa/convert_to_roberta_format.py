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
# pylint: disable=redefined-outer-name
"""Convert the data format to Roberta input."""
import argparse
import json
import os


def qa_format(predictions, asqa, path='.'):
  """Convert data format."""
  res = {'data': []}

  for key in predictions:
    p = predictions[key]
    for idx, qa_pair in enumerate(asqa[key]['qa_pairs']):
      question_id = key + '_' + str(idx)
      answers = {'text': qa_pair['short_answers'], 'answer_start': []}
      question = qa_pair['question']
      res['data'].append({
          'context': p,
          'id': question_id,
          'question': question,
          'answers': answers
      })

  with open(os.path.join(path, 'qa.json'), 'w') as fid:
    json.dump(res, fid)


def parse_args(argv=None):
  """Parse input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--asqa', type=str, help='Path to the ASQA data')
  parser.add_argument(
      '--split',
      type=str,
      default='dev',
      help='What data split you want to evaluate on')
  parser.add_argument('--predictions', type=str, help='Path to predictions')
  parser.add_argument(
      '--output_path', type=str, help='Path to output directory')
  # parse the arguments
  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_args()
  try:
    with open(args.asqa, 'r') as handler:
      asqa = json.load(handler)[args.split]
  except FileNotFoundError as filenotfound:
    raise ValueError('Cannot open ASQA, abort') from filenotfound
  except KeyError as keyerror:
    raise ValueError('Wrong split is provided, abort') from keyerror

  try:
    with open(args.predictions, 'r') as handler:
      predictions = json.load(handler)
  except FileNotFoundError as filenotfound:
    raise ValueError('Cannot open predictions, abort') from filenotfound

  qa_format(predictions, asqa, path=args.output_path)

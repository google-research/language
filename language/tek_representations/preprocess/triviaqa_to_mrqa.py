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
"""Create MRQA version of TriviaQA."""
from __future__ import print_function

import gzip
import json

import tensorflow.compat.v1 as tf


def shard_files(in_filename, out_file_prefix, num_shards=100):
  """Shard input file into num_shards and write to output."""
  lines = []
  with gzip.GzipFile(fileobj=tf.gfile.Open(in_filename, 'rb')) as f:
    for i, line in enumerate(f):
      lines.append(line.decode('utf-8', 'replace').strip())
  for i in range(num_shards):
    with tf.gfile.Open(
        out_file_prefix + '-' + str(i).zfill(5) + '-of-' +
        str(num_shards).zfill(5), 'w') as f:
      for line in lines[i::num_shards]:
        f.write(line + '\n')


def join_by_double_dash(qid):
  if '.txt' not in qid:
    return qid
  else:
    question_id = ''
    underscore_occ = 0
    for c in qid:
      question_id += ('--' if c == '_' and underscore_occ == 1 else c)
      underscore_occ += int(c == '_')
  return question_id


def get_ngrams(qid, n):
  return [qid[i:i + n] for i in range(len(qid))]


def find_closest(qid, lower_to_orig, n=3):
  """Hacky method for finding the closest qid."""
  candidates = [
      candidate for candidate in lower_to_orig.keys()
      if candidate.split('--')[0] == qid.split('--')[0]
  ]
  closest = ''
  qid_set = set(get_ngrams(qid, n))
  for candidate in candidates:
    if qid_set.intersection(set(get_ngrams(
        candidate, n))) > qid_set.intersection(set(get_ngrams(closest, n))):
      closest = candidate
  return closest


def recover_original_qid(qid, lower_to_orig):
  if qid in lower_to_orig:
    return lower_to_orig[qid]
  else:
    closest_qid = find_closest(qid, lower_to_orig)
    return lower_to_orig[closest_qid]


def process_triviaqa_in_squad_format(fname_prefix,
                                     out_file_prefix,
                                     triviaqa_file,
                                     num_shards=100):
  """Convert SQuAD-format TriviaQA from Clark and Gardner to MRQA."""
  datasets = []
  for fname in tf.gfile.Glob(fname_prefix):
    print(fname)
    with open(fname) as f:
      datasets += [json.loads(f.read())]
  with open(triviaqa_file) as f:
    qa_dataset = json.loads(f.read().replace('\n', '').strip())
  given_answers = {
      datum['QuestionId']: datum.get('Answer', {})
      for datum in qa_dataset['Data']
  }
  original_qids = []
  lower_to_orig = {}
  for qdata in qa_dataset['Data']:
    for doc in qdata.get('EntityPages', []) + qdata.get('SearchResults', []):
      qid = qdata['QuestionId'] + '--' + doc['Filename']
      original_qids.append(qid)
      lower_to_orig[qid.lower()] = qid

  mrqa = []
  int_id = 0
  for dataset in datasets:
    for article in dataset['data']:
      example = {}
      mrqa.append(example)
      example['qas'] = [{}]
      example_qa = example['qas'][0]
      offset = 0
      for i, paragraph_json in enumerate(article['paragraphs']):
        assert len(paragraph_json['qas']) == 1
        article_qid = paragraph_json['qas'][0]['qid']
        if i == 0:
          example['context'] = paragraph_json['context']
          example_qa['qid'] = recover_original_qid(
              join_by_double_dash(article_qid), lower_to_orig)
          example_qa['question'] = paragraph_json['qas'][0]['question']
          example_qa['int_id'] = int_id
          question_id = '_'.join(article_qid.split('_')[:2])
          example_qa['answers'] = given_answers[question_id]
          example_qa['detected_answers'] = []
        else:
          assert article_qid == paragraph_json['qas'][0]['qid'], (
              article_qid, paragraph_json['qas'][0]['qid'])
          offset = len(example['context']) + 1
          example['context'] += ' ' + paragraph_json['context']
        qa = paragraph_json['qas'][0]
        for answer in qa['answers']:
          char_spans = [[
              offset + answer['answer_start'],
              offset + answer['answer_start'] + len(answer['text']) - 1
          ]]
          example_qa['detected_answers'] += [{
              'text': answer['text'],
              'char_spans': char_spans
          }]
      int_id += 1

  for i in range(num_shards):
    with tf.gfile.Open(
        out_file_prefix + '-' + str(i).zfill(5) + '-of-' +
        str(num_shards).zfill(5), 'w') as f:
      for qjson in mrqa[i::num_shards]:
        f.write(json.dumps(qjson) + '\n')


def main():
  triviaqa_squad_format_dir = '/usr/local/google/home/mandarj/data/triviaqa_squad/'
  triviaqa_data_dir = '/usr/local/google/home/mandarj/debias/triviaqa_cp/triviaqa-rc/qa/'

  # wikipedia
  output_dir = '/usr/local/google/home/mandarj/data/triviaqa_squad/output/'
  for split in ['dev', 'test', 'train']:
    fpath = triviaqa_squad_format_dir + 'e8475trivia-qa' + split + '-bert-8.json'
    triviaqa_file_path = triviaqa_data_dir + 'wikipedia-' + split.replace(
        'test', 'test-without-answers') + '.json'
    output_prefix = output_dir + split + '/ShardedTriviaQAWikiTfIdf.jsonl'
    process_triviaqa_in_squad_format(fpath, output_prefix, triviaqa_file_path)

  # web
  output_dir = '/usr/local/google/home/mandarj/data/triviaqa_squad/output/'
  for split in ['dev', 'test', 'train']:
    fpath = triviaqa_squad_format_dir + 'webe8v3_475trivia-qa' + split + '-bert-*'
    triviaqa_file_path = triviaqa_data_dir + 'web-' + split.replace(
        'test', 'test-without-answers') + '.json'
    output_prefix = output_dir + split + '/ShardedTriviaQAWebTfIdf.jsonl'
    print(fpath)
    process_triviaqa_in_squad_format(fpath, output_prefix, triviaqa_file_path)


if __name__ == '__main__':
  main()

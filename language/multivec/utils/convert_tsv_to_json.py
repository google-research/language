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
"""convert MSMARCO tsv files to json format."""
import json

from absl import app
from absl import flags

flags.DEFINE_string('data_dir', None, 'Path to input directory.')

FLAGS = flags.FLAGS


def convert_tsv_to_json(infn, outfn, col=1, answers=None):
  """convert data format for queries and filter queries without answers."""
  data = {}
  fns = infn.split(',')
  for fn in fns:
    for line in open(fn):
      elements = line.split('\t')
      if answers:
        if elements[0] not in answers:
          continue
      if elements[0] not in data:
        data[elements[0]] = elements[col]
      else:
        data[elements[0]] = data[elements[0]] + '\n' + elements[col]

  with open(outfn, 'w') as outfile:
    json.dump(data, outfile)
  print('Processed: ' + infn)
  print('Output to: ' + outfn)
  print('Number of entries: ' + str(len(data)))
  return data


def convert_neighbor_to_json(infn, outfn):
  """convert data format for nearest neighbors.

  Args:
    infn: tsv file downloaded from MSMARCO dataset.
    outfn: path to output neighors.

  Returns:
    The output neighbor data format is as follows:
    dict(qID=[list of ranked neighbors in the format of [pID, score]])
  """
  data = {}
  fns = infn.split(',')
  for fn in fns:
    for line in open(fn):
      elements = line.split('\t')
      if elements[0] not in data:
        data[elements[0]] = []
      # no BM25 score provided in MSMARCO dataset, use 0 instead
      neighbor = [elements[1], 0]  # replace 0 with BM25 score
      # data[elements[0]] needs to be sorted by BM25 score
      data[elements[0]].append(neighbor)

  with open(outfn, 'w') as outfile:
    json.dump(data, outfile)
  print('Processed: ' + infn)
  print('Output to: ' + outfn)
  print('Number of entries: ' + str(len(data)))
  return data


def main(_):
  fn = FLAGS.data_dir + '/qrels.train.tsv,' + FLAGS.data_dir + '/qrels.dev.small.tsv'
  outfn = FLAGS.data_dir + '/answers.json'
  answers = convert_tsv_to_json(fn, outfn, col=2)
  fn = FLAGS.data_dir + '/queries.eval.small.tsv'
  outfn = FLAGS.data_dir + 'query_test.json'
  convert_tsv_to_json(fn, outfn)
  fn = FLAGS.data_dir + '/queries.train.tsv'
  outfn = FLAGS.data_dir + '/queries_train.json'
  convert_tsv_to_json(fn, outfn, answers=answers)
  fn = FLAGS.data_dir + '/queries.dev.small.tsv'
  outfn = FLAGS.data_dir + '/queries_dev.json'
  convert_tsv_to_json(fn, outfn, answers=answers)
  fn = FLAGS.data_dir + '/collection.tsv'
  outfn = FLAGS.data_dir + '/passages.json'
  convert_tsv_to_json(fn, outfn)
  fn = FLAGS.data_dir + '/collection.tsv'
  outfn = FLAGS.data_dir + '/passages.json'
  convert_tsv_to_json(fn, outfn)
  fn = FLAGS.data_dir + '/top1000.dev'
  outfn = FLAGS.data_dir + '/neighbors_example.dev.json'
  convert_neighbor_to_json(fn, outfn)


if __name__ == '__main__':
  app.run(main)

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
"""Preprocessing Query2Box dataset."""

import pickle

from absl import app
from absl import flags
import tensorflow.compat.v1.io.gfile as gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('query2box_dir', None, 'Base directory for Query2box data.')
flags.DEFINE_string('output_dir', None, 'Output directory for Query2box data.')


def generate_kb(dataset):
  """Recover KB from processed pkl file."""
  ind2rel = pickle.load(
      gfile.GFile(FLAGS.query2box_dir + '%s/ind2rel.pkl' % dataset, 'rb'))
  rel2ind = {v: k for k, v in ind2rel.items()}
  ind2ent = pickle.load(
      gfile.GFile(FLAGS.query2box_dir + '%s/ind2ent.pkl' % dataset, 'rb'))
  ent2ind = {v: k for k, v in ind2ent.items()}

  f_out = open(FLAGS.output_dir + '%s/kb.txt' % dataset, 'w')
  f_out_test = open(FLAGS.output_dir + '%s/kb_test.txt' % dataset, 'w')
  f_ind2rel_out = open(FLAGS.output_dir + '%s/ind2rel.pkl' % dataset, 'wb')
  f_rel2ind_out = open(FLAGS.output_dir + '%s/rel2ind.pkl' % dataset, 'wb')
  f_ind2ent_out = open(FLAGS.output_dir + '%s/ind2ent.pkl' % dataset, 'wb')
  f_ent2ind_out = open(FLAGS.output_dir + '%s/ent2ind.pkl' % dataset, 'wb')

  all_facts = set()
  all_facts_test = set()
  for split in ['train', 'valid', 'test']:
    data_1c_file = FLAGS.query2box_dir + '%s/%s_ans_1c.pkl' % (dataset, split)
    data_1c = pickle.load(open(data_1c_file, 'rb'))
    for query, answers in data_1c.items():
      subj, rels = query[0]
      rel = rels[0]
      for obj in answers:
        all_facts_test.add((ind2ent[subj], ind2rel[rel], ind2ent[obj]))
        if split != 'test':
          all_facts.add((ind2ent[subj], ind2rel[rel], ind2ent[obj]))

  for subj, rel, obj in all_facts:
    if not rel.endswith('_reverse'):
      f_out.write('%s\t%s\t%s\n' % (subj, rel, obj))

  for subj, rel, obj in all_facts_test:
    if not rel.endswith('_reverse'):
      f_out_test.write('%s\t%s\t%s\n' % (subj, rel, obj))

  pickle.dump(ind2rel, f_ind2rel_out)
  pickle.dump(rel2ind, f_rel2ind_out)
  pickle.dump(ind2ent, f_ind2ent_out)
  pickle.dump(ent2ind, f_ent2ind_out)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for dataset in ['NELL', 'FB15k', 'FB15k-237']:
    print('processing %s ...' % dataset)
    generate_kb(dataset)


if __name__ == '__main__':
  app.run(main)

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
"""Preprocess MetaQA dataset."""

import os


def main():
  os.mkdir('metaqa/')

  # preprocess train/dev/test data
  for num_hop in range(1, 4):
    for split in ['train', 'dev', 'test']:
      data_filename = 'MetaQA/%d-hop/vanilla/qa_%s.txt' % (num_hop, split)
      data_filename_out = 'metaqa/qa_van%d_%s.exam' % (num_hop, split)
      with open(data_filename) as f_in, open(data_filename_out, 'w') as f_out:
        for lid, line in enumerate(f_in):
          question_id = 'van_%d_%s_%d' % (num_hop, split, lid)
          question, answers = line.strip().split('\t')
          question_entity_start = question.find('[') + 1
          question_entity_end = question.find(']')
          question_entity = question[question_entity_start:question_entity_end]
          answers_list = answers.split('|')
          new_answers = ' || '.join(answers_list)
          new_line = '%s\t%s\t%s\t%s\n' % (question_id, question,
                                           question_entity, new_answers)
          f_out.write(new_line)

  # preprocess kb
  relations = set()
  with open('MetaQA/kb.txt') as f_in, open('metaqa/kb.cfacts', 'w') as f_out:
    for line in f_in:
      subj, rel, obj = line.strip().split('|')
      f_out.write('%s\t%s\t%s\n' % (rel, subj, obj))
      relations.add(rel)

  # write relations
  with open('metaqa/rels.txt', 'w') as f_out:
    for rel in relations:
      f_out.write('%s\n' % (rel))


if __name__ == '__main__':
  main()

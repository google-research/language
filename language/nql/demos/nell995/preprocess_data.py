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
"""Preprocess data for nell995 experiments."""
import os
from tqdm import tqdm

TASKS = [
    'concept_agentbelongstoorganization', 'concept_athletehomestadium',
    'concept_athleteplaysforteam', 'concept_athleteplayssport',
    'concept_organizationheadquarteredincity',
    'concept_organizationhiredperson', 'concept_personborninlocation',
    'concept_personleadsorganization', 'concept_teamplaysinleague',
    'concept_teamplayssport', 'concept_worksfor'
]


def main():
  root_dir = 'MINERVA/datasets/data_preprocessed/nell-995'
  kb = list()
  relations = set()
  print('loading nell995 kb ...')
  with open('%s/full_graph.txt' % root_dir) as f_kb:
    for line in tqdm(f_kb):
      subj, rel, obj = line.strip().split('\t')
      rel = rel.replace(':', '_')
      kb.append((subj, rel, obj))
      relations.add(rel)

  train_data, dev_data, test_data = dict(), dict(), dict()
  for split, split_data in zip(['train', 'dev', 'test'],
                               [train_data, dev_data, test_data]):
    with open('%s/%s.txt' % (root_dir, split)) as f_data:
      for lid, line in enumerate(f_data):
        subj, rel, obj = line.strip().split('\t')
        rel = rel.replace(':', '_')
        if rel not in split_data:
          split_data[rel] = list()
        data_id = 'nell995-%s-%d' % (split, lid)
        split_data[rel].append((data_id, subj, rel, obj))

  # write to folders
  os.mkdir('nell995/')
  print('dumping task data ...')
  for task in tqdm(TASKS):
    os.mkdir('nell995/%s' % task)
    # output kb
    with open('nell995/%s/kb.cfacts' % task, 'w') as f_kb_out:
      for subj, rel, obj in kb:
        if rel == task or rel == task + '_inv':
          continue
        f_kb_out.write('%s\t%s\t%s\n' % (rel, subj, obj))
    # output train/dev/test data
    for split, split_data in zip(['train', 'dev', 'test'],
                                 [train_data, dev_data, test_data]):
      with open('nell995/%s/%s.exam' % (task, split), 'w') as f_data_out:
        for data_id, subj, rel, obj in split_data[task]:
          f_data_out.write('%s\t%s\t%s\t%s\n' % (data_id, rel, subj, obj))
    # output relations
    with open('nell995/%s/rels.txt' % task, 'w') as f_rel_out:
      for rel in relations:
        if rel == task or rel == task + '_inv':
          continue
        f_rel_out.write('%s\tentity_t\tentity_t\n' % (rel))


if __name__ == '__main__':
  main()

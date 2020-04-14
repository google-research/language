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
# Lint as: python3
"""Preprocessing code for MetaQA datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags

from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('metaqa_dir', None, 'Base directory for METAQA data.')

flags.DEFINE_string('output_dir', None,
                    'Base directory to store preprocessed data.')


def add_one_fact(kb, subj, rel, obj):
  if subj not in kb:
    kb[subj] = dict()
  if rel not in kb[subj]:
    kb[subj][rel] = set()
  kb[subj][rel].add(obj)


def load_kb(kb_filename):
  """Load KB into dictionary.

  Args:
    kb_filename: kb filename

  Returns:
    A dictionary {subject: {rel: objects}}

  """
  kb = dict()
  entities = set()
  for line in open(kb_filename):
    line = line.strip()
    subj, rel, obj = line.split('|')
    entities.add(subj.lower())
    entities.add(obj.lower())
    add_one_fact(kb, subj, rel, obj)
    if rel not in ['release_year', 'in_language', 'has_genre', 'has_tags']:
      add_one_fact(kb, obj, rel + '-inv', subj)

  return kb, list(entities)


def get_topic_ent(question):
  ent_start = question.find('[') + 1
  ent_end = question.find(']')
  ent = question[ent_start:ent_end]
  return ent


def get_shortest_path(kb, topic_ent, answers, num_hop):
  """Get all shortest path from topic entity to answers.

  Args:
    kb: dictionary to store the KB {subject: {relation: objects}}
    topic_ent: topic entity
    answers: a set of answers
    num_hop: max number of hops

  Returns:
    a list of shortest paths
  """
  cur_ents = set([(topic_ent, ())])
  candidate_chains = set()
  for _ in range(num_hop):
    next_ents = set()
    for ent, prev_path in cur_ents:
      prev_path = list(prev_path)
      if ent in kb:
        for rel, objs in kb[ent].items():
          if objs & answers:
            candidate_chains.add(tuple(prev_path + [rel]))
          next_ents.update([(obj, tuple(prev_path + [rel])) for obj in objs])
    cur_ents = next_ents

  return [list(chain) for chain in candidate_chains]


def get_intermediate_entities(kb, topic_ent, chain):
  """Get all tail entities of a topic entity following a chain of relations.

  Args:
    kb: dictionary to store the KB {subject: {relation: objects}}
    topic_ent: topic entity to start with
    chain: a list of relations to follow

  Returns:
    a set of tail entities
  """
  cur_ents = set([topic_ent])
  intermediate_entities = [cur_ents]
  for rel in chain:
    next_ents = set()
    for ent in cur_ents:
      if ent in kb and rel in kb[ent]:
        objs = kb[ent][rel]
        next_ents.update(objs)
    next_ents.discard(topic_ent)
    cur_ents = next_ents
    intermediate_entities.append(cur_ents)

  return intermediate_entities


def postprocess_candidate_chains(kb, topic_ent, answers, candidate_chains,
                                 num_hop):
  """Postprocess shortest paths and keep the one that leads to answers.

  Args:
    kb: dictionary to store the KB {subject: {relation: objects}}
    topic_ent: topic entity to start with
    answers: a set of answers
    candidate_chains: all possible shortest paths
    num_hop: max number of hops

  Returns:
    the best shortest path (None if not exist)
  """
  for chain in candidate_chains:
    if num_hop == 3:
      if len(chain) < 3 or chain[1] != chain[0] + '-inv':
        continue

    intermediate_entities = get_intermediate_entities(kb, topic_ent, chain)
    if intermediate_entities[-1] == answers:
      return chain, intermediate_entities

  return None, None


def _link_entity_list(entity_list, entity2id):
  new_list = []
  for item in entity_list:
    new_list.append({
        'text': item,
        'kb_id': entity2id[item.lower()],
    })
  return new_list


def _link_question(question, entity2id):
  """Add entity links for this question."""
  question['answers'] = _link_entity_list(question['answers'], entity2id)
  question['intermediate_entities'] = [
      _link_entity_list(el, entity2id)
      for el in question['intermediate_entities']
  ]
  question['entities'] = _link_entity_list(question['question_entities'],
                                           entity2id)
  del question['question_entities']
  question['question'] = question['question'].replace(
      '__ent__', question['entities'][0]['text'])
  return question


def preprocess_metaqa(kb, entity2id, data_in_filename, data_out_filename,
                      num_hop):
  """Runner of the preprocessing code.

  Args:
    kb: kb dict
    entity2id: entity to int id
    data_in_filename: input filename
    data_out_filename: output filename
    num_hop: num hop
  """
  num_found = 0
  num_data = 0
  with open(data_in_filename) as f_in, open(data_out_filename, 'w') as f_out:
    for line in tqdm(f_in):
      num_data += 1
      line = line.strip()
      question, answers_str = line.split('\t')
      topic_ent = get_topic_ent(question)
      answers = set(answers_str.split('|'))
      candidate_chains = get_shortest_path(kb, topic_ent, answers, num_hop)

      best_chain, intermediate_entities = postprocess_candidate_chains(
          kb, topic_ent, answers, candidate_chains, num_hop)
      num_found += int(best_chain is not None)

      out_example = {
          'question': question.replace('[' + topic_ent + ']', '__ent__'),
          'answers': list(answers),
          'entities': _link_entity_list([topic_ent], entity2id),
          'intermediate_entities': [list(es) for es in intermediate_entities] \
              + [list()] * (num_hop - len(intermediate_entities)),
          'inference_chains': best_chain,
      }

      f_out.write('%s\n' % json.dumps(out_example))

  print('shortest path found: %d / %d' % (num_found, num_data))


def main(_):
  kb, entities = load_kb(os.path.join(FLAGS.metaqa_dir, 'kb.txt'))
  entity2id = {ee: ii for ii, ee in enumerate(entities)}
  with open(os.path.join(FLAGS.output_dir, 'entities.txt'), 'w') as f:
    f.write('\n'.join(entities))
  for num_hop in range(1, 4):
    for split in ['train', 'dev', 'test']:
      data_in_filename = os.path.join(
          FLAGS.metaqa_dir, '%d-hop/vanilla/qa_%s.txt' % (num_hop, split))
      data_out_filename = os.path.join(FLAGS.output_dir,
                                       '%d-hop/%s.json' % (num_hop, split))
      preprocess_metaqa(kb, entity2id, data_in_filename, data_out_filename,
                        num_hop)


if __name__ == '__main__':
  app.run(main)

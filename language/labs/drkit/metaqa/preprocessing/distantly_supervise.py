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
"""Script to align facts to their mentions in text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import random

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("paragraphs_file", None,
                    "JSON file with entity linked paragraphs.")

flags.DEFINE_string("kb_file", None, "JSON file with the kb facts.")

flags.DEFINE_string(
    "entity_file", None, "Text file with one entity per line."
    "Line number, starting from 0, gives the entity ID.")

flags.DEFINE_string(
    "train_file", None, "JSON file with train set questions. Facts part of an "
    "inference chain here will be included in pretraining.")

flags.DEFINE_string(
    "test_file", None, "JSON file with test set questions. Facts part of an "
    "inference chain here will be excluded from pretraining.")

flags.DEFINE_string("output_path", None, "Output file to store all queries to.")

flags.DEFINE_integer("num_dev_entities", 500,
                     "Number of entities to hold out for the dev set.")

# Natural language phrasing of relations and their inverses.
RELATIONS = {
    "directed_by": ["is directed by", "directed"],
    "written_by": ["is written by", "wrote"],
    "starred_actors": ["starred actors", "acted in"],
    "release_year": ["has release year"],
    "in_language": ["is in language"],
    "has_genre": ["has genre"],
    "has_imdb_rating": ["has imdb rating"],
    "has_imdb_votes": ["has imdb votes"],
    "has_tags": ["is described by", "describes"],
}

random.seed(123)


def _get_all_facts(in_file):
  """Read all facts in inference chains."""
  facts = set()
  with tf.gfile.Open(in_file) as f:
    for line in f:
      item = json.loads(line.strip())
      for ir, relation in enumerate(item["inference_chains"][0]):
        for subj in item["intermediate_entities"][ir]:
          for obj in item["intermediate_entities"][ir + 1]:
            if relation.endswith("-inv"):
              facts.add(obj["text"].lower() + "::" + relation[:-4] + "::" +
                        subj["text"].lower())
            else:
              facts.add(subj["text"].lower() + "::" + relation + "::" +
                        obj["text"].lower())
  return facts


def main(_):
  # Read entities.
  tf.logging.info("Reading entities...")
  entity2id = {}
  with tf.gfile.Open(FLAGS.entity_file, "rb") as f:
    for ii, line in enumerate(f):
      entity2id[line.strip().decode("utf-8")] = ii
  id2entity = {i: e for e, i in entity2id.items()}
  tf.logging.info("Read %d entities", len(entity2id))

  # Read test inference chains.
  tf.logging.info("Reading train and test inference chains...")
  train_facts = _get_all_facts(FLAGS.train_file)
  test_facts = _get_all_facts(FLAGS.test_file)
  facts_to_ignore = test_facts - train_facts
  tf.logging.info("Ignoring %d facts (e.g. %s)", len(facts_to_ignore),
                  ", ".join(list(facts_to_ignore)[:3]))

  # Read paragraphs and index by entities.
  tf.logging.info("Reading paragraphs...")
  with tf.gfile.Open(FLAGS.paragraphs_file) as f:
    all_paragraphs = []
    entity2para = collections.defaultdict(list)
    for line in f:
      para = json.loads(line.strip())
      for mention in para["title"]["mentions"]:
        entity2para[mention["kb_id"]].append(len(all_paragraphs))
      # Fix mention text.
      for mention in para["mentions"]:
        start = mention["start"]
        end = start + len(mention["text"])
        mention["text"] = para["context"][start:end]
      all_paragraphs.append(para)
  tf.logging.info("Read %d paragraphs about %d entities", len(all_paragraphs),
                  len(entity2para))

  # Read facts and index by subject entities.
  tf.logging.info("Reading facts...")
  subject2relation2object = collections.defaultdict(
      lambda: collections.defaultdict(list))
  num_ignored = 0
  with tf.gfile.Open(FLAGS.kb_file) as f:
    for line in f:
      subject, relation, obj = line.strip().split("|")
      str_fact = subject.lower() + "::" + relation + "::" + obj.lower()
      if str_fact in facts_to_ignore:
        num_ignored += 1
        continue
      if relation not in RELATIONS:
        tf.logging.warn("%s not in RELATIONS", relation)
        continue
      if obj.lower() not in entity2id:
        tf.logging.warn("%s (object) not in entities", obj)
        continue
      if subject.lower() not in entity2id:
        tf.logging.warn("%s (subject) not in entities", subject)
        continue
      subject_id = entity2id[subject.lower()]
      object_id = entity2id[obj.lower()]
      if subject_id not in entity2para:
        tf.logging.warn("%s (%d) not in paragraphs", subject, subject_id)
        continue
      subject2relation2object[subject_id][relation].append(object_id)
  tf.logging.info("Excluded %d facts in the test file.", num_ignored)
  tf.logging.info("Found %d matching subjects.", len(subject2relation2object))

  # Create distantly supervised queries.
  positive_qrys, random_qrys, relation_qrys, entity_qrys = [], [], [], []
  positive_map = collections.defaultdict(list)
  random_map = collections.defaultdict(list)
  relation_map = collections.defaultdict(list)
  entity_map = collections.defaultdict(list)
  all_entities = set()
  for subject_id in tqdm(subject2relation2object):
    my_para_ind = entity2para[subject_id]
    my_obj_to_id = {}
    my_subj_mentions = {}
    my_seen_relations = set()
    for para_ind in my_para_ind:
      for im, mention in enumerate(all_paragraphs[para_ind]["mentions"]):
        my_obj_to_id[mention["kb_id"]] = (para_ind, im)
        if mention["kb_id"] == subject_id:
          my_subj_mentions[para_ind] = im
    if not my_subj_mentions:
      continue
    for relation in subject2relation2object[subject_id]:
      for object_id in subject2relation2object[subject_id][relation]:
        if object_id in my_obj_to_id:
          para_ind = my_obj_to_id[object_id][0]
          if para_ind not in my_subj_mentions:
            continue
          s_mention = all_paragraphs[para_ind]["mentions"][
              my_subj_mentions[para_ind]]
          o_mention = all_paragraphs[para_ind]["mentions"][
              my_obj_to_id[object_id][1]]
          subj_obj = {
              "mentions": [s_mention],
              "mention": s_mention,
              "name": s_mention["name"],
              "aliases": {},
              "wikidata_id": str(subject_id),
          }
          obj_obj = {
              "mention": o_mention,
              "mentions": [o_mention],
              "name": o_mention["name"],
              "aliases": {
                  id2entity[obj_id]: 1
                  for obj_id in subject2relation2object[subject_id][relation]
              },
              "wikidata_id": str(object_id),
          }
          # Create positive queries.
          qry = {
              "id": str(subject_id) + "_" + relation + "_" + str(object_id),
              "context_type": "paragraph",
              "context": all_paragraphs[para_ind]["context"],
              "subject": subj_obj,
              "object": obj_obj,
              "relation": {
                  "text": [RELATIONS[relation][0]],
                  "wikidata_id": relation,
              },
              "is_impossible": False,
          }
          positive_map[qry["relation"]["wikidata_id"]].append(
              len(positive_qrys))
          positive_qrys.append(qry)
          all_entities.add(subj_obj["wikidata_id"])
          my_seen_relations.add(relation)
          if len(RELATIONS[relation]) > 1:
            qry_inv = {
                "id":
                    (str(object_id) + "_inv_" + relation + "_" + str(subject_id)
                    ),
                "context_type": "paragraph",
                "context": all_paragraphs[para_ind]["context"],
                "subject": obj_obj,
                "object": subj_obj,
                "relation": {
                    "text": [RELATIONS[relation][1]],
                    "wikidata_id": "inv_" + relation,
                },
                "is_impossible": False,
            }
            positive_map[qry_inv["relation"]["wikidata_id"]].append(
                len(positive_qrys))
            positive_qrys.append(qry_inv)
            all_entities.add(obj_obj["wikidata_id"])
            my_seen_relations.add("inv_" + relation)
          # Create negative queries.
          while True:
            random_subj = random.choice(list(subject2relation2object.keys()))
            if random_subj != subject_id:
              break
          random_para = random.choice(entity2para[random_subj])
          # Random negative.
          qry_rand = {
              "id": str(subject_id) + "_" + relation + "_None",
              "context_type": "random negative",
              "context": all_paragraphs[random_para]["context"],
              "subject": subj_obj,
              "object": None,
              "relation": {
                  "text": [RELATIONS[relation][0]],
                  "wikidata_id": relation,
              },
              "is_impossible": True,
          }
          random_map[qry_rand["relation"]["wikidata_id"]].append(
              len(random_qrys))
          random_qrys.append(qry_rand)
          if len(RELATIONS[relation]) > 1 and not any(
              str(mention["kb_id"]) == obj_obj["wikidata_id"]
              for mention in all_paragraphs[random_para]["mentions"]):
            qry_rand_inv = {
                "id": str(object_id) + "_inv_" + relation + "_None",
                "context_type": "random negative",
                "context": all_paragraphs[random_para]["context"],
                "subject": obj_obj,
                "object": None,
                "relation": {
                    "text": [RELATIONS[relation][1]],
                    "wikidata_id": "inv_" + relation,
                },
                "is_impossible": True,
            }
            random_map[qry_rand_inv["relation"]["wikidata_id"]].append(
                len(random_qrys))
            random_qrys.append(qry_rand_inv)
          # Relation negative.
          if relation in subject2relation2object[random_subj]:
            random_objs = subject2relation2object[random_subj][relation]
            rel_obj_to_id = {}
            rel_subj_mentions = {}
            for para_ind in entity2para[random_subj]:
              for im, mention in enumerate(
                  all_paragraphs[para_ind]["mentions"]):
                rel_obj_to_id[mention["kb_id"]] = (para_ind, im)
                if mention["kb_id"] == random_subj:
                  rel_subj_mentions[para_ind] = im
            if not rel_subj_mentions:
              continue
            for rel_object_id in random_objs:
              if rel_object_id in rel_obj_to_id:
                rel_para_ind = rel_obj_to_id[rel_object_id][0]
                if rel_para_ind not in rel_subj_mentions:
                  continue
                rel_s_mention = all_paragraphs[rel_para_ind]["mentions"][
                    rel_subj_mentions[rel_para_ind]]
                rel_o_mention = all_paragraphs[rel_para_ind]["mentions"][
                    rel_obj_to_id[rel_object_id][1]]
                rel_subj_obj = {
                    "mentions": [rel_s_mention],
                    "mention": rel_s_mention,
                    "name": rel_s_mention["name"],
                    "aliases": {},
                    "wikidata_id": str(rel_s_mention["kb_id"]),
                }
                rel_obj_obj = {
                    "mention": rel_o_mention,
                    "mentions": [rel_o_mention],
                    "name": rel_o_mention["name"],
                    "aliases": {},
                    "wikidata_id": str(rel_o_mention["kb_id"]),
                }
                qry_rel = {
                    "id": str(subject_id) + "_" + relation + "_relation",
                    "context_type": "relation negative",
                    "context": all_paragraphs[rel_para_ind]["context"],
                    "subject": subj_obj,
                    "object": rel_obj_obj,
                    "relation": {
                        "text": [RELATIONS[relation][0]],
                        "wikidata_id": relation,
                    },
                    "is_impossible": True,
                }
                relation_map[qry_rel["relation"]["wikidata_id"]].append(
                    len(relation_qrys))
                relation_qrys.append(qry_rel)
                if len(RELATIONS[relation]) > 1:
                  qry_rel_inv = {
                      "id": str(object_id) + "_inv_" + relation + "_relation",
                      "context_type": "relation negative",
                      "context": all_paragraphs[rel_para_ind]["context"],
                      "subject": obj_obj,
                      "object": rel_subj_obj,
                      "relation": {
                          "text": [RELATIONS[relation][1]],
                          "wikidata_id": "inv_" + relation,
                      },
                      "is_impossible": True,
                  }
                  relation_map[qry_rel_inv["relation"]["wikidata_id"]].append(
                      len(relation_qrys))
                  relation_qrys.append(qry_rel_inv)
                break
    # Entity negatives for this subject.
    if not my_seen_relations:
      continue
    para_ind = entity2para[subject_id][0]
    for relation in RELATIONS:
      if relation not in my_seen_relations:
        qry_ent = {
            "id": str(subject_id) + "_" + relation + "_entity",
            "context_type": "entity negative",
            "context": all_paragraphs[para_ind]["context"],
            "subject": subj_obj,
            "object": None,
            "relation": {
                "text": [RELATIONS[relation][0]],
                "wikidata_id": relation,
            },
            "is_impossible": True,
        }
        entity_map[qry_ent["relation"]["wikidata_id"]].append(len(entity_qrys))
        entity_qrys.append(qry_ent)
  tf.logging.info("Found %d positive, %d random, %d relation, %d entity qrys",
                  len(positive_qrys), len(random_qrys), len(relation_qrys),
                  len(entity_qrys))
  tf.logging.info(
      "Positive queries: %s",
      "\t".join(["%s:%d" % (k, len(v)) for k, v in positive_map.items()]))
  tf.logging.info(
      "Random queries: %s",
      "\t".join(["%s:%d" % (k, len(v)) for k, v in random_map.items()]))
  tf.logging.info(
      "Relation queries: %s",
      "\t".join(["%s:%d" % (k, len(v)) for k, v in relation_map.items()]))
  tf.logging.info(
      "Entity queries: %s",
      "\t".join(["%s:%d" % (k, len(v)) for k, v in entity_map.items()]))

  # Sub-sample entity negatives.
  sampled_entity_qrys = []
  for k, v in entity_map.items():
    if len(v) <= len(positive_map[k]):
      sampled_entity_qrys.extend([entity_qrys[ei] for ei in v])
    else:
      sampled_entity_qrys.extend(
          [entity_qrys[ei] for ei in random.sample(v, len(positive_map[k]))])
  tf.logging.info("%d entity queries retained after sampling.",
                  len(sampled_entity_qrys))

  # Sample dev and train entities.
  dev_entities = set(random.sample(list(all_entities), FLAGS.num_dev_entities))
  train_entities = set(all_entities) - dev_entities

  # Save pre-training queries.
  f_tr = tf.gfile.Open(os.path.join(FLAGS.output_path, "train.json"), "w")
  f_de = tf.gfile.Open(os.path.join(FLAGS.output_path, "dev.json"), "w")
  for qry in positive_qrys + relation_qrys + sampled_entity_qrys:
    qry_s = json.dumps(qry)
    if qry["subject"]["wikidata_id"] in train_entities:
      f_tr.write(qry_s + "\n")
    else:
      f_de.write(qry_s + "\n")

  # Save slot-filling queries.
  f_tr = tf.gfile.Open(os.path.join(FLAGS.output_path, "train_sf.json"), "w")
  f_de = tf.gfile.Open(os.path.join(FLAGS.output_path, "dev_sf.json"), "w")
  for qry in positive_qrys:
    # pylint: disable=g-complex-comprehension
    qry_s = json.dumps({
        "question":
            qry["subject"]["name"] + " . " + qry["relation"]["text"][0],
        "entities": [{
            "kb_id": int(qry["subject"]["wikidata_id"]),
            "text": qry["subject"]["name"],
        }],
        "answers": [{
            "kb_id": int(qry["object"]["wikidata_id"]),
            "text": qry["object"]["name"]
        }] + [{
            "kb_id": entity2id[ans.lower()],
            "text": ans
        } for ans in qry["object"]["aliases"] if ans != qry["object"]["name"]],
    })
    if qry["subject"]["wikidata_id"] in train_entities:
      f_tr.write(qry_s + "\n")
    else:
      f_de.write(qry_s + "\n")


if __name__ == "__main__":
  app.run(main)

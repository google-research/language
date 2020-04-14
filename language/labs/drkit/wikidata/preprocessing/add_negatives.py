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
"""Script to add negatives to distantly supervised data.

Also creates train and dev splits by splitting along entities.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import copy
import json
import os
import random
import re

from absl import app
from absl import flags

from tqdm import tqdm

random.seed(123)

flags.DEFINE_string("input_pattern", None,
                    "Path to read distantly annotated facts from")

flags.DEFINE_string("output_prefix", None,
                    "Path to store output JSON train and dev files to.")

flags.DEFINE_integer("num_dev_entities", 1000,
                     "Number of entities to hold out for the dev set.")

FLAGS = flags.FLAGS


def main(_):
  rel_to_items = collections.defaultdict(list)
  ent_to_items = collections.defaultdict(list)
  items = []
  context_lens = []
  num_entity_negs = 0
  with open(FLAGS.input_pattern) as f:
    for ii, line in tqdm(enumerate(f)):
      item = json.loads(line.strip())
      context_lens.append(len(item["context"].split()))
      if item["context_type"] == "entity negative":
        item["is_impossible"] = True
        num_entity_negs += 1
      else:
        item["is_impossible"] = False
        rel_to_items[item["relation"]["wikidata_id"]].append(ii)
      items.append(item)
      ent_to_items[item["subject"]["wikidata_id"]].append(ii)

  # Context length statistics
  print("Average context length = %.1f" %
        (float(sum(context_lens)) / len(context_lens)))
  sorted_lens = sorted(context_lens)
  idx50 = int(0.5 * len(context_lens))
  idx90 = int(0.9 * len(context_lens))
  idx99 = int(0.99 * len(context_lens))
  print("50 pctile: %d 90 pctile: %d 99 pctile: %d" %
        (sorted_lens[idx50], sorted_lens[idx90], sorted_lens[idx99]))

  # sample negatives
  relation_negs, random_negs = [], []
  for ii, item in enumerate(tqdm(items)):
    if item["is_impossible"]:
      continue
    # sample a relation negative
    relation = item["relation"]["wikidata_id"]
    orig_answer = item["object"]["mention"]["text"]
    if len(rel_to_items[relation]) == 1:
      continue
    random_item_idx = random.choice(rel_to_items[relation])
    random_item = copy.deepcopy(items[random_item_idx])
    if re.search(re.escape(orig_answer), random_item["context"],
                 re.IGNORECASE) is not None:
      continue
    random_item["subject"] = copy.deepcopy(item["subject"])
    random_item["is_impossible"] = True
    random_item["id"] = (
        random_item["subject"]["wikidata_id"] + "_" +
        random_item["relation"]["wikidata_id"] + "_" +
        str(random_item["object"]["wikidata_id"]))
    random_item["context_type"] = "relation negative"
    relation_negs.append(random_item)
    # sample a random negative
    orig_subject = item["subject"]["wikidata_id"]
    random_item_idx = random.randint(0, len(items) - 1)
    random_item = copy.deepcopy(items[random_item_idx])
    if random_item["subject"]["wikidata_id"] == orig_subject:
      continue
    random_item["subject"] = copy.deepcopy(item["subject"])
    random_item["relation"] = copy.deepcopy(item["relation"])
    random_item["object"] = None
    random_item["is_impossible"] = True
    random_item["id"] = (
        random_item["subject"]["wikidata_id"] + "_" +
        random_item["relation"]["wikidata_id"] + "_None")
    random_item["context_type"] = "random negative"
    random_negs.append(random_item)
  print("Found %d pos, %d entity neg, %d relation neg %d random neg" %
        (len(items), num_entity_negs, len(relation_negs), len(random_negs)))

  # merge positives and negatives
  for item in relation_negs + random_negs:
    ent_to_items[item["subject"]["wikidata_id"]].append(len(items))
    items.append(item)

  # split by entities
  all_entities = ent_to_items.keys()
  dev_entities = set(random.sample(all_entities, FLAGS.num_dev_entities))
  train_entities = set(all_entities) - dev_entities
  with open(os.path.join(FLAGS.output_prefix, "train.json"), "w") as f_tr, open(
      os.path.join(FLAGS.output_prefix, "dev.json"), "w") as f_de:
    for item in items:
      if item["subject"]["wikidata_id"] in train_entities:
        f_tr.write(json.dumps(item) + "\n")
      else:
        f_de.write(json.dumps(item) + "\n")
  with open(os.path.join(FLAGS.output_prefix, "dev_entities.json"), "w") as f_e:
    json.dump(list(dev_entities), f_e)


if __name__ == "__main__":
  app.run(main)

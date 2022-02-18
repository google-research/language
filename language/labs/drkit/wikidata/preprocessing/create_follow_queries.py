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
"""Script to create 2-hop relation following queries from 1-hop queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import random

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf
from tqdm import tqdm

MAX_IN_DEGREE = 100

FLAGS = flags.FLAGS

flags.DEFINE_string("paragraphs_file", None,
                    "File with paragraphs and mentions.")

flags.DEFINE_string("queries_file", None, "File with queries over paragraphs.")

flags.DEFINE_string("output_paragraphs_file", None,
                    "File to store paragraphs and mentions.")

flags.DEFINE_string("output_queries_file", None,
                    "File to store queries over paragraphs.")

flags.DEFINE_integer(
    "max_paragraphs", None,
    "Total number of paragraphs to keep. If the queries "
    "require more paragraphs that number will be used.")


def main(_):
  tf.logging.info("Reading queries, subjects and objects.")
  with tf.gfile.Open(FLAGS.queries_file) as f:
    queries = []
    subj_to_index = collections.defaultdict(list)
    obj_to_index = collections.defaultdict(list)
    for line in tqdm(f):
      item = json.loads(line.strip())
      if item["object"] is None:
        continue
      subj_to_index[item["subject"]["wikidata_id"]].append(len(queries))
      obj_to_index[item["object"]["wikidata_id"]].append(len(queries))
      queries.append(item)
  tf.logging.info("Read %d queries with %d subjects and %d objects.",
                  len(queries), len(subj_to_index), len(obj_to_index))

  tf.logging.info("Identifying multi-hop queries.")
  multihop_queries = []
  paragraph_ids, entity_ids = set(), set()
  for entity in tqdm(obj_to_index):
    if len(obj_to_index[entity]) > MAX_IN_DEGREE:
      continue
    if entity not in subj_to_index:
      continue
    for fact_1_idx in obj_to_index[entity]:
      fact_1 = queries[fact_1_idx]
      for fact_2_idx in subj_to_index[entity]:
        fact_2 = queries[fact_2_idx]
        if fact_1["subject"]["wikidata_id"] == fact_2["object"]["wikidata_id"]:
          continue  # circular chain
        mh_qry = {}
        mh_qry["subject"] = fact_1["subject"]
        mh_qry["object"] = fact_2["object"]
        mh_qry["relation"] = [fact_1["relation"], fact_2["relation"]]
        mh_qry["para_id"] = [fact_1["para_id"], fact_2["para_id"]]
        mh_qry["bridge"] = {}
        mh_qry["bridge"]["wikidata_id"] = fact_1["object"]["wikidata_id"]
        mh_qry["bridge"]["name"] = fact_1["object"]["name"]
        mh_qry["bridge"]["aliases"] = fact_1["object"]["aliases"]
        mh_qry["bridge"]["mention_1"] = fact_1["object"]["mention"]
        mh_qry["bridge"]["mention_2"] = fact_2["subject"]["mentions"]
        mh_qry["id"] = fact_1["id"] + "_" + fact_2["id"]
        multihop_queries.append(mh_qry)
        paragraph_ids.add(fact_1["para_id"])
        paragraph_ids.add(fact_2["para_id"])
        entity_ids.add(fact_1["subject"]["wikidata_id"])
        entity_ids.add(fact_2["subject"]["wikidata_id"])
  tf.logging.info("Found %d multi-hop queries over %d paragraphs %d entities.",
                  len(multihop_queries), len(paragraph_ids), len(entity_ids))

  with tf.gfile.Open(FLAGS.output_queries_file, "w") as f:
    for qry in multihop_queries:
      f.write(json.dumps(qry) + "\n")

  tf.logging.info("Reading paragraphs.")
  with tf.gfile.Open(FLAGS.paragraphs_file) as f, tf.gfile.Open(
      FLAGS.output_paragraphs_file, "w") as fo:
    candidate_paragraphs = []
    num_stored = 0
    for line in tqdm(f):
      item = json.loads(line.strip())
      if item["wikidata_id"] in entity_ids:
        if item["id"] in paragraph_ids:
          num_stored += 1
          fo.write(line)
        else:
          candidate_paragraphs.append(line)
    tf.logging.info("Stored required %d paragraphs.", num_stored)
    if len(paragraph_ids) < FLAGS.max_paragraphs:
      need_more = FLAGS.max_paragraphs - len(paragraph_ids)
      tf.logging.info("Sampling remaining %d from total %d candidates.",
                      need_more, len(candidate_paragraphs))
      random_paragraphs = random.sample(candidate_paragraphs, need_more)
      for para in random_paragraphs:
        fo.write(para)


if __name__ == "__main__":
  app.run(main)

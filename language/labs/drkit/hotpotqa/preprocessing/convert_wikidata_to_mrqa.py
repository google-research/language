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
"""Convert LSF questions to MRQA questions, possibly using templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import random
from absl import flags
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("in_pattern", None, "Pattern matching input files.")

flags.DEFINE_string("out_pattern", None, "Pattern matching output files.")

flags.DEFINE_string("template_file", None,
                    "Natural language templates for the relations.")


def main(_):
  if FLAGS.template_file is not None:
    with open(FLAGS.template_file) as f:
      templates = json.load(f)
  else:
    templates = {}

  def _convert_question(item, inverse=False):
    if inverse and item["subject"] is None:
      item["is_impossible"] = True
    elif not inverse and item["object"] is None:
      item["is_impossible"] = True
    else:
      item["is_impossible"] = False
    if item["subject"] is None:
      sname = ""
    elif "name" in item["subject"] and item["subject"]["name"] is not None:
      sname = item["subject"]["name"]
    else:
      sname = item["subject"]["mentions"][0]["text"]
    if item["object"] is None:
      oname = ""
    elif "name" in item["object"] and item["object"]["name"] is not None:
      oname = item["object"]["name"]
    else:
      oname = item["ojbect"]["mention"]["text"]
    if not inverse and item["relation"]["text"][0] in templates:
      ques = random.choice(templates[item["relation"]["text"][0]]).replace(
          "XXX", sname)
      relation = item["relation"]["text"][0]
    elif not inverse:
      ques = sname + " . " + item["relation"]["text"][0]
      relation = item["relation"]["text"][0]
    else:
      ques = "inverse " + item["relation"]["text"][0] + " . " + oname
      relation = "inverse " + item["relation"]["text"][0]
    if item["is_impossible"]:
      answers = []
      detected_answers = []
    elif not inverse:
      answers = ([k for k, v in item["object"]["aliases"].items()] +
                 [item["object"]["name"], item["object"]["mention"]["text"]])
      detected_answers = [{
          "char_spans": [[
              item["object"]["mention"]["start"],
              item["object"]["mention"]["start"] +
              len(item["object"]["mention"]["text"]) - 1
          ]],
          "text": item["object"]["mention"]["text"],
      }]
    else:
      answers = (
          [k for k, v in item["subject"]["aliases"].items()] +
          [item["subject"]["name"], item["subject"]["mentions"][0]["text"]])
      detected_answers = [{
          "char_spans": [[
              item["subject"]["mentions"][0]["start"],
              item["subject"]["mentions"][0]["start"] +
              len(item["subject"]["mentions"][0]["text"]) - 1
          ]],
          "text": item["subject"]["mentions"][0]["text"],
      }]
    out = {
        "context":
            item["context"],
        "qas": [{
            "qid": item["id"],
            "is_impossible": item["is_impossible"],
            "relation": relation,
            "question": ques,
            "answers": answers,
            "detected_answers": detected_answers,
        }],
    }
    return out

  for ii in range(10):
    in_file = FLAGS.in_pattern % ii
    out_file = FLAGS.out_pattern % ii
    print("Input %s Output %s" % (in_file, out_file))
    rel2items = collections.defaultdict(list)
    with open(in_file) as f, gzip.open(out_file, "wt") as fo:
      for line in tqdm(f):
        item = json.loads(line.strip())
        out = _convert_question(item)
        fo.write(json.dumps(out) + "\n")
        if not item["is_impossible"]:
          item["id"] = item["id"] + "_inv"
          out_i = _convert_question(item, inverse=True)
          fo.write(json.dumps(out_i) + "\n")
          rel2items[out["qas"][0]["relation"]].append(out)
          rel2items[out_i["qas"][0]["relation"]].append(out_i)

  out_file = FLAGS.out_pattern % -1
  print("Relation negatives output %s" % out_file)
  with gzip.open(out_file, "wt") as fo:
    for relation, items in rel2items.items():
      num_to_add = int(0.5 * len(items))
      items_to_replace = random.sample(items, num_to_add)
      print("replacing %d items for %s" % (num_to_add, relation))
      for item in items_to_replace:
        distractor = {k: v for k, v in random.choice(items).items()}
        if distractor["qas"][0]["qid"] == item["qas"][0]["qid"]:
          continue
        item["qas"][0]["qid"] = item["qas"][0]["qid"] + "_relneg"
        item["context"] = distractor["context"]
        item["qas"][0]["is_impossible"] = True
        item["qas"][0]["answers"] = []
        item["qas"][0]["detected_answers"] = []
        fo.write(json.dumps(item) + "\n")


if __name__ == "__main__":
  tf.app.run()

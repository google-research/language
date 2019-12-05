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
"""Construct auxiliary membership classification test sets (RANDOM, SHUFFLE) to check the generalization of the classifier (Section 6.1)."""
import copy
import json
import random

import tensorflow as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("membership_dev_data", None,
                    "Path containing original membership dev set data")
flags.DEFINE_string("random_membership_dev_data", None,
                    "Path containing RANDOM dataset (first auxiliary dataset)")
flags.DEFINE_string("aux_path", None,
                    "Path to dump the membership inference data")

FLAGS = flags.FLAGS


def main(_):

  with gfile.Open(FLAGS.membership_dev_data, "r") as f:
    membership_dev_data = json.loads(f.read())["data"]

  # keep the same real examples from the original held-out set
  for instance in membership_dev_data:
    for para in instance["paragraphs"]:
      para["qas"] = [qa for qa in para["qas"] if qa["class"] == "true"]

  with gfile.Open(FLAGS.random_membership_dev_data, "r") as f:
    random_dev_data = json.loads(f.read())["data"]

  # pick out all the fake examples from the RANDOM held-out set
  for instance in random_dev_data:
    for para in instance["paragraphs"]:
      para["qas"] = [qa for qa in para["qas"] if qa["class"] == "fake"]

  combined_data = random_dev_data + membership_dev_data
  random.shuffle(combined_data)

  dev_split = {"version": "1.1", "type": "dev", "data": combined_data}
  true_counter = 0
  fake_counter = 0
  for instance in dev_split["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        if qa["class"] == "true":
          true_counter += 1
        else:
          fake_counter += 1
  logging.info("True = %d / %d, Fake = %d / %d", true_counter,
               true_counter + fake_counter, fake_counter,
               true_counter + fake_counter)

  gfile.MakeDirs(FLAGS.aux_path + "/random")

  with gfile.Open(FLAGS.aux_path + "/random/dev-v1.1.json", "w") as f:
    f.write(json.dumps(dev_split))

  # take all the real examples from original held-out set and shuffle them
  shuffled_data = copy.deepcopy(membership_dev_data)
  for instance in shuffled_data:
    for para in instance["paragraphs"]:
      # shuffle both the paragraph tokens and the question tokens
      para_tokens = para["context"].split()
      random.shuffle(para_tokens)
      para["context"] = " ".join(para_tokens)
      for qa in para["qas"]:
        # preserve the question mark at the end of the questions
        qa_tokens = qa["question"].replace("?", "").split()
        random.shuffle(qa_tokens)
        qa["question"] = " ".join(qa_tokens) + "?"
        qa["class"] = "fake"
        # dummy variable which is not used during evaluation
        qa["answers"] = [{"text": para_tokens[0], "answer_start": 0}]

  combined_data = shuffled_data + membership_dev_data
  random.shuffle(combined_data)

  dev_split = {"version": "1.1", "type": "dev", "data": combined_data}
  true_counter = 0
  fake_counter = 0
  for instance in dev_split["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        if qa["class"] == "true":
          true_counter += 1
        else:
          fake_counter += 1
  logging.info("True = %d / %d, Fake = %d / %d", true_counter,
               true_counter + fake_counter, fake_counter,
               true_counter + fake_counter)

  gfile.MakeDirs(FLAGS.aux_path + "/shuffle")

  with gfile.Open(FLAGS.aux_path + "/shuffle/dev-v1.1.json", "w") as f:
    f.write(json.dumps(dev_split))


if __name__ == "__main__":
  app.run(main)

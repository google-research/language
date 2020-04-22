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
"""Construct a held-out / validation set from a large pool of WIKI / RANDOM queries ensuring there is no overlap with the train set."""
import json
import random

import numpy as np

import tensorflow.compat.v1 as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("pool_dataset", None,
                    "Large pool of queries having training set distribution.")
flags.DEFINE_string("train_dataset", None,
                    "Training set of queries used for model extraction.")
flags.DEFINE_integer("dev_dataset_size", 10570,
                     "Number of QAs in held-out set. (default: SQuAD 1.1 size")
flags.DEFINE_string("output_path", None, "Output path for the held-out set.")
flags.DEFINE_integer("random_seed", 42, "Random seed for determinism.")

FLAGS = flags.FLAGS


def main(_):
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  with gfile.Open(FLAGS.pool_dataset, "r") as f:
    pool_data = json.loads(f.read())["data"]

  with gfile.Open(FLAGS.train_dataset, "r") as f:
    train_data = json.loads(f.read())["data"]

  all_train_paras = {}

  for inst in train_data:
    for para in inst["paragraphs"]:
      all_train_paras[para["context"]] = 1

  num_dev_questions = FLAGS.dev_dataset_size

  # sanity check to verify all pool dataset question IDs are unique
  num_pool_questions = 0
  pool_qids = {}

  for inst in pool_data:
    for para in inst["paragraphs"]:
      for qa in para["qas"]:
        num_pool_questions += 1
        pool_qids[qa["id"]] = 1

  assert len(pool_qids) == num_pool_questions

  random.shuffle(pool_data)

  output_data = {"data": [], "version": FLAGS.version}

  for instance in pool_data:
    curr_instance = {"title": "Random dev data", "paragraphs": []}
    for para in instance["paragraphs"]:
      # Even if there is a paragraph overlap, do not consider it for the
      # held-out set since we want to minimize overlap
      if para["context"] in all_train_paras:
        continue
      # Assume different paragraphs have different questions
      curr_instance["paragraphs"].append(para)
      num_dev_questions = num_dev_questions - len(para["qas"])
      if num_dev_questions <= 0:
        break
    if curr_instance["paragraphs"]:
      output_data["data"].append(curr_instance)
    if num_dev_questions <= 0:
      break

  total_questions = 0
  for instance in output_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        total_questions += 1

  logging.info("Final dataset size = %d", total_questions)

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write(json.dumps(output_data))


if __name__ == "__main__":
  app.run(main)

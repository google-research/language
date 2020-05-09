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
"""Construct a dataset for membership classification (Section 6.1), used to classify out-of-distribution inputs."""
import json
import random

import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("original_train_data", None,
                    "Path containing original train data")
flags.DEFINE_string("original_dev_data", None,
                    "Path containing original dev data")
flags.DEFINE_string("attack_data", None,
                    "Path containing dataset used to extract model")
flags.DEFINE_float("train_split_size", 0.9,
                   "Split fraction of the train set in the train-dev split")
flags.DEFINE_string("output_path", None,
                    "Path to output the final membership dataset")

FLAGS = flags.FLAGS


def main(_):

  with gfile.Open(FLAGS.original_train_data, "r") as f:
    original_train_data = json.loads(f.read())

  with gfile.Open(FLAGS.original_dev_data, "r") as f:
    original_dev_data = json.loads(f.read())

  with gfile.Open(FLAGS.attack_data, "r") as f:
    attack_data = json.loads(f.read())

  # in our membership classification setting, the original train / original dev
  # set examples are "real examples" the extraction dataset examples are
  # "fake examples"
  for instance in original_train_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        qa["class"] = "true"

  for instance in original_dev_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        qa["class"] = "true"

  for instance in attack_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        qa["class"] = "fake"

  combined_data = []

  combined_data.extend(original_train_data["data"])
  combined_data.extend(original_dev_data["data"])

  random.shuffle(combined_data)

  # Take an approximately equal number of real / fake examples
  combined_data = combined_data[:len(attack_data["data"])]
  combined_data.extend(attack_data["data"])

  random.shuffle(combined_data)

  train_size = int(0.9 * len(combined_data))

  train_split = {
      "version": "1.1",
      "type": "train",
      "data": combined_data[:train_size]
  }
  dev_split = {
      "version": "1.1",
      "type": "dev",
      "data": combined_data[train_size:]
  }

  # small dev set for debugging quickly
  dev_small_split = {
      "version":
          "1.1",
      "type":
          "dev-small",
      "data": [{
          "title": "dev_test",
          "paragraphs": combined_data[train_size]["paragraphs"][0:1]
      }]
  }

  for split in [train_split, dev_split]:
    true_counter = 0
    fake_counter = 0
    for instance in split["data"]:
      for para in instance["paragraphs"]:
        for qa in para["qas"]:
          if qa["class"] == "true":
            true_counter += 1
          else:
            fake_counter += 1
    logging.info("True = %d / %d, Fake = %d / %d", true_counter,
                 true_counter + fake_counter, fake_counter,
                 true_counter + fake_counter)

  with gfile.Open(FLAGS.output_path + "/train-v1.1.json", "w") as f:
    f.write(json.dumps(train_split))

  with gfile.Open(FLAGS.output_path + "/dev-v1.1.json", "w") as f:
    f.write(json.dumps(dev_split))

  with gfile.Open(FLAGS.output_path + "/dev-small-v1.1.json", "w") as f:
    f.write(json.dumps(dev_small_split))


if __name__ == "__main__":
  app.run(main)

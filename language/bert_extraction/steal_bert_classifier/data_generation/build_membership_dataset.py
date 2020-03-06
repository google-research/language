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
"""Construct a dataset of domain membership, used to identify random inputs."""
import random

import tensorflow.compat.v1 as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("original_train_data", None,
                    "Path containing original train data")
flags.DEFINE_string("original_dev_data", None,
                    "Path containing original dev data")
flags.DEFINE_string("attack_data", None,
                    "Path containing dataset used to attack")
flags.DEFINE_float("train_split_size", 0.9,
                   "Split fraction of the training dataset")
flags.DEFINE_string("output_path", None,
                    "Path to dump the membership inference data")
flags.DEFINE_string("task_name", "mnli", "Task in consideration")

FLAGS = flags.FLAGS
relevant_headers = {"sst2": ["sentence"], "mnli": ["sentence1", "sentence2"]}


def main(_):
  task_name = FLAGS.task_name.lower()

  with gfile.Open(FLAGS.original_train_data, "r") as f:
    orig_train_data = f.read().strip().split("\n")
    orig_train_header = orig_train_data[0]
    orig_train_data = orig_train_data[1:]

  indices_orig_train_data = [
      orig_train_header.split("\t").index(x)
      for x in relevant_headers[task_name]
  ]

  with gfile.Open(FLAGS.original_dev_data, "r") as f:
    orig_dev_data = f.read().strip().split("\n")
    orig_dev_header = orig_dev_data[0]
    orig_dev_data = orig_dev_data[1:]

  indices_orig_dev_data = [
      orig_dev_header.split("\t").index(x) for x in relevant_headers[task_name]
  ]

  with gfile.Open(FLAGS.attack_data, "r") as f:
    attack_data = f.read().strip().split("\n")
    attack_data_header = attack_data[0]
    attack_data = attack_data[1:]

  indices_attack_data = [
      attack_data_header.split("\t").index(x)
      for x in relevant_headers[task_name]
  ]

  true_data_membership = []

  for point in orig_train_data:
    input_shards = [
        point.split("\t")[index] for index in indices_orig_train_data
    ]
    input_shards.append("true")
    true_data_membership.append(input_shards)

  for point in orig_dev_data:
    input_shards = [point.split("\t")[index] for index in indices_orig_dev_data]
    input_shards.append("true")
    true_data_membership.append(input_shards)

  attack_data_membership = []
  for point in attack_data:
    input_shards = [point.split("\t")[index] for index in indices_attack_data]
    input_shards.append("fake")
    attack_data_membership.append(input_shards)

  random.shuffle(true_data_membership)

  # Take an approximately equal number of true / fake examples
  if len(attack_data_membership) > len(true_data_membership):
    combined_data = true_data_membership[:len(attack_data_membership)]
    combined_data.extend(attack_data_membership)
  else:
    combined_data = attack_data_membership[:len(true_data_membership)]
    combined_data.extend(true_data_membership)

  random.shuffle(combined_data)

  train_size = int(0.9 * len(combined_data))

  train_split = combined_data[:train_size]
  dev_split = combined_data[train_size:]
  dev_small_split = combined_data[:1000]

  for split in [train_split, dev_split]:
    true_counter = 0
    fake_counter = 0
    for instance in split:
      if instance[-1] == "true":
        true_counter += 1
      else:
        fake_counter += 1

    # expected to be a 1:1 ratio of true/fake examples
    logging.info("True = %d / %d, Fake = %d / %d", true_counter,
                 true_counter + fake_counter, fake_counter,
                 true_counter + fake_counter)

  header = "index\t%s\tlabel" % ("\t".join(relevant_headers[task_name]))

  train_split = "\n".join(
      [header] +
      ["%d\t%s" % (i, "\t".join(x)) for i, x in enumerate(train_split)])

  dev_split = "\n".join(
      [header] +
      ["%d\t%s" % (i, "\t".join(x)) for i, x in enumerate(dev_split)])

  dev_small_split = "\n".join(
      [header] +
      ["%d\t%s" % (i, "\t".join(x)) for i, x in enumerate(dev_small_split)])

  with gfile.Open(FLAGS.output_path + "/train.tsv", "w") as f:
    f.write(train_split)

  with gfile.Open(FLAGS.output_path + "/dev.tsv", "w") as f:
    f.write(dev_split)

  with gfile.Open(FLAGS.output_path + "/dev_small.tsv", "w") as f:
    f.write(dev_small_split)


if __name__ == "__main__":
  app.run(main)

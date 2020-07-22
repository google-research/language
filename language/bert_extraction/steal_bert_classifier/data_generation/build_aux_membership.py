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
import random

import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("membership_dev_data", None,
                    "File with original membership classification dev data")

flags.DEFINE_string(
    "random_membership_dev_data", None,
    "membership classification dev data built from RANDOM scheme")

flags.DEFINE_string("aux_path", None,
                    "Path to output the auxiliary membership datasets")

FLAGS = flags.FLAGS


def main(_):

  with gfile.Open(FLAGS.membership_dev_data, "r") as f:
    orig_dev_data = f.read().strip().split("\n")
    orig_dev_header = orig_dev_data[0]
    orig_dev_data = orig_dev_data[1:]

  true_data_membership = []

  for point in orig_dev_data:
    if point.split("\t")[-1] == "true":
      true_data_membership.append(point.split("\t")[1:])

  random.shuffle(true_data_membership)

  combined_data = []
  # shuffle both premise and hypothesis of the original dev data to create
  # "fake" examples
  for point in true_data_membership:
    combined_data.append(point)
    premise_tokens = point[0].split()
    hypo_tokens = point[1].split()
    random.shuffle(premise_tokens)
    random.shuffle(hypo_tokens)
    fake_point = [" ".join(premise_tokens), " ".join(hypo_tokens), "fake"]
    combined_data.append(fake_point)

  random.shuffle(combined_data)

  final_split = "\n".join(
      [orig_dev_header] +
      ["%d\t%s" % (i, "\t".join(x)) for i, x in enumerate(combined_data)])

  gfile.MakeDirs(FLAGS.aux_path + "/shuffle")

  with gfile.Open(FLAGS.aux_path + "/shuffle/dev.tsv", "w") as f:
    f.write(final_split)

  with gfile.Open(FLAGS.random_membership_dev_data, "r") as f:
    random_dev_data = f.read().strip().split("\n")
    random_dev_data = random_dev_data[1:]

  fake_data_membership = []

  for point in random_dev_data:
    if point.split("\t")[-1] == "fake":
      fake_data_membership.append(point.split("\t")[1:])

  # combine the "true" examples from the original membership dev set with "fake"
  # examples from the RANDOM dev set
  combined_data = true_data_membership + fake_data_membership

  random.shuffle(combined_data)

  final_split = "\n".join(
      [orig_dev_header] +
      ["%d\t%s" % (i, "\t".join(x)) for i, x in enumerate(combined_data)])

  gfile.MakeDirs(FLAGS.aux_path + "/random")

  with gfile.Open(FLAGS.aux_path + "/random/dev.tsv", "w") as f:
    f.write(final_split)


if __name__ == "__main__":
  app.run(main)

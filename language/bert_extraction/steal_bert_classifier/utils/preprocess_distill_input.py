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
"""Combine victim model's outputs with queries to form a new training dataset for extraction."""

import numpy as np
import tensorflow as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("task_name", "sst2", "Name of task to preprocess")
flags.DEFINE_string("sents_path", None, "Path containing sentence data")
flags.DEFINE_string("probs_path", None, "Path containing probability data")
flags.DEFINE_string("output_path", None, "Output path for preprocessing")
flags.DEFINE_string("split_type", "train", "Type of preprocessing")

FLAGS = flags.FLAGS

num_labels = {"sst-2": 2, "mnli": 3}

mnli_map = {
    "contradiction": "\t1\t0\t0",
    "entailment": "\t0\t1\t0",
    "neutral": "\t0\t0\t1"
}


def main(_):
  task_name = FLAGS.task_name.lower()

  with gfile.Open(FLAGS.sents_path, "r") as f:
    sents_data = f.read().strip().split("\n")

  header = sents_data[0] + "".join(
      ["\tlabel%d_prob" % i for i in range(num_labels[task_name])])
  sents_data = sents_data[1:]

  if FLAGS.probs_path:
    with gfile.Open(FLAGS.probs_path, "r") as f:
      probs_data = f.read().strip().split("\n")
  else:
    probs_data = None

  if FLAGS.split_type == "train":
    assert len(sents_data) == len(probs_data)
    output_data = [
        x.strip() + "\t" + y.strip() for x, y in zip(sents_data, probs_data)
    ]

  elif FLAGS.split_type == "train_argmax":
    assert len(sents_data) == len(probs_data)
    # Round the probability vectors before adding them to file
    output_data = []
    for x, y, in zip(sents_data, probs_data):
      # Convert tsv probability vector to numpy style array
      prob_vector = np.array([float(yy) for yy in y.split("\t")])
      # initialize a vector with zeros
      argmax_prob_vector = np.zeros_like(prob_vector)
      # keep only the argmax prediction
      argmax_prob_vector[np.argmax(prob_vector)] = 1.0
      argmax_prob_str = "\t".join([str(yy) for yy in argmax_prob_vector])
      output_data.append(x.strip() + "\t" + argmax_prob_str.strip())

  elif FLAGS.split_type == "dev":
    if task_name == "sst-2":
      output_data = [
          x.strip() + "\t1\t0" if x.split("\t")[1] == "0" else x.strip() +
          "\t0\t1" for x in sents_data
      ]
    elif task_name == "mnli":
      output_data = [
          x.strip() + mnli_map[x.split("\t")[-1]] for x in sents_data
      ]

  output_data = [header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

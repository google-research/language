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
"""Combine victim model's outputs (adding watermarks at random) with queries to form a new training dataset for extraction.

This script also outputs the watermark details for subsequent verification.
"""
import random

import numpy as np
import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("task_name", "sst2", "Name of task to preprocess")
flags.DEFINE_string("sents_path", None, "Path containing sentence data")
flags.DEFINE_string("probs_path", None, "Path containing probability data")
flags.DEFINE_float("watermark_fraction", 0.001,
                   "Fraction of points that need to be watermarked")
flags.DEFINE_string("output_path", None, "Output path for preprocessing")
flags.DEFINE_string("watermark_path", None, "Output path for watermark")
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

  watermark_prob_str = "".join(
      ["\twatermark%d_prob" % i for i in range(num_labels[task_name])])
  original_prob_str = "".join(
      ["\toriginal%d_prob" % i for i in range(num_labels[task_name])])
  watermark_header = sents_data[0] + watermark_prob_str + original_prob_str

  sents_data = sents_data[1:]

  with gfile.Open(FLAGS.probs_path, "r") as f:
    probs_data = f.read().strip().split("\n")

  number_watermarks = int(FLAGS.watermark_fraction * len(sents_data))
  watermark_ids = list(range(len(sents_data)))
  random.shuffle(watermark_ids)
  watermark_ids = {x: 1 for x in watermark_ids[:number_watermarks]}

  if FLAGS.split_type == "train":
    assert len(sents_data) == len(probs_data)

    output_data = []
    watermark_data = []

    for i, (x, y) in enumerate(zip(sents_data, probs_data)):
      if i in watermark_ids:
        orig_prob_vector = np.array([float(yy) for yy in y.split("\t")])
        new_prob_vector = np.array([float(yy) for yy in y.split("\t")])

        while np.argmax(new_prob_vector) == np.argmax(orig_prob_vector):
          np.random.shuffle(new_prob_vector)

        # use watermarked input for the new string
        new_prob_str = "\t".join([str(yy) for yy in new_prob_vector])
        output_data.append(x.strip() + "\t" + new_prob_str.strip())

        # add the watermarked data for future checks
        watermark_data.append(x.strip() + "\t" + new_prob_str.strip() + "\t" +
                              y.strip())

      else:
        output_data.append(x.strip() + "\t" + y.strip())

  elif FLAGS.split_type == "train_argmax":
    assert len(sents_data) == len(probs_data)
    # Round the probability vectors before adding them to file
    output_data = []
    watermark_data = []

    for i, (x, y) in enumerate(zip(sents_data, probs_data)):
      # Convert tsv probability vector to numpy style array
      prob_vector = np.array([float(yy) for yy in y.split("\t")])
      # initialize a vector with zeros
      argmax_prob_vector = np.zeros_like(prob_vector)
      # keep only the argmax prediction
      argmax_prob_vector[np.argmax(prob_vector)] = 1.0
      argmax_prob_str = "\t".join([str(yy) for yy in argmax_prob_vector])

      if i in watermark_ids:
        new_prob_vector = np.copy(argmax_prob_vector)
        while np.argmax(new_prob_vector) == np.argmax(argmax_prob_vector):
          np.random.shuffle(new_prob_vector)

        # use watermarked input for the new string
        new_prob_str = "\t".join([str(yy) for yy in new_prob_vector])
        output_data.append(x.strip() + "\t" + new_prob_str.strip())

        # add the watermarked data for future checks
        watermark_data.append(x.strip() + "\t" + new_prob_str.strip() + "\t" +
                              argmax_prob_str.strip())

      else:
        output_data.append(x.strip() + "\t" + argmax_prob_str.strip())

  logging.info("Total dataset size = %d", len(output_data))
  logging.info("Total watermarked instances = %d", len(watermark_data))

  output_data = [header] + output_data
  watermark_data = [watermark_header] + watermark_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")

  with gfile.Open(FLAGS.watermark_path, "w") as f:
    f.write("\n".join(watermark_data) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

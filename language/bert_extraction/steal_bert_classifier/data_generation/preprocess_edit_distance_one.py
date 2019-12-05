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
"""Preprocess the input to include sentences with edit distance one."""
import collections
import random

import tensorflow as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("input_path", None, "Path containing sentence data")
flags.DEFINE_integer("num_pertubations", 1, "Number of random flips to be made")
flags.DEFINE_float("fraction", 1, "What fraction of data should be used")
flags.DEFINE_integer("random_seed", 42, "")
flags.DEFINE_bool("keep_only_original", False,
                  "Should only original subset be preserved")
flags.DEFINE_bool("export_other", False, "Export the other fraction of data")
flags.DEFINE_string("output_path", None, "Output path for preprocessed data")

FLAGS = flags.FLAGS


def build_vocab(sents_data):
  # find all words in corpus to build a vocabulary
  vocab = collections.defaultdict(int)
  for sent in sents_data:
    original_sent = sent.split("\t")[0].split()
    for word in original_sent:
      vocab[word] = 1
  # convert it to a list for future
  vocab = list(vocab.keys())
  return vocab


def build_subset(sents_data):
  subset_size = int(FLAGS.fraction * len(sents_data))
  random.shuffle(sents_data)
  if FLAGS.export_other:
    return sents_data[subset_size:]
  else:
    return sents_data[:subset_size]


def main(_):
  random.seed(FLAGS.random_seed)

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = f.read().strip().split("\n")

  header = sents_data[0]
  sents_data = sents_data[1:]

  vocab = build_vocab(sents_data)
  subset_sents_data = build_subset(sents_data)

  output_data = []

  for sent in subset_sents_data:
    output_data.append(sent)
    data_point_parts = sent.split("\t")
    original_sent = data_point_parts[0].split()

    if FLAGS.keep_only_original:
      continue

    # For each pertubation, construct a new sentence and randomly replace a word
    for _ in range(FLAGS.num_pertubations):
      pertubed = [x for x in original_sent]
      pertubed[random.randint(0, len(original_sent) - 1)] = random.choice(vocab)
      output_data.append(" ".join(pertubed) + " \t" +
                         "\t".join(data_point_parts[1:]))

  output_data = [header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

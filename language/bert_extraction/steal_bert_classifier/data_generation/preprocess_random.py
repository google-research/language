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
# Lint as: python3
"""Preprocess script to write random sequence training data."""
import random

from bert_extraction.steal_bert_classifier.data_generation import preprocess_util as pp_util

import tensorflow.compat.v1 as tf
import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("task_name", "sst2", "Which task's data is being used")
flags.DEFINE_string("scheme", "random", "Use random words for all occasions")
flags.DEFINE_integer("ed1_changes", 1, "Number of edit-distance once changes")
flags.DEFINE_string("input_path", None, "Path containing sentence data")
flags.DEFINE_integer("augmentations", 1, "Number of random sets to make")
flags.DEFINE_integer("random_seed", 42, "")

flags.DEFINE_integer(
    "dataset_size", None,
    "Flexibly choose dataset size. Default is input path size")

flags.DEFINE_string("lengths_scheme", "uniform_random",
                    "Scheme to adopt while sampling lengths of instances.")

flags.DEFINE_string("output_path", None, "Output path for preprocessed data")
flags.DEFINE_string("vocab_mode", "downstream_vocab",
                    "Strategy to build vocabulary")
flags.DEFINE_string("vocab_path", None, "Path to custom vocab file")
flags.DEFINE_string("thief_dataset", None,
                    "Path containing sentence data to build vocabulary out of")

FLAGS = flags.FLAGS


def main(_):
  random.seed(FLAGS.random_seed)

  task_name = FLAGS.task_name.lower()

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = f.read().strip().split("\n")

  header = sents_data[0]
  sents_data = sents_data[1:]

  if FLAGS.thief_dataset:
    with gfile.Open(FLAGS.thief_dataset, "r") as f:
      thief_data = f.read().strip().split("\n")
    vocab, probs = pp_util.build_vocab(
        sents_data=thief_data,
        task_name="list-sentences",
        vocab_mode=FLAGS.vocab_mode,
        vocab_path=FLAGS.vocab_path)
    thief_lengths_pool = pp_util.get_lengths_pool(thief_data)
  else:
    vocab, probs = pp_util.build_vocab(
        sents_data=sents_data,
        task_name=task_name,
        vocab_mode=FLAGS.vocab_mode,
        vocab_path=FLAGS.vocab_path)
    thief_lengths_pool = None

  output_data = []

  if FLAGS.dataset_size:
    points_remaining = FLAGS.dataset_size
    new_sents_data = []
    while points_remaining > len(sents_data):
      new_sents_data.extend(list(sents_data))
      points_remaining = points_remaining - len(sents_data)
    new_sents_data.extend(list(sents_data[:points_remaining]))
    sents_data = new_sents_data

  for _ in range(FLAGS.augmentations):
    for sent in tqdm.tqdm(sents_data):
      data_point_parts = sent.split("\t")

      if FLAGS.scheme.startswith("random_ed_k_"):
        # only relevant for pairwise text classification tasks
        premise_ind, hypo_ind = pp_util.task_input_indices[task_name]
        # Randomly choose premise
        original_premise = data_point_parts[premise_ind].split()

        new_len = pp_util.get_length(
            original_sequence=original_premise,
            thief_lengths_pool=thief_lengths_pool,
            lengths_scheme=FLAGS.lengths_scheme)

        # randomly sample a word for every position in the premise
        new_premise = pp_util.sample_next_sequence(
            vocab=vocab, probs=probs, seq_length=new_len, scheme=FLAGS.scheme)

        data_point_parts[premise_ind] = pp_util.detokenize(
            new_premise, FLAGS.vocab_mode)
        # Starting from premise, make multiple ed1 changes to form hypothesis
        new_premise = pp_util.token_replace(
            token_list=new_premise,
            vocab=vocab,
            probs=probs,
            num_changes=FLAGS.ed1_changes)

        data_point_parts[hypo_ind] = pp_util.detokenize(new_premise,
                                                        FLAGS.vocab_mode)

      elif FLAGS.scheme.startswith("random_"):
        # For every index having textual input, do a random replacement
        for index in pp_util.task_input_indices[task_name]:
          original_sent = data_point_parts[index].split()

          new_len = pp_util.get_length(
              original_sequence=original_sent,
              thief_lengths_pool=thief_lengths_pool,
              lengths_scheme=FLAGS.lengths_scheme)
          # randomly sample a word for every position in the premise
          new_sent = pp_util.sample_next_sequence(
              vocab=vocab, probs=probs, seq_length=new_len, scheme=FLAGS.scheme)

          data_point_parts[index] = pp_util.detokenize(new_sent,
                                                       FLAGS.vocab_mode)

      elif FLAGS.scheme.startswith("shuffle_"):
        # only relevant for pairwise text classification tasks
        premise_ind, hypo_ind = pp_util.task_input_indices[task_name]
        # Randomly choose premise
        original_premise = data_point_parts[premise_ind].split()

        # sample lengths according to a thief dataset or uniform random sampling
        new_len = pp_util.get_length(
            original_sequence=original_premise,
            thief_lengths_pool=thief_lengths_pool,
            lengths_scheme=FLAGS.lengths_scheme)

        # randomly sample a word for every position in the premise
        new_premise = pp_util.sample_next_sequence(
            vocab=vocab, probs=probs, seq_length=new_len, scheme=FLAGS.scheme)

        data_point_parts[premise_ind] = pp_util.detokenize(
            new_premise, FLAGS.vocab_mode)
        # Shuffle words for hypothesis
        random.shuffle(new_premise)

        data_point_parts[hypo_ind] = pp_util.detokenize(new_premise,
                                                        FLAGS.vocab_mode)

      # Once all sentences have been replaced, add to corpus
      output_data.append("\t".join(data_point_parts))

  output_data = [header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

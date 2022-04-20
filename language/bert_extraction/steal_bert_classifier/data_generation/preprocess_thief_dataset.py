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
"""Preprocess script to leverage thief dataset to build training data."""
import random

from bert_extraction.steal_bert_classifier.data_generation import preprocess_util as pp_util

import tensorflow.compat.v1 as tf
import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("thief_dataset", None, "File containing thief dataset")
flags.DEFINE_string("task_name", "sst-2", "Which task's data is being used")
flags.DEFINE_string("scheme", "random_uniform",
                    "Use random words for all occasions")
flags.DEFINE_integer("ed1_changes", 1, "Number of edit-distance once changes")
flags.DEFINE_string("input_path", None, "Path containing sentence data")

flags.DEFINE_integer(
    "dataset_size", None,
    "Flexibly choose dataset size. Default is input path size")

flags.DEFINE_integer("augmentations", 1, "Number of random sets to make")
flags.DEFINE_integer("random_seed", 42, "")
flags.DEFINE_string("output_path", None, "Output path for preprocessed data")
flags.DEFINE_string("vocab_mode", "full_corpus", "Strategy to build vocabulary")
flags.DEFINE_string("vocab_path", None, "Path to custom vocab file")
# In the full_corpus_top_k mode whether or not to replace OOVs with random vocab
flags.DEFINE_bool("sanitize_samples", False,
                  "Sanitize OOV words in random thief dataset samples")

FLAGS = flags.FLAGS


def main(_):
  random.seed(FLAGS.random_seed)
  task_name = FLAGS.task_name.lower()

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = f.read().strip().split("\n")

  with gfile.Open(FLAGS.thief_dataset, "r") as f:
    thief_data = f.read().strip().split("\n")

  header = sents_data[0]
  sents_data = sents_data[1:]

  vocab, probs = pp_util.build_vocab(
      thief_data,
      task_name="list-sentences",
      vocab_mode=FLAGS.vocab_mode,
      vocab_path=FLAGS.vocab_path)
  vocab_dict = {x: i for i, x in enumerate(vocab)}
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
        premise_ind, hypo_ind = pp_util.task_input_indices[task_name]
        # sample random sentence from the thief dataset
        new_premise = pp_util.sample_thief_data(
            thief_data,
            sanitize=FLAGS.sanitize_samples,
            vocab=vocab,
            vocab_dict=vocab_dict).split()
        data_point_parts[premise_ind] = pp_util.detokenize(new_premise)
        # Starting from premise, make multiple ed1 changes to form hypothesis
        new_premise = pp_util.token_replace(
            token_list=new_premise,
            vocab=vocab,
            probs=None,
            num_changes=FLAGS.ed1_changes)

        data_point_parts[hypo_ind] = pp_util.detokenize(new_premise)

      elif FLAGS.scheme.startswith("random_"):
        # For every index having textual input, do a random replacement
        for index in pp_util.task_input_indices[task_name]:
          # sample random sentence from the thief dataset
          new_sent = pp_util.sample_thief_data(
              thief_data,
              sanitize=FLAGS.sanitize_samples,
              vocab=vocab,
              vocab_dict=vocab_dict).split()
          data_point_parts[index] = pp_util.detokenize(new_sent)

      elif FLAGS.scheme.startswith("shuffle_"):
        # only a valid scheme for pairwise datasets
        premise_ind, hypo_ind = pp_util.task_input_indices[task_name]
        # sample random sentence from the thief dataset
        new_premise = pp_util.sample_thief_data(
            thief_data,
            sanitize=FLAGS.sanitize_samples,
            vocab=vocab,
            vocab_dict=vocab_dict).split()
        data_point_parts[premise_ind] = pp_util.detokenize(new_premise)
        # Shuffle words for hypothesis
        random.shuffle(new_premise)
        data_point_parts[hypo_ind] = pp_util.detokenize(new_premise)

      elif FLAGS.scheme.startswith("random_ed_all_"):
        premise_ind, hypo_ind = pp_util.task_input_indices[task_name]
        # sample random sentence from the thief dataset
        new_premise = pp_util.sample_thief_data(
            thief_data,
            sanitize=FLAGS.sanitize_samples,
            vocab=vocab,
            vocab_dict=vocab_dict).split()
        data_point_parts[premise_ind] = pp_util.detokenize(new_premise)
        # Starting from premise, make multiple ed1 changes to form hypothesis

        # First, randomly sample the type of change that needs to be made
        change_type = random.choice(["replace", "drop", "add", "random"])
        # Next, randomly sample the number of ed1 changes that need to be made
        # FLAGS.ed1_changes represents the upper-bound
        num_changes = random.choice(list(range(1, FLAGS.ed1_changes + 1)))

        if change_type == "drop" and num_changes >= len(new_premise):
          change_type = random.choice(["replace", "add"])

        if change_type == "replace":
          new_premise = pp_util.token_replace(
              token_list=new_premise,
              vocab=vocab,
              probs=probs,
              num_changes=num_changes)

        elif change_type == "drop":
          new_premise = pp_util.token_drop(
              token_list=new_premise, num_changes=num_changes)

        elif change_type == "add":
          new_premise = pp_util.token_add(
              token_list=new_premise,
              vocab=vocab,
              probs=probs,
              scheme=FLAGS.scheme,
              num_changes=num_changes)

        elif change_type == "random":
          # in the random mode, just sample another sentence from corpus
          new_premise = pp_util.sample_thief_data(
              thief_data,
              sanitize=FLAGS.sanitize_samples,
              vocab=vocab,
              vocab_dict=vocab_dict).split()

        data_point_parts[hypo_ind] = pp_util.detokenize(new_premise)

      # Once all sentences have been replaced, add to corpus
      output_data.append("\t".join(data_point_parts))

  logging.info("Final dataset size = %d", len(output_data))

  output_data = [header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

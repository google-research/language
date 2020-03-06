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
"""Build BoolQ RANDOM / WIKI splits leveraging a thief dataset (wikitext103)."""

import json
import random

from bert_extraction.steal_bert_qa.data_generation import preprocess_util as pp_util
import numpy as np

import tensorflow.compat.v1 as tf
import tqdm

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("thief_dataset", None,
                    "File containing thief dataset, typically wikitext103.")
flags.DEFINE_string("para_scheme", "original_para",
                    "Scheme used to construct the paragraphs of the dataset.")
flags.DEFINE_string("input_path", None,
                    "Path containing original training data")
flags.DEFINE_string("question_sampling_scheme", "random_postprocess",
                    "Scheme used to construct the questions of the dataset.")
flags.DEFINE_integer("augmentations", 1, "Number of dataset augmentations.")
flags.DEFINE_integer(
    "dataset_size", None,
    "Custom dataset size, typically used for smaller than 1x.")
flags.DEFINE_integer("random_seed", 42, "Random seed for determinism.")
flags.DEFINE_string("output_path", None, "Output path for final query dataset.")

FLAGS = flags.FLAGS


def main(_):
  np.random.seed(FLAGS.random_seed)

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = [json.loads(x) for x in f.read().strip().split("\n")]

  output_data = []

  if FLAGS.thief_dataset:
    with gfile.Open(FLAGS.thief_dataset, "r") as f:
      thief_paragraphs = f.read().strip().split("\n")
  else:
    thief_paragraphs = None

  # build a vocab of frequent top question words
  q_tokens, q_probs = pp_util.get_boolq_question_starters()

  thief_tokens, thief_probs, thief_lens = pp_util.build_thief_vocab(
      thief_paragraphs)

  if FLAGS.dataset_size:
    random.shuffle(sents_data)
    sents_data = sents_data[:FLAGS.dataset_size]

  for _ in range(FLAGS.augmentations):
    # Preserve the original dataset size and paragraphs, choose random questions
    for instance in tqdm.tqdm(sents_data):
      if FLAGS.para_scheme == "original_para":
        para_text = instance["passage"]
      elif FLAGS.para_scheme == "thief_para":
        para_text = random.choice(thief_paragraphs)

      elif FLAGS.para_scheme == "uniform_sampling":
        para_text = pp_util.uniform_sampling_paragraph(thief_tokens)
      elif FLAGS.para_scheme == "frequency_sampling":
        para_text = pp_util.frequency_sampling_paragraph(
            thief_tokens, thief_probs)

      elif FLAGS.para_scheme == "uniform_sampling_orig_length":
        para_text = pp_util.uniform_sampling_paragraph(
            thief_tokens, para_len=len(instance["passage"].split()))
      elif FLAGS.para_scheme == "frequency_sampling_orig_length":
        para_text = pp_util.frequency_sampling_paragraph(
            thief_tokens,
            thief_probs,
            para_len=len(instance["passage"].split()))

      elif FLAGS.para_scheme == "uniform_sampling_sample_length":
        para_text = pp_util.uniform_sampling_paragraph(
            thief_tokens, para_len=random.choice(thief_lens))
      elif FLAGS.para_scheme == "frequency_sampling_sample_length":
        para_text = pp_util.frequency_sampling_paragraph(
            thief_tokens, thief_probs, para_len=random.choice(thief_lens))

      else:
        para_text = None

      # choose a question statement. Randomly sample from a gaussian
      # distribution centered at a random anchor word
      question = pp_util.choose_random_question(para_text,
                                                FLAGS.question_sampling_scheme)
      # post-process the question to make it look more like one!
      if "postprocess" in FLAGS.question_sampling_scheme:
        question = pp_util.postprocess_question_boolq(
            question, q_tokens, q_probs, FLAGS.question_sampling_scheme)
      output_data.append({
          "title": "What is the answer to life, universe and everything?",
          "passage": para_text,
          "question": question.lower(),
          "answer": False
      })

  logging.info("Final dataset size = %d", len(output_data))

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join([json.dumps(x) for x in output_data]) + "\n")

  return


if __name__ == "__main__":
  app.run(main)

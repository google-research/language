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
"""Build SQuAD RANDOM / WIKI splits using a thief dataset like wikitext103."""
import json
import random

from bert_extraction.steal_bert_qa.data_generation import preprocess_util as pp_util
import numpy as np

import tensorflow as tf
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
flags.DEFINE_string("question_sampling_scheme", "anchor_gaussian_postprocess",
                    "Scheme used to construct the questions of the dataset.")
flags.DEFINE_integer("augmentations", 1, "Number of dataset augmentations.")
flags.DEFINE_float("fraction", None,
                   "fraction of dataset, for partial data experiments")
flags.DEFINE_integer("random_seed", 42, "")
flags.DEFINE_string("output_path", None, "Output path for preprocessed data")

FLAGS = flags.FLAGS


def main(_):
  np.random.seed(FLAGS.random_seed)

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = json.loads(f.read())

  output_data = {"data": [], "version": FLAGS.version}

  if FLAGS.thief_dataset:
    with gfile.Open(FLAGS.thief_dataset, "r") as f:
      thief_paragraphs = f.read().strip().split("\n")
  else:
    thief_paragraphs = None

  # build a vocab of frequent top question words
  q_tokens, q_probs = pp_util.get_squad_question_starters()

  thief_tokens, thief_probs, thief_lens = pp_util.build_thief_vocab(
      thief_paragraphs)
  question_ids = []
  qa_lens = []
  for instance in sents_data["data"]:
    for para in instance["paragraphs"]:
      qa_lens.append(len(para["qas"]))
      for qa in para["qas"]:
        question_ids.append(qa["id"])

  random.shuffle(question_ids)
  if FLAGS.fraction:
    num_final_questions = int(round(len(question_ids) * FLAGS.fraction))
  else:
    num_final_questions = len(question_ids)
  question_ids = {x: 1 for x in question_ids[:num_final_questions]}

  ans_not_found = 0

  # Preserve the original dataset size and paragraphs, choose random questions
  for i in range(FLAGS.augmentations):
    for instance in tqdm.tqdm(sents_data["data"]):
      instance_data = {"title": instance["title"], "paragraphs": []}
      for para in instance["paragraphs"]:

        if FLAGS.para_scheme == "original_para":
          para_text = para["context"]
        elif FLAGS.para_scheme == "thief_para":
          para_text = random.choice(thief_paragraphs)

        elif FLAGS.para_scheme == "uniform_sampling":
          para_text = pp_util.uniform_sampling_paragraph(thief_tokens)
        elif FLAGS.para_scheme == "frequency_sampling":
          para_text = pp_util.frequency_sampling_paragraph(
              thief_tokens, thief_probs)

        elif FLAGS.para_scheme == "uniform_sampling_orig_length":
          para_text = pp_util.uniform_sampling_paragraph(
              thief_tokens, para_len=len(para["context"].split()))
        elif FLAGS.para_scheme == "frequency_sampling_orig_length":
          para_text = pp_util.frequency_sampling_paragraph(
              thief_tokens, thief_probs, para_len=len(para["context"].split()))

        elif FLAGS.para_scheme == "uniform_sampling_sample_length":
          para_text = pp_util.uniform_sampling_paragraph(
              thief_tokens, para_len=random.choice(thief_lens))
        elif FLAGS.para_scheme == "frequency_sampling_sample_length":
          para_text = pp_util.frequency_sampling_paragraph(
              thief_tokens, thief_probs, para_len=random.choice(thief_lens))

        else:
          para_text = None

        para_text = para_text.strip()
        para_instance = {"context": para_text, "qas": []}

        for qa in para["qas"]:
          if FLAGS.fraction and qa["id"] not in question_ids:
            continue
          if FLAGS.question_sampling_scheme == "original_question":
            question = qa["question"]
            para_instance["qas"].append({
                "question": question,
                "id": qa["id"] + hex(i)[2:],
                "answers": [{
                    "answer_start": 0,
                    "text": para_text.split()[0]
                }],
                "is_impossible": False
            })
          else:
            # choose a question statement. Randomly sample from a gaussian
            # distribution centered at a random anchor word
            question = pp_util.choose_random_question(
                para_text, FLAGS.question_sampling_scheme)
            # post-process the question to make it look more like one!
            if "postprocess" in FLAGS.question_sampling_scheme:
              question = pp_util.postprocess_question(
                  question, q_tokens, q_probs, FLAGS.question_sampling_scheme)

            para_instance["qas"].append({
                "question": question,
                "id": qa["id"] + hex(i)[2:],
                "answers": [{
                    "answer_start": 0,
                    "text": para_text.split()[0]
                }],
                "is_impossible": False
            })

        instance_data["paragraphs"].append(para_instance)
      output_data["data"].append(instance_data)

  total_questions = 0
  for instance in output_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        total_questions += 1

  logging.info("Final dataset size = %d", total_questions)

  logging.info("Answers not found = %d", ans_not_found)

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write(json.dumps(output_data))

  return


if __name__ == "__main__":
  app.run(main)

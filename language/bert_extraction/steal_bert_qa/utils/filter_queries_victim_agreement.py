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
"""Filter pool of queries based on their predictions on various victim model.

Used in Section 5.1 of the academic paper.
"""
import json
import random

from bert_extraction.steal_bert_qa.utils import evaluate_squad
import numpy as np

import tensorflow.compat.v1 as tf
import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("pool_dataset", None,
                    "Pool of queries to sort and filter, such as 10x datasets")
flags.DEFINE_string(
    "prediction_files", None,
    "Comma-separated list of predictions from different victim seeds")
flags.DEFINE_enum("scheme", "top_f1", ["top_f1", "bottom_f1", "random"],
                  "Scheme to carry out sorting and filtering of pool dataset")
flags.DEFINE_string("output_dir", None,
                    "Output directory to store filtered dataset")
flags.DEFINE_integer("train_set_size", 87599, "Target size of filtered dataset")
FLAGS = flags.FLAGS


def main(_):

  with gfile.Open(FLAGS.pool_dataset, "r") as f:
    pool_data = json.loads(f.read())["data"]

  preds_data = []
  for pred_file in FLAGS.prediction_files.split(","):
    if pred_file.strip():
      with gfile.Open(pred_file, "r") as f:
        preds_data.append(json.loads(f.read()))

  # Calculate the pairwise F1 scores of the predicted answers across files.
  # Store the average score for sorting the list.
  qa_f1 = []
  for inst in tqdm.tqdm(pool_data):
    for para in inst["paragraphs"]:
      for qa in para["qas"]:
        f1_scores = evaluate_squad.f1_score_multiple(
            [x[qa["id"]] for x in preds_data])
        qa_f1.append([
            qa["id"], f1_scores,
            np.mean(f1_scores), qa["answers"][0],
            [x[qa["id"]] for x in preds_data]
        ])
  # Sort the pool dataset based on average pairwise F1 score.
  qa_f1.sort(key=lambda x: x[2], reverse=True)

  # Filer the dataset based on the filtering scheme.
  if FLAGS.scheme == "random":
    random.shuffle(qa_f1)
    qa_f1 = {x[0]: x[2] for x in qa_f1[:FLAGS.train_set_size]}
  elif FLAGS.scheme == "top_f1":
    qa_f1 = {x[0]: x[2] for x in qa_f1[:FLAGS.train_set_size]}
  elif FLAGS.scheme == "bottom_f1":
    qa_f1 = {x[0]: x[2] for x in qa_f1[-1 * FLAGS.train_set_size:]}
  else:
    logging.error("error")
    return

  output_data_orig = {"data": [], "version": FLAGS.version}

  # A total of len(preds_data) + 1 datasets are constructed, each with all
  # all possible answers for the filtered questions.

  # First, make a dataset with the original pool_dataset's answers.

  # Run through the pool dataset and add all those questions which survived
  # the filtering scheme.
  for inst in tqdm.tqdm(pool_data):
    inst1 = {"title": "original ans", "paragraphs": []}
    for para in inst["paragraphs"]:
      para_text = para["context"]
      para1 = {"context": para_text, "qas": []}
      for qa in para["qas"]:
        if qa["id"] not in qa_f1:
          continue
        para1["qas"].append(qa)
      # only add paragraphs with non-zero QAs.
      if para1["qas"]:
        inst1["paragraphs"].append(para1)
    # only add instances with non-zero paragraphs.
    if inst1["paragraphs"]:
      output_data_orig["data"].append(inst1)

  total_questions = 0
  for instance in output_data_orig["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        total_questions += 1
  logging.info("Orig answers dataset size = %d", total_questions)

  gfile.MakeDirs(FLAGS.output_dir + "/orig_answers")

  with gfile.Open(FLAGS.output_dir + "/orig_answers/train-v1.1.json", "w") as f:
    f.write(json.dumps(output_data_orig))

  # Next, make datasets with each of the predicted file's answers.
  # These datasets have been used in the academic publication. For schemes like
  # top_f1 there will be a lot of redundancy (and hence low variance in plots).
  for pp, pred_data1 in enumerate(preds_data):
    output_data_preds = {"data": [], "version": FLAGS.version}

    for inst in tqdm.tqdm(pool_data):
      inst1 = {"title": "pred answer %d" % pp, "paragraphs": []}
      for para in inst["paragraphs"]:
        para_text = para["context"]
        para1 = {"context": para_text, "qas": []}
        for qa in para["qas"]:
          if qa["id"] not in qa_f1:
            continue
          if pred_data1[qa["id"]] not in para_text:
            continue
          para1["qas"].append({
              "question": qa["question"],
              "id": qa["id"],
              "answers": [{
                  "answer_start": para_text.index(pred_data1[qa["id"]]),
                  "text": pred_data1[qa["id"]]
              }],
              "is_impossible": False
          })
        if para1["qas"]:
          inst1["paragraphs"].append(para1)
      if inst1["paragraphs"]:
        output_data_preds["data"].append(inst1)

    total_questions = 0
    for instance in output_data_preds["data"]:
      for para in instance["paragraphs"]:
        for qa in para["qas"]:
          total_questions += 1
    logging.info("Final prediction #%d dataset size = %d", pp, total_questions)

    gfile.MakeDirs(FLAGS.output_dir + "/pred_answer%d" % pp)

    with gfile.Open(FLAGS.output_dir + "/pred_answer%d/train-v1.1.json" % pp,
                    "w") as f:
      f.write(json.dumps(output_data_preds))

  return


if __name__ == "__main__":
  app.run(main)

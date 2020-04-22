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
"""Script to combine predicted answers with queries, while inserting a watermark (Section 6.2)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from bert_extraction.steal_bert_qa.data_generation import preprocess_util as pp_util

import tensorflow.compat.v1 as tf
import tqdm

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("questions_path", None, "Path to original questions data")
flags.DEFINE_string("predictions_path", None, "Path to predictions data")
flags.DEFINE_string("output_path", None,
                    "Path to output the final SQuAD-style dataset")
flags.DEFINE_float("watermark_fraction", 0.001,
                   "Fraction of questions that need to be watermarked")
flags.DEFINE_string("watermark_path", None,
                    "Path to output the watermarked QAs + victim answers")
flags.DEFINE_enum("watermark_length", "same_length",
                  ["same_length", "one", "two"],
                  "The length of the watermark answers")
FLAGS = flags.FLAGS


def main(_):
  with gfile.Open(FLAGS.questions_path, "r") as f:
    questions_data = json.loads(f.read())

  with gfile.Open(FLAGS.predictions_path, "r") as f:
    predictions_data = json.loads(f.read())

  counter = 0
  unanswerable = 0
  total = 0

  watermark_data = {"data": []}

  question_ids = []

  for instance in questions_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        question_ids.append(qa["id"])

  # Randomly choose the queries that will be watermarked
  random.shuffle(question_ids)
  num_watermarks = int(FLAGS.watermark_fraction * len(question_ids))
  watermark_question_ids = {x: 1 for x in question_ids[:num_watermarks]}

  for instance in tqdm.tqdm(questions_data["data"]):

    watermark_instance = {"paragraphs": []}

    for para in instance["paragraphs"]:
      para_text = para["context"]
      para_tokens = para_text.split()

      watermark_paragraph = {"context": para_text, "qas": []}

      for qa in para["qas"]:
        answer_text = predictions_data[qa["id"]]
        total += 1

        if qa["id"] in watermark_question_ids:

          if FLAGS.watermark_length == "same_length":
            wm_length = len(answer_text.split())
          elif FLAGS.watermark_length == "one":
            wm_length = 1
          elif FLAGS.watermark_length == "two":
            wm_length = 2

          ans_pos = random.randint(0, len(para_tokens) - wm_length)
          watermark_ans = " ".join(para_tokens[ans_pos:ans_pos + wm_length])

          # iterate over the watermark until you are sure it's not degenerate
          # and it has minimal f1 overlap with the victim's answer
          while (not pp_util.normalize_text(watermark_ans) or pp_util.f1_score(
              pp_util.normalize_text(answer_text),
              pp_util.normalize_text(watermark_ans)) > 0.2):
            ans_pos = random.randint(0, len(para_tokens) - wm_length)
            watermark_ans = " ".join(para_tokens[ans_pos:ans_pos + wm_length])

          # Once watermark answer has been satisfactorily constructed,
          # update the watermark information
          watermark_paragraph["qas"].append({
              "question": qa["question"],
              "id": qa["id"],
              "is_impossible": False,
              "answers": [{
                  "text": watermark_ans,
                  "answer_start": 0
              }],
              "original_answers": [{
                  "text": answer_text,
                  "answer_start": 0
              }],
          })
          answer_text = watermark_ans

        if answer_text.strip():
          qa["is_impossible"] = False
          # due to minor data processing issues, there are a few cases where the
          # predicted answer does not exist exactly in the paragraph text.
          # In this case, check if the first word of the answer is present in
          # the paragraph and approximate the answer_start using it.
          if answer_text not in para_text:
            counter += 1
            # If even the first word is not in the paragraph, ignore this QA
            if answer_text.split()[0] not in para_text:
              continue
            else:
              # approximate answer_start by the position of the first word
              qa["answers"] = [{
                  "text": answer_text,
                  "answer_start": para_text.index(answer_text.split()[0])
              }]
              continue
          # the usual case where answer_text is exactly present in para_text
          qa["answers"] = [{
              "text": answer_text,
              "answer_start": para_text.index(answer_text)
          }]

        else:
          # this code makes it compatible to SQuAD 2.0
          unanswerable += 1
          qa["answers"] = []
          qa["is_impossible"] = True

      # appending all the non-zero paragraphs
      if watermark_paragraph["qas"]:
        watermark_instance["paragraphs"].append(watermark_paragraph)

    # appending all the non-zero instances
    if watermark_instance["paragraphs"]:
      watermark_data["data"].append(watermark_instance)

  logging.info("Final watermark size = %d questions",
               len(watermark_question_ids))
  logging.info("%d / %d answers were unanswerable", unanswerable, total)
  logging.info("%d / %d answers didn't have an exact match", counter, total)

  with gfile.Open(FLAGS.watermark_path, "w") as f:
    f.write(json.dumps(watermark_data))

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write(json.dumps(questions_data))


if __name__ == "__main__":
  app.run(main)

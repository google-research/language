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
"""Preprocessing script that combines questions and answers into single file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
import tqdm

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("questions_path", None, "Path to original questions data")
flags.DEFINE_string("predictions_path", None, "Path to predictions data")
flags.DEFINE_string("output_path", None,
                    "Path to output the final SQuAD-style dataset")

FLAGS = flags.FLAGS


def main(_):
  with gfile.Open(FLAGS.questions_path, "r") as f:
    questions_data = json.loads(f.read())

  with gfile.Open(FLAGS.predictions_path, "r") as f:
    predictions_data = json.loads(f.read())

  counter = 0
  unanswerable = 0
  total = 0

  for instance in tqdm.tqdm(questions_data["data"]):

    for para in instance["paragraphs"]:
      para_text = para["context"]

      for qa in para["qas"]:
        answer_text = predictions_data[qa["id"]]
        total += 1

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

  logging.info("%d / %d answers were unanswerable", unanswerable, total)
  logging.info("%d / %d answers didn't have an exact match", counter, total)

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write(json.dumps(questions_data))


if __name__ == "__main__":
  app.run(main)

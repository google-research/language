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
"""Preprocessing script to randomly select a fraction of a SQuAD dataset."""
import json
import random

import tensorflow.compat.v1 as tf
import tqdm

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("input_path", None,
                    "Path containing the SQuAD training data.")
flags.DEFINE_integer("random_seed", 42, "Random seed for determinism.")
flags.DEFINE_float("fraction", 1.0, "Fraction of dataset to preserve.")
flags.DEFINE_string("output_path", None, "Output path for smaller new dataset.")

FLAGS = flags.FLAGS


def main(_):
  random.seed(FLAGS.random_seed)

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = json.loads(f.read())

  output_data = {"data": [], "version": FLAGS.version}

  # Find all the question IDs in the SQuAD dataset
  question_ids = []
  for instance in sents_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        question_ids.append(qa["id"])

  # Randomly shuffle the question IDs, and choose FLAGS.fraction percent of them
  random.shuffle(question_ids)
  num_final_questions = int(round(len(question_ids) * FLAGS.fraction))
  question_ids = {x: 1 for x in question_ids[:num_final_questions]}

  # Preserve the original dataset size and paragraphs, choose random questions
  # based on the question IDs which survived the filtering.
  for instance in tqdm.tqdm(sents_data["data"]):
    instance_data = {"title": instance["title"], "paragraphs": []}
    for para in instance["paragraphs"]:
      para_instance = {"context": para["context"], "qas": []}
      for qa in para["qas"]:
        # Only choose those questions which survived the filtering.
        if qa["id"] in question_ids:
          para_instance["qas"].append(qa)
      # Don't append paras with no QAs
      if para_instance["qas"]:
        instance_data["paragraphs"].append(para_instance)
    # Don't append instances with no paragraphs.
    if instance_data["paragraphs"]:
      output_data["data"].append(instance_data)

  # Count the total number of questions in the final, smaller dataset.
  total_questions = 0
  for instance in output_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        total_questions += 1

  logging.info("Final dataset size = %d", total_questions)

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write(json.dumps(output_data))

  return


if __name__ == "__main__":
  app.run(main)

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
r"""Script for computing accuracy at pairwise prediction for eval metric.

Basic Usage:
  python wikibio_pairwise_accuracy.py \
    --data_dir=<path_to_data_files>

<path_to_data_dir> must contain the CSV files from the EWOK annotation, with the
raw data in JSON format. Details are in `prepare_ewok_data_for_release.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import json
import logging
import os
from absl import app
from absl import flags
from language.table_text_eval import table_text_eval
import numpy as np
import tensorflow as tf
from tqdm import tqdm


np.random.seed(0)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
                    "Base directory containing all the required files.")

LAMBDAS = np.arange(0.1, 1.0, 0.1)


def _tokenize_table(table):
  """Tokenize fields and values in table."""
  return [(field.split(), value.split()) for field, value in table]


def main(_):
  # Filenames.
  raw_data_json = os.path.join(FLAGS.data_dir, "generations.json")
  input_csv = os.path.join(FLAGS.data_dir, "inputs.csv")
  annotation_csv = os.path.join(FLAGS.data_dir, "annotations.csv")
  key_csv = os.path.join(FLAGS.data_dir, "keys.csv")

  # Read raw data
  with tf.gfile.Open(raw_data_json, "r") as f:
    eval_data = json.load(f)

  # Read the item descriptions and their keys.
  item_to_keys = {}
  uniq_keys = set()
  f_in = tf.gfile.Open(input_csv, "r")
  f_key = tf.gfile.Open(key_csv, "r")
  input_reader = csv.reader(f_in)
  input_headers = input_reader.next()
  description_index = input_headers.index("a.description")

  # pylint: disable=g-complex-comprehension
  for data, key in zip(input_reader, f_key):
    item_to_keys[data[description_index]] = [
        [k.split(",") for k in kk.split("..")]
        for kk in key.strip().split("\t")]
    uniq_keys.update(
        [k for key in item_to_keys[data[description_index]]
         for kk in key
         for k in kk])

  uniq_keys = list(uniq_keys)

  # Read annotations.
  item_to_annotations = {}
  with tf.gfile.Open(annotation_csv, "r") as f:
    reader = csv.reader(f)
    headers = reader.next()
    description_index = headers.index("i.description")
    status_index = headers.index("t.status")
    annotation_indices = []
    for i, header in enumerate(headers):
      if "t.s.pair_sentence_selection__sentence_" in header:
        annotation_indices.append(i)
    assert len(annotation_indices) == len(item_to_keys.itervalues().next())
    for row in reader:
      if row[status_index] != "Completed":
        continue
      if row[description_index] not in item_to_keys:
        continue
      item_to_annotations[row[description_index]] = [
          int(row[ii]) for ii in annotation_indices]

  # Collect sentences and references for each key.
  all_sentences = {k: [] for k in uniq_keys}
  all_references = []
  all_tables_tokenized = []
  for n in range(len(eval_data)):
    for key in uniq_keys:
      if key == "reference":
        continue
      all_sentences[key].append(eval_data[n][key].split())
    all_references.append([eval_data[n]["reference"].split()])
    all_tables_tokenized.append(_tokenize_table(eval_data[n]["table"]))

  # Compute PARENT scores for each lambda.
  logging.info("Computing PARENT scores for each system.")
  all_parent_scores = {k: {lbd: [] for lbd in LAMBDAS} for k in uniq_keys}
  for key in uniq_keys:
    if key == "reference":
      continue
    for lbd in LAMBDAS:
      logging.info("System %s Lambda %.1f", key, lbd)
      _, _, _, parent_scores = table_text_eval.parent(
          all_sentences[key],
          all_references,
          all_tables_tokenized,
          lambda_weight=lbd)
      all_parent_scores[key][lbd] = parent_scores
  logging.info("Done.")

  # Compute accuracy of each metric.
  metrics = ["parent-%.1f" % lbd for lbd in LAMBDAS]
  accuracy = {m: 0. for m in metrics}
  total = 0
  for item in tqdm(range(len(eval_data))):
    if str(item) not in item_to_annotations:
      continue
    annotations = item_to_annotations[str(item)]
    list_of_key_pairs = item_to_keys[str(item)]
    for ii, key_pairs in enumerate(list_of_key_pairs):
      annotation = annotations[ii]
      key_pair = key_pairs[0]
      if "reference" in key_pair:
        continue

      # Compute metrics.
      scores = {}
      # PARENT.
      for lbd in LAMBDAS:
        scores["parent-%.1f" % lbd] = [
            all_parent_scores[key_pair[0]][lbd][item],
            all_parent_scores[key_pair[1]][lbd][item]]

      # Accuracies.
      predictions = {}
      for metric in scores:
        pred = 0 if scores[metric][0] >= scores[metric][1] else 1
        predictions[metric] = pred
        if pred == annotation:
          accuracy[metric] += 1.

      total += 1

  print("Accuracies")
  for metric in metrics:
    print(metric + "," + str(accuracy[metric] / total))


if __name__ == "__main__":
  app.run(main)

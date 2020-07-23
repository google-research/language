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
"""Combine two datasets based on a pool-based active learning criteria.

This assumes an extracted model (extracted) is trained on a base_dataset, and is
used for filtering a large pool dataset for the next step of active learning.

For more details look at https://arxiv.org/abs/1905.09165.
"""
import numpy as np

from scipy import stats

import tensorflow.compat.v1 as tf
from tqdm import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("base_dataset", None,
                    "Dataset used for training the extracted model initially")
flags.DEFINE_string("input_path_victim", None,
                    "Path with victim outputs on the pool dataset")
flags.DEFINE_string("input_path_extracted", None,
                    "Path with extracted outputs on the pool dataset")
flags.DEFINE_string("filter_criteria", "top_diff",
                    "Scheme to use while filtering pool dataset")
flags.DEFINE_string("filter_size", "base_dataset",
                    "Number of elements to keep after filtering pool dataset")
flags.DEFINE_string("output_path", None,
                    "New output directory where output corpus will be dumped")
flags.DEFINE_string("task_name", "sst2", "Task in consideration")
flags.DEFINE_bool("ignore_base", False, "Ignore the base dataset.")

FLAGS = flags.FLAGS

num_labels = {"sst2": 2, "mnli": 3}
relevant_headers = {"sst2": ["sentence"], "mnli": ["sentence1", "sentence2"]}


def main(_):
  task_name = FLAGS.task_name.lower()

  output_data = []

  with gfile.Open(FLAGS.base_dataset, "r") as f:
    base_dataset = f.read().strip().split("\n")
    base_dataset_header = base_dataset[0]
    base_dataset = base_dataset[1:]

  indices_base_dataset = [
      base_dataset_header.split("\t").index(x)
      for x in relevant_headers[task_name]
  ]

  if not FLAGS.ignore_base:
    for i, point in enumerate(base_dataset):
      input_shards = [
          point.split("\t")[index] for index in indices_base_dataset
      ]
      output_data.append(("%d\t" % len(output_data)) + "\t".join(input_shards))

  with gfile.Open(FLAGS.input_path_victim, "r") as f:
    sents_data_victim = f.read().strip().split("\n")
    sents_data_victim = sents_data_victim[1:]

  with gfile.Open(FLAGS.input_path_extracted, "r") as f:
    sents_data_extracted = f.read().strip().split("\n")
    sents_data_extracted_header = sents_data_extracted[0]
    sents_data_extracted = sents_data_extracted[1:]

  assert len(sents_data_victim) == len(sents_data_extracted)

  l2_distances = []
  entropies_victim = []
  entropies_extracted = []
  margins_extracted = []

  for x1, x2 in tqdm(zip(sents_data_victim, sents_data_extracted)):

    probs_victim = (x1.split("\t"))[-num_labels[task_name]:]
    probs_victim = np.array([float(x) for x in probs_victim])

    probs_extracted = (x2.split("\t"))[-num_labels[task_name]:]
    probs_extracted = np.array([float(x) for x in probs_extracted])

    l2_distances.append(np.linalg.norm(probs_victim - probs_extracted))
    entropies_victim.append(stats.entropy(probs_victim))
    entropies_extracted.append(stats.entropy(probs_extracted))

    sorted_probs_extracted = np.sort(probs_extracted)[::-1]
    margins_extracted.append(sorted_probs_extracted[0] -
                             sorted_probs_extracted[1])

  if FLAGS.filter_size == "base_dataset":
    filter_size = len(base_dataset)
  elif FLAGS.filter_size.startswith("absolute_"):
    filter_size = int(FLAGS.filter_size[len("absolute_"):])
  elif FLAGS.filter_size == "all":
    filter_size = len(sents_data_extracted)
  else:
    logging.info("Filter size not found!")
    filter_size = None

  if FLAGS.filter_criteria == "top_diff":
    top_data_indices = {
        x: 1 for x in np.argsort(l2_distances)[-1 * filter_size:]
    }
  elif FLAGS.filter_criteria == "top_extracted_entropy":
    top_data_indices = {
        x: 1 for x in np.argsort(entropies_extracted)[-1 * filter_size:]
    }
  elif FLAGS.filter_criteria == "top_victim_entropy":
    top_data_indices = {
        x: 1 for x in np.argsort(entropies_victim)[-1 * filter_size:]
    }
  elif FLAGS.filter_criteria == "min_extracted_margin":
    top_data_indices = {
        x: 1 for x in np.argsort(margins_extracted)[:filter_size]
    }
  elif FLAGS.filter_criteria == "none":
    top_data_indices = {x: 1 for x in range(len(entropies_extracted))}
  else:
    logging.info("Filtering criteria not found!")
    top_data_indices = None

  indices_aux_dataset = [
      sents_data_extracted_header.split("\t").index(x)
      for x in relevant_headers[task_name]
  ]
  for i, point in enumerate(sents_data_extracted):
    if i not in top_data_indices:
      continue
    input_shards = [point.split("\t")[index] for index in indices_aux_dataset]
    output_data.append(("%d\t" % len(output_data)) + "\t".join(input_shards))

  logging.info("Base dataset size = %d", len(base_dataset))
  logging.info("Augmented dataset size = %d", len(top_data_indices))
  logging.info("Final dataset size = %d", len(output_data))

  final_header = "index\t" + "\t".join(relevant_headers[task_name])
  output_data = [final_header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")


if __name__ == "__main__":
  app.run(main)

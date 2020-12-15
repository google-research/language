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
"""Utilities for computing accuracy metrics on retrieved neighbors."""
import collections
import json
import os

from absl import flags
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow.compat.v1 as tf

flags.DEFINE_integer("max_examples", 1000, "Max num of candidates in records.")

FLAGS = flags.FLAGS

RawResult = collections.namedtuple(
    "RawResult", ["q_id", "cand_num", "prob", "label_id", "prior_score"])


def results_file_outputs(output_dir, global_step, eval_name):
  return os.path.join(output_dir, eval_name + "_" + str(global_step) + ".tsv")


def results_file_metrics(output_dir, global_step, eval_name):
  return os.path.join(output_dir,
                      eval_name + "_" + str(global_step) + "_metrics.tsv")


def rr(y_true):
  """Return the reciprical rank of the first positive candidate."""
  for index in range(0, len(y_true)):
    if y_true[index]:
      return 1.0 / (index + 1)
  return 0.0


def bounded_rr(y_true, k):
  rr_full = rr(y_true)
  if rr_full >= 1 / k:
    return rr_full
  return 0.0


def rank_metrics(raw_results_list,
                 output_dir,
                 global_step,
                 eval_name,
                 save_raw=True):
  """Computes recall at N and other metrics for the ranking model."""
  to_return = {}
  predictions_per_query = collections.OrderedDict()
  n_queries = 0
  max_examples = FLAGS.max_examples
  total_reciprocal_rank = 0.0
  total_average_precision = 0.0
  for elem in raw_results_list:
    if elem.q_id not in predictions_per_query:
      predictions_per_query[elem.q_id] = []
    predictions_per_query[elem.q_id].append(elem)

  correct_top = np.zeros(max_examples)

  mrr_s = {}
  map_s = {}
  ks_of_interest = [10, 20, 100, 200, 1000]
  for k in ks_of_interest:
    mrr_s[k] = 0
    map_s[k] = 0

  for _, pred_q in predictions_per_query.items():
    sorted_pred = sorted(pred_q, key=lambda x: (x.prob), reverse=True)
    unique_sorted_pred = []
    set_cands = set()
    for cand in sorted_pred:
      if cand.cand_num in set_cands:
        continue
      set_cands.add(cand.cand_num)
      unique_sorted_pred.append(cand)

    sorted_pred = unique_sorted_pred
    cands = [x.cand_num for x in pred_q]
    s_cands = set(cands)
    if len(unique_sorted_pred) < len(s_cands):
      print("Multiple passages from document.")

    if len(s_cands) > max_examples:
      print("Too many examples!")

    correct_example_index = max_examples

    for i in range(len(sorted_pred)):
      if sorted_pred[i].label_id == 1:
        correct_example_index = i
        break
    # collect labels and scores for average precision
    y_true = []
    y_scores = []
    for elem in sorted_pred:
      y_true.append(elem.label_id)
      y_scores.append(elem.prob)

    mrr_increment = rr(y_true)
    precision_increment = 0.0
    if mrr_increment > 0:
      precision_increment = average_precision_score(y_true, y_scores)
    total_reciprocal_rank += mrr_increment
    total_average_precision += precision_increment

    # get all mrr and map quantities of interest
    for k in ks_of_interest:
      mrr_k = bounded_rr(y_true, k)
      mrr_s[k] += mrr_k
      if mrr_k > 0:
        map_s[k] += average_precision(y_true, y_scores, k)

    j = correct_example_index
    n_queries += 1
    while j < max_examples:
      correct_top[j] += 1
      j += 1

  for i in range(max_examples):
    correct_top[i] = correct_top[i] / n_queries

  print("Queries: %d", n_queries)
  print(correct_top)

  to_return["recall_at_1"] = correct_top[0]
  to_return["recall_at_3"] = correct_top[2]
  to_return["recall_at_5"] = correct_top[4]
  to_return["recall_at_10"] = correct_top[9]
  to_return["recall_at_100"] = correct_top[99]
  to_return["recall_at_200"] = correct_top[199]
  to_return["recall_at_" + str(max_examples)] = correct_top[max_examples - 1]

  to_return["map"] = total_average_precision / n_queries
  to_return["mrr"] = total_reciprocal_rank / n_queries
  for k in ks_of_interest:
    to_return["mrr_" + str(k)] = mrr_s[k] / n_queries
    to_return["map_" + str(k)] = map_s[k] / n_queries

  out_predictions = results_file_outputs(output_dir, global_step, eval_name)
  out_metrics = results_file_metrics(output_dir, global_step, eval_name)
  if save_raw:
    write_predictions(raw_results_list, out_predictions)
  write_metrics(to_return, out_metrics)
  print(to_return)
  print("max examples " + str(max_examples))
  return to_return


def write_predictions(raw_results_list, out_file):
  """Writing out the predictions to a file."""
  print("writing predictions to " + out_file)
  with tf.io.gfile.GFile(out_file, "w") as writer:
    writer.write("QueryID\tDocId\tModelScore\tLabel\tPriorScore\n")
    for raw_result in raw_results_list:
      writer.write(
          str(raw_result.q_id) + "\t" + str(raw_result.cand_num) + "\t" +
          str(raw_result.prob) + "\t" + str(raw_result.label_id) + "\t" +
          str(raw_result.prior_score) + "\n")


def average_precision(y_true, y_scores, k):
  rr_k = bounded_rr(y_true, k)
  if rr_k > 0:
    return average_precision_score(y_true[:k], y_scores[:k])
  else:
    return 0.0


def mean_squared_error(pred_list):
  to_return = 0.0
  for elem in pred_list:
    diff = elem.prob - elem.prior_score
    if diff < 0:
      diff = -diff
    to_return += diff
  return to_return / len(pred_list)


def write_metrics(results, out_file):
  """Writing out metrics to a file."""
  print("writing stats to " + out_file)
  with tf.io.gfile.GFile(out_file, "w") as writer:
    writer.write(json.dumps(results, indent=2, sort_keys=True) + "\n")

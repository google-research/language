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
r"""Script to compute correlations of eval metric for WebNLG data.

Basic Usage:
  python webnlg_correlations.py \
    --data_dir=<data_dir>

<data_dir> must contain a file webnlg_submissions.json.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import random
from absl import app
from absl import flags
from language.table_text_eval import table_text_eval
import nltk
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
from tqdm import tqdm


random.seed(0)
np.random.seed(0)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
                    "Directory containing the JSON file with the data. Output "
                    "correlations will also be stored here.")

flags.DEFINE_integer("num_bootstrap", 500,
                     "Number of bootstrap iterations.")

LAMBDAS = np.arange(0.1, 1.0, 0.1)


def _table(table):
  """Convert table to field, value format."""
  def _tokenize(x):
    return nltk.word_tokenize(" ".join(x.lower().split("_")))
  return [([relation], _tokenize(head) + _tokenize(value))
          for (head, relation, value) in table]


def _text(x):
  """Lowercase and tokenize text."""
  return nltk.word_tokenize(x.lower())


def main(_):
  keys_to_exclude = ["reference"]
  input_json = os.path.join(FLAGS.data_dir, "webnlg_submissions.json")
  save_output = os.path.join(FLAGS.data_dir, "correlations.json")

  # Read raw data
  with tf.gfile.Open(input_json, "r") as f:
    eval_data = json.load(f)
  uniq_keys = set([k[:-5] for k in eval_data[0] if k.endswith("-pred")])
  uniq_keys.add("reference")
  uniq_keys = list(uniq_keys)

  # Compute scores for each lambda.
  # pylint: disable=g-complex-comprehension
  logging.info("Computing scores for each system.")
  all_parent_scores = {k: {lbd: [] for lbd in LAMBDAS} for k in uniq_keys}
  for key in uniq_keys:
    if key in keys_to_exclude:
      continue
    sentences = [_text(item[key + "-pred"]) for item in eval_data]
    references = [[_text(reference) for reference in item["references"]]
                  for item in eval_data]
    tables = [_table(item["table"]) for item in eval_data]
    for lbd in LAMBDAS:
      logging.info("System %s Lambda %.1f", key, lbd)
      _, _, _, parent_scores = table_text_eval.parent(
          sentences, references, tables, lambda_weight=lbd)
      all_parent_scores[key][lbd] = parent_scores
  logging.info("Done.")

  # Bootstrap sampling.
  metrics = ["grammar", "fluency", "semantics"] + [
      "parent-%.1f" % lbd for lbd in LAMBDAS]
  human_metrics = ["grammar", "fluency", "semantics"]
  metric_to_scores = {m: {k: [] for k in uniq_keys} for m in metrics}
  metric_to_correlations = {m: {m_: [] for m_ in metrics} for m in metrics}
  for m in metrics:
    metric_to_correlations[m]["average"] = []

  for _ in tqdm(range(FLAGS.num_bootstrap)):

    # Get the bootstrap sample based on the eval_subset.
    all_keys = range(len(eval_data))
    bootstrap_sample = [
        random.choice(all_keys) for _ in range(len(eval_data))]

    # Compute average scores available.
    key_to_grammar = {k: [] for k in uniq_keys}
    key_to_fluency = {k: [] for k in uniq_keys}
    key_to_semantics = {k: [] for k in uniq_keys}
    key_to_sentences = {k: [] for k in uniq_keys}
    key_to_references = {k: [] for k in uniq_keys}
    for ii in bootstrap_sample:
      for k in uniq_keys:
        if k in keys_to_exclude:
          continue
        key_to_grammar[k].append(float(eval_data[ii][k + "-grammar"]))
        key_to_fluency[k].append(float(eval_data[ii][k + "-fluency"]))
        key_to_semantics[k].append(float(eval_data[ii][k + "-semantics"]))
        key_to_sentences[k].append(_text(eval_data[ii][k + "-pred"]))
        key_to_references[k].append(
            [_text(reference) for reference in eval_data[ii]["references"]])
    key_to_parent = {
        k: {lbd: [all_parent_scores[k][lbd][n] for n in bootstrap_sample]
            for lbd in LAMBDAS}
        for k in uniq_keys if k not in keys_to_exclude}

    # Compute average scores.
    for k in uniq_keys:
      if k in keys_to_exclude:
        continue
      metric_to_scores["grammar"][k].append(sum(key_to_grammar[k]) /
                                            len(key_to_grammar[k]))
      metric_to_scores["fluency"][k].append(sum(key_to_fluency[k]) /
                                            len(key_to_fluency[k]))
      metric_to_scores["semantics"][k].append(sum(key_to_semantics[k]) /
                                              len(key_to_semantics[k]))
      # PARENT.
      for lbd in LAMBDAS:
        metric_to_scores["parent-%.1f" % lbd][k].append(
            np.mean(key_to_parent[k][lbd]))

    # Correlations.
    for m1 in metrics:
      scores_1 = [
          metric_to_scores[m1][k][-1]
          for k in uniq_keys if k not in keys_to_exclude]
      for m2 in metrics:
        scores_2 = [
            metric_to_scores[m2][k][-1]
            for k in uniq_keys if k not in keys_to_exclude]
        metric_to_correlations[m1][m2].append(pearsonr(scores_1, scores_2)[0])
      metric_to_correlations[m1]["average"].append(
          sum([metric_to_correlations[m1][m2][-1] for m2 in human_metrics]) / 3)

  # Mean and 95% CI for each model on each metric.
  all_models = [k for k in uniq_keys if k not in keys_to_exclude]
  print("Model," + ",".join(metrics))
  for model in all_models:
    means = []
    for metric in metrics:
      scores = sorted(metric_to_scores[metric][model])
      means.append(np.mean(scores))
    print(model + "," + ",".join(
        "%.3f" % means[ii] for ii in range(len(means))))

  # Average correlation and 95% CI for each metric's correlation.
  human_metrics += ["average"]
  print("Correlations," + ",".join(human_metrics))
  for metric in metric_to_correlations:
    corrs = []
    for hm in human_metrics:
      scores = sorted(metric_to_correlations[metric][hm])
      mean = np.mean(scores)
      corrs.append(mean)
    print(metric + "," +
          ",".join("%.3f" % mean for mean in corrs))

  # Save correlations to JSON.
  json.dump(
      {m: {m_: str(v_) for m_, v_ in v.iteritems()}
       for m, v in metric_to_correlations.iteritems()},
      tf.gfile.Open(save_output, "w"))


if __name__ == "__main__":
  app.run(main)

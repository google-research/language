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
r"""Script to compute correlations of eval metric for each bootstrap sample.

Basic Usage:
  python compute_correlations.py \
    --bootstrap_file=<bootstrap_file> \
    --data_file=<data_file> \
    --save_output=<output_file>

<data_file> contains all the references, tables and generations in JSON
format.

<bootstrap_file> is a JSON file with a list of dicts, corresponding to the
bootstrap samples. Each dict lists the ids into the generations above for that
particular sample, and the human_eval scores for each model on that bootstrap
sample.

<output_file>.correlations.json and <output_file>.scores.json will hold the
correlations and PARENT scores for all the bootstrap samples, respectively.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
from absl import app
from absl import flags
from language.table_text_eval import table_text_eval
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("data_file", None,
                    "JSON file containing all the references, tables and "
                    "generations.")

flags.DEFINE_string("bootstrap_file", None,
                    "JSON file listing bootstrap sample IDs and their "
                    "corresponding human eval scores.")

flags.DEFINE_string("save_output", None,
                    "File to save correlations to.")

LAMBDAS = np.arange(0.1, 1.0, 0.1)


def main(_):
  # Read the data.
  with tf.gfile.Open(FLAGS.data_file, "r") as f:
    raw_data = json.load(f)
  with tf.gfile.Open(FLAGS.bootstrap_file, "r") as f:
    bootstrap = json.load(f)

  uniq_keys = raw_data["all_sentences"].keys()

  # Compute PARENT scores for each lambda.
  # pylint: disable=g-complex-comprehension
  logging.info("Computing PARENT scores for each system.")
  all_parent_scores = {k: {lbd: [] for lbd in LAMBDAS} for k in uniq_keys}
  for key in uniq_keys:
    if key == "reference":
      continue
    for lbd in LAMBDAS:
      logging.info("System %s Lambda %.1f", key, lbd)
      _, _, _, parent_scores = table_text_eval.parent(
          raw_data["all_sentences"][key],
          raw_data["all_references"],
          raw_data["all_tables_tokenized"],
          lambda_weight=lbd)
      all_parent_scores[key][lbd] = parent_scores
  logging.info("Done.")

  # Correlations for each bootstrap sample.
  metrics = ["human"] + ["parent-%.1f" % lbd for lbd in LAMBDAS]
  metric_to_scores = {m: {k: [] for k in uniq_keys} for m in metrics}
  metric_to_correlations = {m: {m_: [] for m_ in metrics} for m in metrics}
  for ii in range(len(bootstrap)):
    bootstrap_sample = bootstrap[ii]["ids"]
    quality_scores = bootstrap[ii]["human_eval"]
    key_to_parent = {
        k: {lbd: [all_parent_scores[k][lbd][n] for n in bootstrap_sample]
            for lbd in LAMBDAS}
        for k in uniq_keys if k != "reference"}

    # Scores.
    for k in uniq_keys:
      if k == "reference":
        continue
      for lbd in LAMBDAS:
        metric_to_scores["parent-%.1f" % lbd][k].append(
            np.mean(key_to_parent[k][lbd]))
      metric_to_scores["human"][k].append(quality_scores[k])

    # Correlations.
    for m1 in metrics:
      scores_1 = [
          metric_to_scores[m1][k][-1] for k in uniq_keys if k != "reference"]
      for m2 in metrics:
        scores_2 = [
            metric_to_scores[m2][k][-1] for k in uniq_keys if k != "reference"]
        metric_to_correlations[m1][m2].append(pearsonr(scores_1, scores_2)[0])

  # Mean for each model on each metric.
  all_models = [k for k in uniq_keys if k != "reference"]
  print("Model," + ",".join(metrics))
  for model in all_models:
    means = []
    for metric in metrics:
      scores = sorted(metric_to_scores[metric][model])
      means.append(np.mean(scores))
    print(model + "," +
          ",".join("%.3f" % means[ii] for ii in range(len(means))))

  # Average correlation and std for each metric's correlation.
  print("Correlations")
  for metric in metric_to_correlations:
    scores = sorted(metric_to_correlations[metric]["human"])
    mean = np.mean(scores)
    std = np.std(scores)
    print(metric + "," + "%.3f,%.3f" % (mean, std))

  # Save correlations to JSON.
  json.dump(
      {m: {m_: str(v_) for m_, v_ in v.iteritems()}
       for m, v in metric_to_correlations.iteritems()},
      tf.gfile.Open(FLAGS.save_output + ".correlations.json", "w"))
  json.dump(
      {m: {m_: str(v_) for m_, v_ in v.iteritems()}
       for m, v in metric_to_scores.iteritems()},
      tf.gfile.Open(FLAGS.save_output + ".scores.json", "w"))


if __name__ == "__main__":
  app.run(main)

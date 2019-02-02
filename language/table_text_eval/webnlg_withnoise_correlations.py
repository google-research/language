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
r"""Script to compute correlations of eval metric for WebNLG data with noise.

Basic Usage:
  python webnlg_withnoise_correlations.py \
    --data_dir=<data_dir> \
    --noise_type=<add/remove>

<data_dir> must contain a file webnlg_submissions.json.

<noise_type> decides whether tokens are randomly added or removed from the
references.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
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

flags.DEFINE_integer("num_bootstrap", 25,
                     "Number of bootstrap iterations.")

flags.DEFINE_string("noise_type", "add",
                    "Type of noise to to corrupt references with. Can be "
                    "either `add` or `remove`.")

NOISE = np.linspace(0.0, 1.0, 10, endpoint=False)


def _table(table):
  """Convert table to field, value format."""
  def _tokenize(x):
    return nltk.word_tokenize(" ".join(x.lower().split("_")))
  return [([relation], _tokenize(head) + _tokenize(value))
          for (head, relation, value) in table]


def _text(x):
  """Lowercase and tokenize text."""
  return nltk.word_tokenize(x.lower())


def _remove_noise(x, p):
  """Remove tokens with probability p."""
  out = []
  for token in x:
    if np.random.binomial(1, p) == 1:
      continue
    out.append(token)
  return out


def _add_noise(x, p, vocab):
  """Add random tokens with probability p."""
  out = []
  for token in x:
    out.append(token)
    if np.random.binomial(1, p) == 1:
      out.append(random.choice(vocab))
  return out


def main(_):
  keys_to_exclude = ["reference"]
  input_json = os.path.join(FLAGS.data_dir, "webnlg_submissions.json")
  save_output = os.path.join(FLAGS.data_dir,
                             "noise_%s_correlations.json" % FLAGS.noise_type)

  # Read raw data
  with tf.gfile.Open(input_json, "r") as f:
    eval_data = json.load(f)
  uniq_keys = set([k[:-5] for k in eval_data[0] if k.endswith("-pred")])
  uniq_keys.add("reference")
  uniq_keys = list(uniq_keys)

  # Get full vocabulary.
  vocab = set()
  for ii in range(len(eval_data)):
    vocab.update(_text(eval_data[ii]["references"][0]))
  vocab = list(vocab)

  # Bootstrap sampling.
  metrics = ["grammar", "fluency", "semantics", "parent-0.5"]
  human_metrics = ["grammar", "fluency", "semantics"]
  means = {k: [] for k in ["parent-0.5"]}
  stds = {k: [] for k in ["parent-0.5"]}
  all_correlations = []

  # pylint: disable=g-complex-comprehension
  for noise_level in tqdm(NOISE):
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
      key_to_tables_tokenized = {k: [] for k in uniq_keys}
      for ii in bootstrap_sample:
        for k in uniq_keys:
          if k in keys_to_exclude:
            continue
          key_to_grammar[k].append(float(eval_data[ii][k + "-grammar"]))
          key_to_fluency[k].append(float(eval_data[ii][k + "-fluency"]))
          key_to_semantics[k].append(float(eval_data[ii][k + "-semantics"]))
          key_to_sentences[k].append(_text(eval_data[ii][k + "-pred"]))
          if FLAGS.noise_type == "add":
            key_to_references[k].append(
                [_add_noise(_text(reference), noise_level, vocab)
                 for reference in eval_data[ii]["references"]])
          else:
            key_to_references[k].append(
                [_remove_noise(_text(reference), noise_level)
                 for reference in eval_data[ii]["references"]])
          key_to_tables_tokenized[k].append(_table(eval_data[ii]["table"]))

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
        _, _, parent_score, _ = table_text_eval.parent(
            key_to_sentences[k], key_to_references[k],
            key_to_tables_tokenized[k])
        metric_to_scores["parent-0.5"][k].append(parent_score)

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
            sum([metric_to_correlations[m1][m2][-1]
                 for m2 in human_metrics]) / 3)

    # Average correlation and 95% CI for each metric's correlation.
    for metric in ["parent-0.5"]:
      scores = metric_to_correlations[metric]["average"]
      means[metric].append(np.mean(scores))
      stds[metric].append(np.std(scores))

    all_correlations.append(
        {m: {m_: [str(vv) for vv in v_] for m_, v_ in v.iteritems()}
         for m, v in metric_to_correlations.iteritems()})

  print(means)
  print(stds)

  # Save correlations to JSON.
  json.dump(all_correlations, tf.gfile.Open(save_output, "w"))


if __name__ == "__main__":
  app.run(main)

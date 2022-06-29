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
r"""Generative metrics."""

import re

from language.fruit import rendering_utils
# Note: requires https://github.com/google-research/google-research/tree/master/rouge  pylint: disable=line-too-long
from rouge_score import rouge_scorer
from rouge_score import scoring
import seqio


def edit_rouge(targets, predictions):
  """Measures a variety of different ROUGE scores."""
  # We do not measure ROUGE-L for updates since LCS is likely entirely contained
  # in source.
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"])
  aggregator = scoring.BootstrapAggregator()

  for prediction, target in zip(predictions, targets):

    all_scores = {}

    target_additions = rendering_utils.extract_additions(
        source=target["normalized_inputs"],
        target=target["normalized_targets"],
    )
    target_additions = " ".join(target_additions)
    prediction_additions = rendering_utils.extract_additions(
        source=target["normalized_inputs"],
        target=prediction["normalized_targets"],
    )
    prediction_additions = " ".join(prediction_additions)

    addition_scores = scorer.score(
        target=target_additions,
        prediction=prediction_additions,
    )

    if target_additions.strip() or prediction_additions.strip():
      all_scores.update({f"update_{k}": v for k, v in addition_scores.items()})
    else:
      all_scores.update(
          {f"update_{k}": 100.0 for k, _ in addition_scores.items()})

    aggregator.add_scores(all_scores)

  result = aggregator.aggregate()
  return {key: value.mid.fmeasure * 100 for key, value in result.items()}


def exact_match(targets, predictions):
  """Measures exact match of targets and predictions."""
  numerator = 0
  denominator = 0
  for target, prediction in zip(targets, predictions):
    if target["normalized_targets"] == prediction["normalized_targets"]:
      numerator += 1
    denominator += 1
  return {"exact_match": numerator / (denominator + 1e-13)}


def surface_recall(targets, predictions):
  """Measures recall of generatable surfaces in predictions."""
  numerator = 0.0
  filtered_denominator = 0.0
  denominator = 0.0
  for target, prediction in zip(targets, predictions):
    input_text = target["inputs"]
    predicted_text = prediction["targets"]
    generatable_surfaces = target["generatable_surfaces"]
    for surfaces in generatable_surfaces.values():
      filtered = False  # True if surface is in evidence.
      appears = False
      for surface in surfaces:
        if surface in input_text:
          filtered = True
        if surface in predicted_text:
          appears = True
      denominator += 1.0
      filtered_denominator += 1.0 if filtered else 0.0
      numerator += 1.0 if appears and filtered else 0.0
  recall = numerator / (denominator + 1e-13)
  filtered_recall = numerator / (filtered_denominator + 1e-13)
  return {"surface_recall": recall, "filtered_surface_recall": filtered_recall}


def delimiter_f1(targets, predictions, delimiter_range_pair):
  """Measures p/r/f1 of predicted delimiters."""
  out = {}
  for key, delimiter in [
      ("reference", delimiter_range_pair.evidence_delimiter_range),
      ("retention", delimiter_range_pair.sentence_delimiter_range),
  ]:
    tp = 0
    fp = 0
    fn = 0
    for target, prediction in zip(targets, predictions):
      # Extract sets of delimiters
      ground_truth = set(delimiter.finduids(target["targets"]))
      predicted = set(delimiter.finduids(prediction["targets"]))
      # Update counts
      tp += len(ground_truth & predicted)
      fp += len(predicted - ground_truth)
      fn += len(ground_truth - predicted)
    p = tp / (tp + fp + 1e-13)
    r = tp / (tp + fn + 1e-13)
    f1 = 2 * p * r / (p + r + 1e-13)
    out[f"{key}_p"] = p
    out[f"{key}_r"] = r
    out[f"{key}_f1"] = f1
  return out


SAMPLE_LENGTH = 100
RE_MARKDOWN_SPECIAL_CHARS = re.compile(r"[\\`*_{}\[\]()#+\-.!]")
TEMPLATE = (
    "## Example {index}\n\n**Input**\n\n{input}\n\n**Target**\n\n{target}\n\n"
    "**Prediction**\n\n{prediction}\n\n")


def _escape_md(string):
  """Escapes special markdown characters in a string."""
  last_start = 0
  chunks = []
  for match in RE_MARKDOWN_SPECIAL_CHARS.finditer(string):
    chunks.append(string[last_start:match.start()])
    last_start = match.start()
  chunks.append(string[last_start:])
  return "\\".join(chunks)


def print_predictions(targets, predictions):
  """Prints out a sample of correct and incorrect predictions."""
  correct = []
  incorrect = []
  for target, prediction in zip(targets, predictions):
    if (target["targets"] == prediction["targets"] and
        len(correct) < SAMPLE_LENGTH):
      correct.append(
          TEMPLATE.format(
              index=len(correct),
              input=_escape_md(target["inputs"]),
              target=_escape_md(target["targets"]),
              prediction=_escape_md(prediction["targets"]),
          ))
    elif len(incorrect) < SAMPLE_LENGTH:
      incorrect.append(
          TEMPLATE.format(
              index=len(incorrect),
              input=_escape_md(target["inputs"]),
              target=_escape_md(target["targets"]),
              prediction=_escape_md(prediction["targets"]),
          ))
  correct_text = seqio.metrics.Text(textdata="\n".join(correct))
  incorrect_text = seqio.metrics.Text(textdata="\n".join(incorrect))
  return {
      "correct_predictions": correct_text,
      "incorrect_predictions": incorrect_text
  }

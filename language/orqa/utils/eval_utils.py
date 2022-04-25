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
"""Evaluation utilities."""
import json
import re
import string

import unicodedata

import tensorflow.compat.v1 as tf


def normalize_answer(s):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, ground_truth):
  try:
    regex = re.compile(
        ground_truth, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    return regex.match(prediction) is not None
  except re.error:
    return False


def metric_max_over_ground_truths(metric_fn, prediction,
                                  ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def is_correct(answers, prediction,
               is_regex):
  if is_regex:
    metric_fn = regex_match_score
  else:
    metric_fn = exact_match_score
  return metric_max_over_ground_truths(
      metric_fn=metric_fn, prediction=prediction, ground_truths=answers)


def evaluate_predictions_impl(references,
                              predictions,
                              is_regex):
  """Calculates and returns metrics."""
  missing_predictions = 0
  correct = 0
  for q, a in references.items():
    if q in predictions:
      correct += int(
          is_correct(answers=a, prediction=predictions[q], is_regex=is_regex))
    else:
      missing_predictions += 1

  return dict(
      missing_predictions=missing_predictions,
      num_correct=correct,
      num_total=len(references),
      accuracy=correct / float(len(references)))


def evaluate_predictions(
    references_path,
    predictions_path,
    is_regex,
    answer_field = "answer"):
  """Calculates and returns metrics."""
  if is_regex != ("CuratedTrec" in references_path):
    print("Warning: regex evaluation should (only) be applied to CuratedTrec.")

  references = {}
  with tf.io.gfile.GFile(references_path) as f:
    for line in f:
      example = json.loads(line)
      references[example["question"]] = example[answer_field]
  print("Found {} references in {}".format(len(references), references_path))

  predictions = {}
  with tf.io.gfile.GFile(predictions_path) as f:
    for line in f:
      example = json.loads(line)
      predictions[example["question"]] = example["prediction"]
  print("Found {} predictions in {}".format(len(predictions), predictions_path))

  return evaluate_predictions_impl(
      references=references, predictions=predictions, is_regex=is_regex)

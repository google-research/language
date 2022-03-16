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
"""Utilities for model evaluation."""

from absl import logging
from language.compgen.csl.model.inference import inference_parser
from language.compgen.csl.qcfg import qcfg_parser


def eval_model(wrapper, examples, fallback_predictions, verbose=False):
  """Compute counts on examples."""
  # Initialize stats.
  num_examples = 0
  num_correct = 0
  num_covered = 0
  num_can_parse = 0
  num_fallback_correct = 0
  num_hybrid_correct = 0

  predictions = []
  for idx, (example, fallback_prediction) in enumerate(
      zip(examples, fallback_predictions)):
    if verbose:
      logging.info("Processing example %s.", idx)
      logging.info("(%s, %s)", example[0], example[1])

    num_examples += 1
    source = example[0]
    gold_target = example[1]

    if qcfg_parser.can_parse(source, gold_target, wrapper.rules):
      num_can_parse += 1
    elif verbose:
      logging.info("Output set does not contain gold target.")

    predicted_target = inference_parser.get_top_output(source, wrapper)
    if verbose:
      logging.info("Predicted target: %s", predicted_target)

    if predicted_target:
      num_covered += 1
    if predicted_target == gold_target:
      num_correct += 1
      num_hybrid_correct += 1
    elif verbose:
      logging.info("Incorrect prediction.")
    predictions.append(predicted_target)

    if fallback_prediction == gold_target:
      num_fallback_correct += 1
      if not predicted_target:
        num_hybrid_correct += 1

  counts = {
      "num_examples": num_examples,
      "num_correct": num_correct,
      "num_covered": num_covered,
      "num_can_parse": num_can_parse,
      "num_fallback_correct": num_fallback_correct,
      "num_hybrid_correct": num_hybrid_correct,
  }
  return counts, predictions


def compute_metrics(counts):
  """Compute metrics using counts."""

  num_examples = counts["num_examples"]
  num_correct = counts["num_correct"]
  num_covered = counts["num_covered"]
  num_can_parse = counts["num_can_parse"]
  num_fallback_correct = counts["num_fallback_correct"]
  num_hybrid_correct = counts["num_hybrid_correct"]

  metrics_dict = {
      "coverage": num_covered / num_examples,
      "accuracy": num_correct / num_examples,
      "precision": num_correct / max(num_covered, 1),
      "can_parse": num_can_parse / num_examples,
      "hybrid_accuracy": num_hybrid_correct / num_examples,
      "fallback_accuracy": num_fallback_correct / num_examples,
  }
  return metrics_dict

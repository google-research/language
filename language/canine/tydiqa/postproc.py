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
"""Postprocesses outputs from a NN compute graph into well-formed answers.

This module has only light dependencies on Tensorflow (5-10 lines in
`compute_pred_dict` and `compute_predictions`).
"""

import collections
import json

from absl import logging
from language.canine.tydiqa import data
import numpy as np
import tensorflow.compat.v1 as tf


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.minimal_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def read_candidates_from_one_split(file_obj):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  for line in file_obj:
    json_dict = json.loads(line)
    candidates_dict[
        json_dict["example_id"]] = json_dict["passage_answer_candidates"]
  return candidates_dict


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


# IMPROVE ME (PULL REQUESTS WELCOME): This takes more than half the runtime and
# just runs on CPU; we could speed this up by parallelizing it (or moving it to
# Apache Beam).
def compute_predictions(eval_example, candidate_beam, max_answer_length):
  """Converts an eval_example into a `ScoreSummary` object for evaluation.

  This performs python post-processing (after running NN graph in tensorflow)
  in order to get the best answer.

  Args:
    eval_example: `EvalExample` instance with features, results.
    candidate_beam: see FLAGS.candidate_beam.
    max_answer_length: see FLAGS.max_answer_length.

  Returns:
    A `ScoreSummary` or `None` if no passage prediction could be found.
  """

  predictions = []
  n_best_size = candidate_beam

  if not eval_example.results:
    return None
  if len(eval_example.features) != len(eval_example.results):
    logging.warning(
        "ERROR: len(features)=%s, but len(results)=%d for eval_example %s",
        len(eval_example.features), len(eval_example.results),
        eval_example.example_id)
    return None

  for unique_id, result in eval_example.results.items():
    if unique_id not in eval_example.features:
      logging.warning("No feature found with unique_id: %s", unique_id)
      return None

    wp_start_offset = (
        eval_example.features[unique_id]["wp_start_offset"].int64_list.value)
    wp_end_offset = (
        eval_example.features[unique_id]["wp_end_offset"].int64_list.value)
    language_id = (
        eval_example.features[unique_id]["language_id"].int64_list.value[0])
    language_name = data.Language(language_id).name.lower()
    start_indexes = get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = get_best_indexes(result["end_logits"], n_best_size)
    cls_token_score = result["start_logits"][0] + result["end_logits"][0]
    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        # This means these are dummy tokens (like separators).
        if wp_start_offset[start_index] == -1:
          continue
        if wp_end_offset[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        summary = ScoreSummary()
        summary.minimal_span_score = (
            result["start_logits"][start_index] +
            result["end_logits"][end_index])
        summary.cls_token_score = cls_token_score
        summary.answer_type_logits = result["answer_type_logits"]

        start_offset = wp_start_offset[start_index]
        end_offset = wp_end_offset[end_index] + 1

        # Span logits minus the [CLS] logits seems to be close to the best.
        score = summary.minimal_span_score - summary.cls_token_score
        predictions.append(
            (score, summary, language_name, start_offset, end_offset))

  if not predictions:
    logging.warning("No predictions for eval_example %s",
                    eval_example.example_id)
    return None

  score, summary, language_name, start_span, end_span = max(
      predictions, key=lambda x: x[0])
  minimal_span = Span(start_span, end_span)
  passage_span_index = 0
  for c_ind, c in enumerate(eval_example.candidates):
    start = minimal_span.start_byte_offset
    end = minimal_span.end_byte_offset
    if c["plaintext_start_byte"] <= start and c["plaintext_end_byte"] >= end:
      passage_span_index = c_ind
      break
  else:
    logging.warning("No passage predicted for eval_example %s. Choosing first.",
                    eval_example.example_id)
  summary.predicted_label = {
      "example_id": eval_example.example_id,
      "language": language_name,
      "passage_answer_index": passage_span_index,
      "passage_answer_score": score,
      "minimal_answer": {
          "start_byte_offset": minimal_span.start_byte_offset,
          "end_byte_offset": minimal_span.end_byte_offset
      },
      "minimal_answer_score": score,
      "yes_no_answer": "NONE"
  }
  return summary


Span = collections.namedtuple("Span", ["start_byte_offset", "end_byte_offset"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


# IMPROVE ME: This function and its children takes more than half the processing
# time and it's entirely outside the tf graph. We should take advantage of the
# fact that this is embarrassingly parallel and run in on many CPU threads.
# Pull requests welcome.
def compute_pred_dict(candidates_dict, dev_features, raw_results,
                      candidate_beam, max_answer_length):
  """Computes official answer key from raw logits.

  This function joins three pieces needed for eval script for each example,
  based on the unique_id:

  1. Examples, which come from the original JSON definition of the dataset;
    each has a unique `example_id`.
  2. Features, which are the windowed sequences of wordpieces given to the
    neural network graph (and may be smaller than a single passage);
    each has a `unique_id`.
  3. Raw results, which are the predictions coming from the execution of the
    neural network graph; each has a `unique_id`.

  Because of the way `unique_ids` are assigned by `CreateTFExampleFn`,
  all `unique_ids` associated with an `example_id` should numerically be sorted
  after that `example_id`. The `for` loop over `datum`s below takes advantage
  of this in order to merge these three things together.

  Finally, with all of these things available together, this function delegates
  to `compute_predictions(...)` to post-process the prediction for each example
  and turn it into the JSON prediction format expected by the eval script.

  Args:
    candidates_dict: A dictionary containing the annotations from jsonl file.
    dev_features: Features loaded from tf_record file.
    raw_results: Output from running tensorflow graph.
    candidate_beam: see FLAGS.candidate_beam.
    max_answer_length: see FLAGS.max_answer_length.

  Returns:
    A dictionary containing predictions.
  """
  logging.info("Post-processing predictions started.")

  raw_results_by_id = [(int(res["unique_id"] + 1), res) for res in raw_results]
  # Cast example id to int32 for each example, similarly to the raw results.
  sess = tf.Session()
  all_candidates = candidates_dict.items()
  example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates
                                     ])).eval(session=sess)
  examples_by_id = list(zip(example_ids, all_candidates))
  if not examples_by_id:
    raise ValueError("No examples candidates found.")
  feature_ids = []
  features = []
  for f in dev_features:
    feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
    features.append(f.features.feature)
  feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
  features_by_id = list(zip(feature_ids, features))

  # Join examples with features and raw results.
  eval_examples = []
  merged = sorted(
      examples_by_id + raw_results_by_id + features_by_id,
      key=lambda pair: pair[0])

  # Error counters
  num_failed_matches = 0

  ex_count = 0
  feature_count = 0
  result_count = 0
  # `feature_unique_id` is an example ID or an example ID with something
  # appended on the end of it such that it sorts after the appropriate
  # example ID (see `convert_examples_to_features`).
  for feature_unique_id, datum in merged:
    # if from `examples_by_id`
    if isinstance(datum, tuple):
      ex_count += 1
      eval_examples.append(
          EvalExample(example_id=datum[0], candidates=datum[1]))
    # if from `features_by_id`
    elif "wp_start_offset" in datum:
      feature_count += 1
      # Join with the example that we just appended above, by
      # adding to the `EvalExample`'s `features` dict.
      if not eval_examples:
        logging.warning("Expected to already have example for this example id. "
                        "Dataset / predictions mismatch?")
        num_failed_matches += 1
        continue
      eval_examples[-1].features[feature_unique_id] = datum
    # if from `raw_results_by_id`
    else:
      result_count += 1
      # Join with the example that we just appended above, by
      # adding to the `EvalExample`'s `results` dict.
      if not eval_examples:
        logging.warning("Expected to already have example for this example id. "
                        "Dataset / predictions mismatch?")
        num_failed_matches += 1
        continue
      eval_examples[-1].results[feature_unique_id] = datum

  logging.info("  Num candidate examples found: %d", ex_count)
  logging.info("  Num candidate features found: %d", feature_count)
  logging.info("  Num results found: %d", result_count)
  logging.info("  len(merged): %d", len(merged))
  if num_failed_matches > 0:
    logging.warning("  Num failed matches: %d", num_failed_matches)

  # Construct prediction objects.
  tydi_pred_dict = {}
  for i, eval_example in enumerate(eval_examples):
    if i % 10000 == 0:
      logging.info("  Computing predictions for input example: %d/%d", i,
                   len(eval_examples))

    summary = compute_predictions(eval_example, candidate_beam,
                                  max_answer_length)
    if summary is not None:
      tydi_pred_dict[eval_example.example_id] = summary.predicted_label

  return tydi_pred_dict

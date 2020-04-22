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
# Lint as: python3
"""Evaluate predictions."""
import json

from absl import app
from absl import flags
from language.orqa.utils import eval_utils

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "references_path", None,
    "Path to a references file, where each line is a JSON "
    "dictionary with a `question` field and an `answer` field "
    "with a list of possible answers.")
flags.DEFINE_string(
    "predictions_path", None,
    "Path to a predictions file, where each line is a JSON "
    "dictionary with a `question` field and an `prediction` "
    "field with a single predicted answer string.")
flags.DEFINE_boolean(
    "is_regex", False,
    "Whether answer references are formatted as regexes. Only "
    "applicable to CuratedTrec")


def is_correct(answers, prediction):
  if FLAGS.is_regex:
    metric_fn = eval_utils.regex_match_score
  else:
    metric_fn = eval_utils.exact_match_score
  return eval_utils.metric_max_over_ground_truths(
      metric_fn=metric_fn, prediction=prediction, ground_truths=answers)


def main(_):
  if FLAGS.is_regex != ("CuratedTrec" in FLAGS.references_path):
    print("Warning: regex evaluation should (only) be applied to CuratedTrec.")

  references = {}
  with tf.io.gfile.GFile(FLAGS.references_path) as f:
    for line in f:
      example = json.loads(line)
      references[example["question"]] = example["answer"]
  print("Found {} references in {}".format(
      len(references), FLAGS.references_path))

  predictions = {}
  with tf.io.gfile.GFile(FLAGS.predictions_path) as f:
    for line in f:
      example = json.loads(line)
      predictions[example["question"]] = example["prediction"]
  print("Found {} predictions in {}".format(
      len(predictions), FLAGS.predictions_path))

  missing_predictions = 0
  correct = 0
  for q, a in references.items():
    if q in predictions:
      correct += int(is_correct(answers=a, prediction=predictions[q]))
    else:
      missing_predictions += 1
  print("Found {} missing predictions.".format(missing_predictions))
  print("Accuracy: {:.4f} ({}/{})".format(correct / float(len(references)),
                                          correct, len(references)))


if __name__ == "__main__":
  flags.mark_flag_as_required("references_path")
  flags.mark_flag_as_required("predictions_path")
  app.run(main)

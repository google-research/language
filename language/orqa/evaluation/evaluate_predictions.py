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
from absl import app
from absl import flags
from language.orqa.utils import eval_utils

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


def main(_):
  metrics = eval_utils.evaluate_predictions(FLAGS.references_path,
                                            FLAGS.predictions_path,
                                            FLAGS.is_regex)
  print("Found {} missing predictions.".format(metrics["missing_predictions"]))
  print("Accuracy: {:.4f} ({}/{})".format(metrics["accuracy"],
                                          metrics["num_correct"],
                                          metrics["num_total"]))


if __name__ == "__main__":
  flags.mark_flag_as_required("references_path")
  flags.mark_flag_as_required("predictions_path")
  app.run(main)

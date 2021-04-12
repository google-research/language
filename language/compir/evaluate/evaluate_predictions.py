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
"""Main script for evaluating predictions against test data."""



from absl import app
from absl import flags

from language.compir.evaluate import evaluate_predictions_utils

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "dataset",
    "scan",
    ["scan", "cfq", "atis", "geo", "scholar"],
    "The dataset to use.",
)

# Short description of the possible transformations:
# none (no transformation), rir (reversible), lird2 (lossy-direct),
# lird_rir2 (lossy-direct and reversible), lirind2 (lossy indirect),
# lirind_rir2 (lossy-indirect and reversible).
flags.DEFINE_enum(
    "transformation", "none",
    ["none", "rir", "lird2", "lird_rir2", "lirind2", "lirind_rir2"],
    "The transformation that was applied to generate the predictions")

flags.DEFINE_string("train_data_path", "", "Path to the training data.")

flags.DEFINE_string("test_data_path", "", "Path to the test data.")

flags.DEFINE_string(
    "prediction_path", "",
    "Path to test data predictions, to be evaluated against gold programs.")


def main(argv):

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  evaluate_predictions_utils.evaluate_predictions(FLAGS.dataset,
                                                  FLAGS.transformation,
                                                  FLAGS.train_data_path,
                                                  FLAGS.test_data_path,
                                                  FLAGS.prediction_path)


if __name__ == "__main__":
  app.run(main)

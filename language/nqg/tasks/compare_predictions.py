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
"""Compare a txt file of predictions with gold targets from a TSV file."""

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("gold", "", "tsv file containing gold targets.")

flags.DEFINE_string("predictions", "", "txt file with predicted targets.")


def main(unused_argv):
  gold_examples = tsv_utils.read_tsv(FLAGS.gold)

  preds = []
  with gfile.GFile(FLAGS.predictions, "r") as f:
    for line in f:
      preds.append(line.rstrip())

  correct = 0
  incorrect = 0
  for pred, gold_example in zip(preds, gold_examples):
    if pred == gold_example[1]:
      correct += 1
    else:
      incorrect += 1
      print("Incorrect for example %s.\nTarget: %s\nPrediction: %s" %
            (gold_example[0], gold_example[1], pred))

  print("correct: %s" % correct)
  print("incorrect: %s" % incorrect)
  print("pct: %s" % str(float(correct) / float(correct + incorrect)))


if __name__ == "__main__":
  app.run(main)

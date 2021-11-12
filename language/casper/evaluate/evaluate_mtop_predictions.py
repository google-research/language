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
r"""Evaluate the model's predictions on MTOP.

The gold and predicted parses should be in aligned rows in TSV files.
"""
from absl import app
from absl import flags
from absl import logging
from language.casper.evaluate import top_metrics
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("gold_file", None,
                    "TSV file containing the gold MTOP parses.")
flags.DEFINE_integer("gold_column", 1,
                     "Column index of the gold parses in gold_file")
flags.DEFINE_string("pred_file", None,
                    "TSV file containing the predicted MTOP parses.")
flags.DEFINE_integer("pred_column", 0,
                     "Column index of the predicted parses in pred_file")


def main(_):
  # Read the gold and pred parses.
  golds = []
  with tf.io.gfile.GFile(FLAGS.gold_file) as f_gold:
    for line in f_gold:
      golds.append(line.split("\t")[FLAGS.gold_column].strip())
  preds = []
  with tf.io.gfile.GFile(FLAGS.pred_file) as f_pred:
    for line in f_pred:
      preds.append(line.split("\t")[FLAGS.pred_column].strip())
  if len(golds) != len(preds):
    raise ValueError("Unequal number of parses: gold = {}, pred = {}".format(
        len(golds), len(preds)))

  # Evaluate
  eval_results = top_metrics.top_metrics(golds, preds)
  logging.info("Number of examples: %d", int(eval_results["num_total"]))
  logging.info("Exact match accuracy: %.2f",
               eval_results["full_accuracy"] * 100)
  logging.info("Template accuracy: %.2f",
               eval_results["intent_arg_accuracy"] * 100)


if __name__ == "__main__":
  app.run(main)

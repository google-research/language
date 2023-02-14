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
"""Computes end-to-end evaluation metrics."""

from absl import app
from absl import flags

from language.quest.common import example_utils
from language.quest.eval import eval_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("gold", "", "Examples jsonl file with gold document set.")

flags.DEFINE_string("pred", "", "Examples jsonl file with predicted documents.")


def main(unused_argv):
  gold_examples = example_utils.read_examples(FLAGS.gold)
  pred_examples = example_utils.read_examples(FLAGS.pred)

  # List of values for average precision, recall, and f1.
  p_vals = []
  r_vals = []
  f1_vals = []

  query_to_pred_example = {ex.query: ex for ex in pred_examples}
  for gold_example in gold_examples:
    if not gold_example.docs:
      raise ValueError("Example has 0 docs.")

    pred_example = query_to_pred_example[gold_example.query]

    predicted_docs = set(pred_example.docs)
    gold_docs = set(gold_example.docs)
    tp = len(gold_docs.intersection(predicted_docs))
    fp = len(predicted_docs.difference(gold_docs))
    fn = len(gold_docs.difference(predicted_docs))
    if tp:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * precision * recall / (precision + recall)
    else:
      precision = 0.0
      recall = 0.0
      f1 = 0.0

    p_vals.append(precision)
    r_vals.append(recall)
    f1_vals.append(f1)

  print("Avg. Precision")
  eval_utils.print_avg(gold_examples, p_vals)
  eval_utils.print_avg_by_template(gold_examples, p_vals)
  print("Avg. Recall")
  eval_utils.print_avg(gold_examples, r_vals)
  eval_utils.print_avg_by_template(gold_examples, r_vals)
  print("Avg. F1")
  eval_utils.print_avg(gold_examples, f1_vals)
  eval_utils.print_avg_by_template(gold_examples, f1_vals)


if __name__ == "__main__":
  app.run(main)

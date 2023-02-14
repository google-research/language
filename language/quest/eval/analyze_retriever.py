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
"""Analyze retriever predictions."""

from absl import app
from absl import flags

from language.quest.common import example_utils
from language.quest.eval import eval_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("gold", "", "Examples jsonl file with gold document set.")

flags.DEFINE_string("pred", "", "Examples jsonl file with predicted documents.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")

flags.DEFINE_bool("verbose", False, "Whether to print out verbose debugging.")

# Values of k to use to compute MRecall@k.
K_VALS = [20, 50, 100, 1000]


def main(unused_argv):
  gold_examples = example_utils.read_examples(FLAGS.gold)
  pred_examples = example_utils.read_examples(FLAGS.pred)

  num_examples = 0
  # Dictionary mapping mrecall k value to number of examples where predicted
  # set is superset of gold set.
  mrecall_vals = {k: [] for k in K_VALS}
  # List of recall for each example.
  recall_vals = {k: [] for k in K_VALS}

  query_to_pred_example = {ex.query: ex for ex in pred_examples}
  for idx, gold_example in enumerate(gold_examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break

    if not gold_example.docs:
      raise ValueError("Example has 0 docs.")

    if FLAGS.verbose:
      print("\n\nProcessing example %s: `%s`" % (idx, gold_example))
      num_examples += 1

    pred_example = query_to_pred_example[gold_example.query]

    for k in K_VALS:
      if FLAGS.verbose:
        print("Evaluating MRecall@%s" % k)
      predicted_docs = set(pred_example.docs[:k])
      gold_docs = set(gold_example.docs)
      if gold_docs.issubset(predicted_docs):
        if FLAGS.verbose:
          print("Contains all docs!")
        mrecall_vals[k].append(1.0)
      else:
        mrecall_vals[k].append(0.0)

      # Compute recall.
      covered_docs = gold_docs.intersection(predicted_docs)
      recall = float(len(covered_docs)) / len(gold_docs)
      recall_vals[k].append(recall)

      # Print debugging.
      extra_docs = predicted_docs.difference(gold_docs)
      missing_docs = gold_docs.difference(predicted_docs)
      if FLAGS.verbose:
        print("Extra docs: %s" % extra_docs)
        print("Missing docs: %s" % missing_docs)

  print("num_examples: %s" % num_examples)

  for k in K_VALS:
    print("MRecall@%s" % k)
    eval_utils.print_avg(gold_examples, mrecall_vals[k])
    eval_utils.print_avg_by_template(gold_examples, mrecall_vals[k])
    print("Avg. Recall@%s" % k)
    eval_utils.print_avg(gold_examples, recall_vals[k])
    eval_utils.print_avg_by_template(gold_examples, recall_vals[k])


if __name__ == "__main__":
  app.run(main)

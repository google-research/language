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
r"""Evaluates retrieval results."""
import json


from absl import app
from absl import flags
from absl import logging
from language.casper.utils import data_types
from language.casper.utils import top_utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("index_file", None, "Index JSONL file.")
flags.DEFINE_string("retrieval_file", None,
                    "Example JSONL file with retrieval results.")
flags.DEFINE_integer("max_num_neighbors", 10,
                     "Maximum number of neighbors to evaluate.")
flags.DEFINE_enum("funcall_format", "top", ["top"],
                  "Format of the output function call or logical form.")

# A generic type for the examples loaded from a JSONL file.
Example = data_types.RawExample


def _read_jsonl(filename):
  """Yields entries from the JSONL file."""
  with tf.io.gfile.GFile(filename) as fin:
    for line in fin:
      yield json.loads(line)


def _get_frame(funcall):
  """Returns the frame string describing all intent and slot labels."""
  if FLAGS.funcall_format == "top":
    return top_utils.get_frame_top(funcall)
  else:
    raise ValueError(f"Unknown funcall_format: {FLAGS.funcall_format}")


def _get_intent(frame):
  """Returns the top intent label."""
  return frame.split("-")[0]


def _get_labels(frame):
  """Returns the set of all intent and slot labels."""
  for separator in ["-", ".", ","]:
    frame = frame.replace(separator, " ")
  return set(frame.strip().split())


def _log_stats(stats_at_k,
               total_count):
  """Logs the statistics as percentages of the total count."""
  logging.info(", ".join(
      "{:.1f}".format(value * 100. / total_count) for value in stats_at_k))


def main(_):
  hashed_id_to_frame = {}
  for entry in _read_jsonl(FLAGS.index_file):
    hashed_id = entry["hashed_id"]
    hashed_id_to_frame[hashed_id] = _get_frame(entry["output_str"])

  count = 0
  # % of examples where one of the top K neighbors has the gold intent
  intent_recall_at_k = [0] * FLAGS.max_num_neighbors
  # % of examples where one of the top K neighbors has the gold frame
  frame_recall_at_k = [0] * FLAGS.max_num_neighbors
  # % of examples where all gold labels are in the top K neighbors
  # (not necessarily in a single neighbor)
  all_label_recall_at_k = [0] * FLAGS.max_num_neighbors
  # % of gold labels covered by the top K neighbors
  num_gold_labels_total = 0
  num_gold_labels_recalled_at_k = [0] * FLAGS.max_num_neighbors

  for entry in _read_jsonl(FLAGS.retrieval_file):
    count += 1
    gold_frame = _get_frame(entry["output_str"])
    gold_intent = _get_intent(gold_frame)
    gold_labels = _get_labels(gold_frame)
    num_gold_labels_total += len(gold_labels)

    # Track whether we have found the gold intent, frame, and labels.
    found_gold_intent = False
    found_gold_frame = False
    gold_labels_left = set(gold_labels)

    # Go through the top K neighbors and update the statistics.
    neighbors = entry["exemplars"]["hashed_ids"][:FLAGS.max_num_neighbors]
    while len(neighbors) < FLAGS.max_num_neighbors:
      neighbors.append(None)
    for k, hashed_id in enumerate(neighbors):
      if hashed_id is not None:
        neighbor_frame = hashed_id_to_frame[hashed_id]
        if gold_frame == neighbor_frame:
          found_gold_frame = True
        neighbor_intent = _get_intent(neighbor_frame)
        if gold_intent == neighbor_intent:
          found_gold_intent = True
        neighbor_labels = _get_labels(neighbor_frame)
        gold_labels_left -= neighbor_labels

      intent_recall_at_k[k] += found_gold_intent
      frame_recall_at_k[k] += found_gold_frame
      all_label_recall_at_k[k] += (not gold_labels_left)
      num_gold_labels_recalled_at_k[k] += (
          len(gold_labels) - len(gold_labels_left))

  logging.info(
      "% of examples where one of the top K neighbors has the gold intent:")
  _log_stats(intent_recall_at_k, count)
  logging.info(
      "% of examples where one of the top K neighbors has the gold frame:")
  _log_stats(frame_recall_at_k, count)
  logging.info(
      "% of examples where all gold labels are in the top K neighbors:")
  _log_stats(all_label_recall_at_k, count)
  logging.info("% of gold labels covered by the top K neighbors:")
  _log_stats(num_gold_labels_recalled_at_k, num_gold_labels_total)


if __name__ == "__main__":
  app.run(main)

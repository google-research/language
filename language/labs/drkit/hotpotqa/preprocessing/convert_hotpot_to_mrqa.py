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
# coding=utf-8
"""Script to convert HotpotQA to MRQA format, with negatives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import re
from absl import flags
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("in_file", None, "HotpotQA train / dev file.")

flags.DEFINE_string("out_file", None, "Output file as gzipped jsonl.")


def _create_question(context, answer, question, id_):
  """Find answer in context and create question."""
  match = re.search(re.escape(answer), context, re.IGNORECASE)
  if match is not None:
    answers = [answer],
    detected_answers = [{
        "char_spans": [[match.start(), match.end() - 1]],
        "text": context[match.start():match.end()],
    }]
    is_impossible = False
  else:
    answers = []
    detected_answers = []
    is_impossible = True
  return {
      "context":
          context,
      "qas": [{
          "qid": id_,
          "is_impossible": is_impossible,
          "question": question,
          "answers": answers,
          "detected_answers": detected_answers,
      }],
  }


def main(_):
  with open(FLAGS.in_file) as f:
    data = json.load(f)
  print("Loaded %d questions" % len(data))

  with gzip.open(FLAGS.out_file, "wt") as fo:
    skipped = 0
    no_answer = 0
    total = 0
    split = 0
    for item in tqdm(data):
      if item["answer"].lower() in ["yes", "no"]:
        skipped += 1
        continue
      total += 1
      found_an_answer = False
      for ii, para in enumerate(item["context"]):
        context = "".join(para[1])
        ques = _create_question(context, item["answer"], item["question"],
                                item["_id"] + "_%d" % ii)
        if not ques["qas"][0]["is_impossible"]:
          found_an_answer = True
        fo.write(json.dumps(ques) + "\n")
        split += 1
      if not found_an_answer:
        no_answer += 1
  print("Total %d Skipped %d No Answer %d Split %d" %
        (total, skipped, no_answer, split))


if __name__ == "__main__":
  flags.mark_flag_as_required("in_file")
  flags.mark_flag_as_required("out_file")
  tf.app.run()

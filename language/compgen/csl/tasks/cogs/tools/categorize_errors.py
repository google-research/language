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
"""Group the errors by the categories from COGS evaluation sets."""
import collections

from absl import app
from absl import flags
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("raw_dataset", "", "Original dataset file.")

flags.DEFINE_string("converted_dataset", "",
                    "Dataset file with converted outputs.")

flags.DEFINE_string("predictions", "",
                    "File containing predictions, one per line.")

flags.DEFINE_string("analysis_file", "",
                    "If specified dump error analysis to this file.")


def main(_):
  n = acc = 0
  category_to_score = collections.Counter()
  with gfile.GFile(FLAGS.raw_dataset, "r") as raw_f, \
      gfile.GFile(FLAGS.converted_dataset, "r") as gold_f, \
      gfile.GFile(FLAGS.predictions, "r") as pred_f, \
      gfile.GFile(FLAGS.analysis_file or "/dev/null", "w") as ana_f:
    for i, (gold, pred, raw) in enumerate(zip(gold_f, pred_f, raw_f)):
      category = raw.strip().split("\t")[-1]
      utt, gold = gold.strip().split("\t")
      pred = pred.strip()
      n += 1
      acc += (gold == pred)
      category_to_score[category] += (gold == pred)
      if gold != pred:
        ana_f.write(f"Example #{i}\n"
                    f"Category: {category}\n"
                    f"Utterance: {utt}\n"
                    f"Gold: {gold}\n"
                    f"Pred: {pred}\n\n")
  print("Accuracy: {} / {} = {:.2f}%".format(acc, n, acc * 100. / n))
  for category, score in category_to_score.most_common():
    print(category, score)


if __name__ == "__main__":
  app.run(main)

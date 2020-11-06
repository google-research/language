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
"""Converts VIZWIZ data to both TFRecord and RC (SQuAD) format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import app
from absl import flags
from language.capwap.utils import image_utils
from language.capwap.utils import io_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")
COCO_DIR = os.path.join(DATA_DIR, "COCO")
VIZWIZ_DIR = os.path.join(DATA_DIR, "VIZWIZ")

flags.DEFINE_string("vizwiz_path", VIZWIZ_DIR, "Path to VIZWIZ data.")

flags.DEFINE_string("output_dir", os.path.join(VIZWIZ_DIR,
                                               "processed/questions"),
                    "Output data directory.")

flags.DEFINE_integer("train_shards", 64,
                     "Number of shards in training TFRecord files.")

flags.DEFINE_integer("val_shards", 4,
                     "Number of shards in validation TFRecord files.")

flags.DEFINE_integer("test_shards", 8,
                     "Number of shards in testing TFRecord files.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab.")

FLAGS = flags.FLAGS


def load_questions(questions_file, vocab):
  """Load and process VIZWIZ questions.

  Args:
    questions_file: JSON file containing VIZWIZ questions.
    vocab: A text_utils.Vocab instance.

  Returns:
    questions: Dictionary of image_id --> question/answers.
  """
  tf.logging.info("Loading %s", questions_file)
  with tf.io.gfile.GFile(questions_file, "r") as f:
    data = json.load(f)

  total = skipped = 0
  questions = collections.defaultdict(lambda: collections.defaultdict(list))
  for entry in data:
    # Skip yes/no and unanswerable.
    if not entry["answerable"]:
      continue
    if entry["answer_type"] == "yes/no":
      continue
    if entry["answer_type"] == "unanswerable":
      continue

    question = entry["question"]
    original_answers = [a["answer"] for a in entry["answers"]]

    # Filter non-applicable questions (check all allowed answers).
    total += 1
    answers = []
    majority = 0
    for answer in original_answers:
      if io_utils.filter_question(question, answer):
        majority -= 1
      else:
        majority += 1
        answers.append(answer)
    if not answers or majority <= 0:
      skipped += 1
      continue

    # Select the most common answer and put it first; it will then get picked
    # for training by default.
    answers = [a[0] for a in collections.Counter(answers).most_common()]

    image_id = os.path.splitext(entry["image"])[0]
    image_id = int(image_id.split("_")[-1])
    question = " ".join(vocab.tokenize(question))
    answers = [" ".join(vocab.tokenize(answer)) for answer in answers]
    questions[image_id]["question_ids"].append(image_id)
    questions[image_id]["questions"].append(question)
    questions[image_id]["answers"].append(answers)
  tf.logging.info("Filtered: %d/%d", skipped, total)
  tf.logging.info("Kept: %d", total - skipped)
  return questions


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Load VIZWIZ data.
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)
  vizwiz_data = collections.defaultdict(lambda: collections.defaultdict(list))
  for split in ["train", "val"]:
    questions_file = "%s/Annotations/%s.json" % (FLAGS.vizwiz_path, split)
    for image_id, entry in load_questions(questions_file, vocab).items():
      for k, v in entry.items():
        vizwiz_data[image_id][k].extend(v)

  # Convert to ImageMetadata.
  images = []
  for image_id, data in vizwiz_data.items():
    images.append(
        image_utils.ImageMetadata(
            image_id=image_id,
            question_ids=data["question_ids"],
            questions=data["questions"],
            answers=data["answers"],
            objects="%s/features.hdf5" % FLAGS.vizwiz_path))

  # Load pre-computed random splits.
  # (This is for backwards compatibility with previous experiments).
  splits_file = "%s/capwap_splits.json" % FLAGS.vizwiz_path
  with tf.io.gfile.GFile(splits_file, "r") as f:
    split_ids = json.load(f)
  splits = dict(
      test=[im for im in images if im.image_id in split_ids["test"]],
      val=[im for im in images if im.image_id in split_ids["val"]],
      train=[im for im in images if im.image_id in split_ids["train"]])

  # Make sure we have the right images.
  assert len(splits["train"]) == len(split_ids["train"])
  assert len(splits["val"]) == len(split_ids["val"])
  assert len(splits["test"]) == len(split_ids["test"])

  # Dump to sharded TFRecords and also RC format.
  for split, split_images in splits.items():
    io_utils.convert_to_tfrecords(
        dataset=split_images,
        num_shards=getattr(FLAGS, "%s_shards" % split),
        basename=os.path.join(FLAGS.output_dir, split),
        example_fn=io_utils.vqa_example)
    output_file = os.path.join(FLAGS.output_dir, split + ".json")
    io_utils.convert_to_rc(split_images, output_file)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)

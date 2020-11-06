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
"""Converts GQA data to both TFRecord and RC (SQuAD) format."""

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
GQA_DIR = os.path.join(DATA_DIR, "GQA")

flags.DEFINE_string("splits", os.path.join(COCO_DIR, "karpathy_splits.json"),
                    "Path to JSON file with pre-split image ids.")

flags.DEFINE_string("gqa_path", GQA_DIR, "Path to GQA data.")

flags.DEFINE_string("output_dir", os.path.join(GQA_DIR, "processed/questions"),
                    "Output data directory.")

flags.DEFINE_integer("train_shards", 256,
                     "Number of shards in training TFRecord files.")

flags.DEFINE_integer("val_shards", 4,
                     "Number of shards in validation TFRecord files.")

flags.DEFINE_integer("test_shards", 8,
                     "Number of shards in testing TFRecord files.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab.")

FLAGS = flags.FLAGS


def load_questions(questions_file, vocab):
  """Load and process GQA questions.

  Args:
    questions_file: JSON file containing GQA questions.
    vocab: A text_utils.Vocab instance.

  Returns:
    questions: Dictionary of image_id --> question/answers.
  """
  with tf.io.gfile.GFile(questions_file, "r") as f:
    data = json.load(f)

  total = skipped = 0
  questions = collections.defaultdict(lambda: collections.defaultdict(list))
  for qid, entry in data.items():
    # Convert image ids to integers if not already.
    image_id = entry["imageId"]
    if "n" in image_id:
      image_id = "1000000" + image_id[1:]
    image_id = int(image_id)

    # Filter non-applicable questions.
    total += 1
    if io_utils.filter_question(entry["question"], entry["answer"]):
      skipped += 1
      continue

    question = " ".join(vocab.tokenize(entry["question"]))
    answer = " ".join(vocab.tokenize(entry["answer"]))
    questions[image_id]["question_ids"].append(int(qid))
    questions[image_id]["questions"].append(question)
    questions[image_id]["answers"].append([answer])
  tf.logging.info("Filtered: %d/%d", skipped, total)
  tf.logging.info("Kept: %d", total - skipped)
  return questions


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Load COCO train image ids to be sure not to use them.
  coco_train = io_utils.load_karpathy_splits(FLAGS.splits)["train"]
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # Load pre-computed random splits.
  # (This is for backwards compatibility with previous experiments).
  splits_file = "%s/capwap_splits.json" % FLAGS.gqa_path
  with tf.io.gfile.GFile(splits_file, "r") as f:
    split_ids = json.load(f)

  # Load GQA train.
  filename = "%s/%s_balanced_questions.json" % (FLAGS.gqa_path, "train")
  gqa_data = load_questions(filename, vocab)
  images = []
  for image_id, data in gqa_data.items():
    images.append(
        image_utils.ImageMetadata(
            image_id=image_id,
            question_ids=data["question_ids"],
            questions=data["questions"],
            answers=data["answers"],
            objects="%s/train_features.hdf5" % FLAGS.gqa_path))

  # Make sure we have the right images.
  images = [im for im in images if im.image_id in split_ids["train"]]
  assert len(images) == len(split_ids["train"])

  # Dump.
  io_utils.convert_to_tfrecords(
      dataset=images,
      num_shards=FLAGS.train_shards,
      basename=os.path.join(FLAGS.output_dir, "train"),
      example_fn=io_utils.vqa_example)
  io_utils.convert_to_rc(images, os.path.join(FLAGS.output_dir, "train.json"))

  # Load GQA dev.
  filename = "%s/%s_balanced_questions.json" % (FLAGS.gqa_path, "val")
  gqa_data = load_questions(filename, vocab)
  images = []
  for image_id, data in gqa_data.items():
    images.append(
        image_utils.ImageMetadata(
            image_id=image_id,
            question_ids=data["question_ids"],
            questions=data["questions"],
            answers=data["answers"],
            objects="%s/val_features.hdf5" % FLAGS.gqa_path))
  images = [image for image in images if image.image_id not in coco_train]

  # Resplit into dev and test.
  splits = dict(
      val=[im for im in images if im.image_id in split_ids["val"]],
      test=[im for im in images if im.image_id in split_ids["test"]])

  # Make sure we have the right images.
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

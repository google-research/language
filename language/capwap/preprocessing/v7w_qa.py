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
"""Converts V7W data to both TFRecord and RC (SQuAD) format."""

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
V7W_DIR = os.path.join(DATA_DIR, "V7W")

flags.DEFINE_string("splits", os.path.join(COCO_DIR, "karpathy_splits.json"),
                    "Path to JSON file with pre-split image ids.")

flags.DEFINE_string("v7w_path", V7W_DIR, "Path to GQA data.")

flags.DEFINE_string("coco_path", COCO_DIR, "Path to COCO data.")

flags.DEFINE_string("output_dir", os.path.join(V7W_DIR, "processed/questions"),
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
  """Load and process V7W questions.

  Args:
    questions_file: JSON file containing V7W questions.
    vocab: A text_utils.Vocab instance.

  Returns:
    questions: Dictionary of image_id --> question/answers.
  """
  with tf.io.gfile.GFile(questions_file, "r") as f:
    data = json.load(f)["images"]

  total = skipped = 0
  questions = collections.defaultdict(lambda: collections.defaultdict(list))
  for image in data:
    for entry in image["qa_pairs"]:
      # Filter non-applicable questions.
      total += 1
      if io_utils.filter_question(entry["question"], entry["answer"]):
        skipped += 1
        continue

      image_id = image["image_id"]
      question = " ".join(vocab.tokenize(entry["question"]))
      answer = " ".join(vocab.tokenize(entry["answer"]))
      questions[image_id]["question_ids"].append(entry["qa_id"])
      questions[image_id]["questions"].append(question)
      questions[image_id]["answers"].append([answer])
  tf.logging.info("Filtered: %d/%d", skipped, total)
  tf.logging.info("Kept: %d", total - skipped)
  return questions


def make_splits(questions_file, visual_genome_file, karpathy_splits):
  r"""Make train, validation, and test splits.

  The following rules are used:
    train = (karpathy_train U v7w_train) \ (karpathy_val U karpathy_test)
    val = (karpathy_val U v7w_val) \ (train U karpathy_test)
    test = (karpathy_test U v7w_test) \ (train U val)

  Args:
    questions_file: JSON file containing V7W questions.
    visual_genome_file: File containing mapping from genome ids to COCO ids.
    karpathy_splits: Train, val, and test ids for the Karpathy COCO splits.

  Returns:
    splits: Dictionary of new train, val, and test splits for V7W.
  """
  # Load mapping from visual genome ids to coco ids.
  mapping = {}
  with tf.io.gfile.GFile(visual_genome_file, "r") as f:
    metadata = json.load(f)
    for image in metadata:
      mapping[image["image_id"]] = image["coco_id"]

  # Load v7w data.
  with tf.io.gfile.GFile(questions_file, "r") as f:
    data = json.load(f)["images"]

  # Create splits.
  splits = dict(train=set(), val=set(), test=set())
  for image in data:
    v7w_split = image["split"]
    image_id = image["image_id"]
    coco_id = mapping.get(image_id)

    if coco_id in karpathy_splits["train"]:
      splits["train"].add(image_id)
    elif coco_id in karpathy_splits["val"]:
      splits["val"].add(image_id)
    elif coco_id in karpathy_splits["test"]:
      splits["test"].add(image_id)
    elif v7w_split == "train":
      splits["train"].add(image_id)
    elif v7w_split == "val":
      splits["val"].add(image_id)
    else:
      splits["test"].add(image_id)

  return splits


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Create dataset splits.
  metadata_file = "%s/image_data.json" % FLAGS.v7w_path
  dataset_file = os.path.join(FLAGS.v7w_path, "dataset_v7w_telling.json")
  karpathy_splits = io_utils.load_karpathy_splits(FLAGS.splits)
  splits = make_splits(dataset_file, metadata_file, karpathy_splits)

  # Load Visual7W questions.
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)
  questions = load_questions(dataset_file, vocab)

  for split, image_ids in splits.items():
    # Convert to ImageMetadata.
    images = []
    for image_id, data in questions.items():
      if image_id not in image_ids:
        continue
      images.append(
          image_utils.ImageMetadata(
              image_id=image_id,
              question_ids=data["question_ids"],
              questions=data["questions"],
              answers=data["answers"],
              objects="%s/features.hdf5" % FLAGS.v7w_path))

    # Dump to sharded TFRecords.
    io_utils.convert_to_tfrecords(
        dataset=images,
        num_shards=getattr(FLAGS, "%s_shards" % split),
        basename=os.path.join(FLAGS.output_dir, split),
        example_fn=io_utils.vqa_example)

    # And to RC formatted file.
    output_file = os.path.join(FLAGS.output_dir, split + ".json")
    io_utils.convert_to_rc(images, output_file)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)

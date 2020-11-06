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
"""Converts VQA data to both TFRecord and RC (SQuAD) format."""

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
VQA_DIR = os.path.join(DATA_DIR, "VQA")

flags.DEFINE_string("splits", os.path.join(COCO_DIR, "karpathy_splits.json"),
                    "Path to JSON file with pre-split image ids.")

flags.DEFINE_string("coco_path", COCO_DIR, "Path to COCO data.")

flags.DEFINE_string("vqa_path", VQA_DIR, "Path to VQA data.")

flags.DEFINE_string("output_dir", os.path.join(VQA_DIR, "processed/questions"),
                    "Output data directory.")

flags.DEFINE_integer("train_shards", 256,
                     "Number of shards in training TFRecord files.")

flags.DEFINE_integer("val_shards", 4,
                     "Number of shards in validation TFRecord files.")

flags.DEFINE_integer("test_shards", 8,
                     "Number of shards in testing TFRecord files.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT directory.")

FLAGS = flags.FLAGS


def load_questions(questions_file, annotations_file, vocab):
  """Load and process VQA questions.

  Args:
    questions_file: JSON file containing VQA questions.
    annotations_file: JSON file containing VQA answers.
    vocab: A text_utils.Vocab instance.

  Returns:
    questions: Dictionary of image_id --> question/answers.
  """
  with tf.io.gfile.GFile(annotations_file, "r") as f:
    annotations = json.load(f)

  # Build mapping of question id to answer list. There are multiple answers from
  # different annotations for each question. We keep all of them, but put the
  # best (most common) answer first so it gets picked for training by default.
  qid_to_answer = {}
  for annotation in annotations["annotations"]:
    qid = annotation["question_id"]
    best = annotation["multiple_choice_answer"]
    answers = set([a["answer"] for a in annotation["answers"]]) - set(best)
    answers = [best] + list(answers)
    qid_to_answer[qid] = answers

  with tf.io.gfile.GFile(questions_file, "r") as f:
    data = json.load(f)

  total = skipped = 0
  questions = collections.defaultdict(lambda: collections.defaultdict(list))
  for entry in data["questions"]:
    image_id = entry["image_id"]
    question = entry["question"]
    qid = entry["question_id"]
    original_answers = qid_to_answer[qid]

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

    question = " ".join(vocab.tokenize(question))
    answers = [" ".join(vocab.tokenize(answer)) for answer in answers]
    questions[image_id]["question_ids"].append(int(qid))
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

  # Load and aggregate all VQA data.
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)
  questions = collections.defaultdict(lambda: collections.defaultdict(list))
  for split in ["train", "val"]:
    questions_file = ("%s/v2_OpenEnded_mscoco_%s2014_questions.json" %
                      (FLAGS.vqa_path, split))
    annotations_file = ("%s/v2_mscoco_%s2014_annotations.json" %
                        (FLAGS.vqa_path, split))
    split_questions = load_questions(questions_file, annotations_file, vocab)
    for image_id, entry in split_questions.items():
      for k, v in entry.items():
        questions[image_id][k].extend(v)

  # Re-split according to Karpathy splits.
  splits = io_utils.load_karpathy_splits(FLAGS.splits)
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
              objects="%s/%s_features.hdf5" % (FLAGS.coco_path, split)))

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

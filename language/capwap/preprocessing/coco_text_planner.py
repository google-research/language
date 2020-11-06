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
"""Convert synthetic question generation data on COCO captions to TFRecords.

This data is used to train the text planner (LEARN function of Alg. 1).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
from language.capwap.utils import image_utils
from language.capwap.utils import io_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")
COCO_DIR = os.path.join(DATA_DIR, "COCO")
SQA_DIR = os.path.join(DATA_DIR, "COCO/SQA")

flags.DEFINE_string("sqa_path", SQA_DIR, "Path to synthetic QA data.")

flags.DEFINE_string("coco_path", COCO_DIR, "Path to COCO data.")

flags.DEFINE_string("split", "train", "Split to use.")

flags.DEFINE_string("output_dir", SQA_DIR + "processed/planner/",
                    "Output data directory.")

flags.DEFINE_integer("num_shards", 256, "Number of shards in TFRecord files.")

FLAGS = flags.FLAGS


def load_synthetic_questions(pattern):
  """Loads synthetic question generation data from disk.

  Args:
    pattern: Glob file pattern for question generation output files.

  Returns:
    images: Dictionary of image_id to QA data.
  """
  # image --> questions/answers
  images = collections.defaultdict(lambda: collections.defaultdict(list))
  for fname in tf.io.gfile.glob(pattern):
    with tf.io.gfile.GFile(fname, "r") as f:
      for line in f:
        image_id, question, answer, caption = line.split("\t")
        image_id = int(image_id)
        images[image_id]["question_ids"].append(-1)
        images[image_id]["questions"].append(question)
        images[image_id]["answers"].append(answer)
        images[image_id]["captions"].append(caption)
  return images


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  pattern = "%s/%s-*" % (FLAGS.sqa_path, FLAGS.split)
  questions = load_synthetic_questions(pattern)

  # Convert to ImageMetadata.
  images = []
  for image_id, data in questions.items():
    metadata = image_utils.ImageMetadata(
        image_id=image_id,
        question_ids=data["question_ids"],
        questions=data["questions"],
        answers=data["answers"],
        captions=data["captions"],
        objects="%s/%s_features.hdf5" % (FLAGS.coco_path, FLAGS.split))
    images.append(metadata)

  # Dump to sharded TFRecords.
  io_utils.convert_to_tfrecords(
      dataset=images,
      num_shards=FLAGS.num_shards,
      basename=os.path.join(FLAGS.output_dir, FLAGS.split),
      example_fn=io_utils.vqa_example)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)

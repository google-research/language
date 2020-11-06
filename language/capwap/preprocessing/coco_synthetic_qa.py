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
"""Convert COCO data to the format expected by the question generation model.

We extract only the captions as the "context" for the generation model.

This data is used as part of line 5 of Alg. 1 (LEARN, input to QGEN).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import app
from absl import flags
from language.capwap.utils import io_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")
COCO_DIR = os.path.join(DATA_DIR, "COCO")

flags.DEFINE_string("splits", os.path.join(COCO_DIR, "karpathy_splits.json"),
                    "Path to JSON file with pre-split image ids.")

flags.DEFINE_string("coco_path", COCO_DIR, "Path to COCO data.")

flags.DEFINE_string("output_dir",
                    os.path.join(COCO_DIR, "processed/qgen/contexts"),
                    "Output data directory.")

flags.DEFINE_integer("train_shards", 8,
                     "Number of shards in training TFRecord files.")

flags.DEFINE_integer("val_shards", 4,
                     "Number of shards in validation TFRecord files.")

flags.DEFINE_integer("test_shards", 8,
                     "Number of shards in testing TFRecord files.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT directory.")

flags.DEFINE_integer("query_length", 20, "Maximum query length.")

flags.DEFINE_integer("max_length", 128, "Maximum input length.")

FLAGS = flags.FLAGS

# Extractive
ANSWER_TYPE = 3

RCInputs = collections.namedtuple(
    "RCInputs", ["image_id", "input_ids", "input_mask", "segment_ids"])


def rc_example(image):
  """Builds an Example proto for an image-caption pair."""
  features = dict(
      unique_ids=io_utils.int64_feature(image.image_id),
      input_ids=io_utils.int64_feature_list(image.input_ids),
      input_mask=io_utils.int64_feature_list(image.input_mask),
      segment_ids=io_utils.int64_feature_list(image.segment_ids),
      start_positions=io_utils.int64_feature(0),
      end_positions=io_utils.int64_feature(0),
      answer_types=io_utils.int64_feature(ANSWER_TYPE))
  return tf.train.Example(features=tf.train.Features(feature=features))


def load_captions(captions_file, vocab):
  """Loads image ids and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    vocab: A text_utils.Vocab instance.

  Returns:
    captions: Dictionary of image_id --> caption RCInputs.
  """
  tf.logging.info("Loading captions from %s", captions_file)
  with tf.io.gfile.GFile(captions_file, "r") as f:
    caption_data = json.load(f)

  image_to_captions = collections.defaultdict(list)
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = vocab.tokenize(annotation["caption"])
    caption = caption[:FLAGS.max_length - FLAGS.query_length - 3]
    question = [vocab.PAD for _ in range(FLAGS.query_length)]
    inputs = [vocab.CLS] + question + [vocab.SEP] + caption + [vocab.SEP]
    input_ids = [vocab.t2i(t) for t in inputs]
    input_mask = [1 for _ in range(len(inputs))]
    segment_ids = ([0 for _ in range(len(question) + 2)] +
                   [1 for _ in range(len(caption) + 1)])
    for _ in range(FLAGS.max_length - len(input_ids)):
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    image_to_captions[image_id].append((input_ids, input_mask, segment_ids))
  return image_to_captions


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Load data and re-split according to Karpathy paper.
  splits = io_utils.load_karpathy_splits(FLAGS.splits)
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)
  captions = collections.defaultdict(list)
  for split in ["train", "val"]:
    captions_file = ("%s/annotations/captions_%s2014.json" %
                     (FLAGS.coco_path, split))
    for image_id, split_captions in load_captions(captions_file, vocab).items():
      captions[image_id].extend(split_captions)

  for split, image_ids in splits.items():
    # Convert to RCInputs.
    inputs = []
    for image_id, image_captions in captions.items():
      if image_id not in image_ids:
        continue
      for input_ids, input_mask, segment_ids in image_captions:
        inputs.append(RCInputs(image_id, input_ids, input_mask, segment_ids))

    # Dump to sharded TFRecords.
    io_utils.convert_to_tfrecords(
        dataset=inputs,
        num_shards=getattr(FLAGS, "%s_shards" % split),
        basename=os.path.join(FLAGS.output_dir, split),
        example_fn=rc_example)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)

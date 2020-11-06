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
"""Converts COCO data to TFRecord file format.

This data is used for the out-of-domain supervised pretraining (Sec. 4.3.1).
It is also used to train the MLE baseline.
"""

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

flags.DEFINE_string("splits", os.path.join(COCO_DIR, "karpathy_splits.json"),
                    "Path to JSON file with pre-split image ids.")

flags.DEFINE_string("coco_path", COCO_DIR, "Path to COCO data.")

flags.DEFINE_string("output_dir", os.path.join(COCO_DIR, "processed/captions"),
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


def load_captions(captions_file, vocab):
  """Loads image ids and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    vocab: A text_utils.Vocab instance.

  Returns:
    captions: Dictionary of image_id --> captions.
  """
  tf.logging.info("Loading captions from %s", captions_file)
  with tf.io.gfile.GFile(captions_file, "r") as f:
    caption_data = json.load(f)
  image_to_captions = collections.defaultdict(list)
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption_tokens = vocab.tokenize(annotation["caption"])
    image_to_captions[image_id].append(" ".join(caption_tokens))
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
    # Convert to ImageMetadata.
    images = []
    for image_id, image_captions in captions.items():
      if image_id not in image_ids:
        continue
      metadata = image_utils.ImageMetadata(
          image_id=image_id,
          captions=image_captions,
          objects="%s/%s_features.hdf5" % (FLAGS.coco_path, split))
      images.append(metadata)

    # Dump to sharded TFRecords.
    io_utils.convert_to_tfrecords(
        dataset=images,
        num_shards=getattr(FLAGS, "%s_shards" % split),
        basename=os.path.join(FLAGS.output_dir, split),
        example_fn=io_utils.caption_example)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)

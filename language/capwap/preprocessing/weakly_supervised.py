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
"""Converts weakly supervised pre-training data to TFRecord file format.

This data is used to weakly-supervise the captioning policy (ADAPT of Alg. 1).
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
import numpy as np
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

flags.DEFINE_string("features", None, "Path to training features.")

flags.DEFINE_string("generation_file", None,
                    "Path to the set G of synthetic QA generations.")

flags.DEFINE_string("output_dir", None, "Output data directory.")

flags.DEFINE_string("reward", "exact_match",
                    "Reward function to use to filter generations.")

flags.DEFINE_float("threshold", 1.0, "Reward threshold to filter by.")

flags.DEFINE_integer("num_shards", 256, "Number of shards in TFRecord files.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab.")

flags.DEFINE_integer("limit", 16, "Max generations accepted per image.")

FLAGS = flags.FLAGS


def load_synthetic_captions(generation_file, vocab):
  """Loads weakly supervised examples.

  Args:
    generation_file: Path to output of conditional generation (text planner).
    vocab: Instance of text_utils.Vocab.

  Returns:
    synthetic_captions: Dictionary of image_id --> list of synthetic captions.
  """
  tf.logging.info("Loading generations.")
  synthetic_captions = collections.defaultdict(list)
  total = 0
  with tf.io.gfile.GFile(generation_file, "r") as f:
    for line in f:
      entry = json.loads(line)
      image_id = entry["image_id"]

      # Take captions that pass the reward filter.
      consistent_captions = []
      for idx in np.argsort(-np.array(entry["span_scores"])):
        caption = vocab.clean([vocab.i2t(i) for i in entry["token_ids"][idx]])
        if entry[FLAGS.reward][idx] >= FLAGS.threshold:
          consistent_captions.append(" ".join(caption))

      # Add (up to the limit) to the synthetic dataset.
      if consistent_captions:
        total += 1
        synthetic_captions[image_id].extend(consistent_captions[:FLAGS.limit])

        if total % 1000 == 0:
          for caption in consistent_captions:
            tf.logging.info(caption)
          tf.logging.info("Loaded %d generations.", total)

  return synthetic_captions


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  if not FLAGS.output_dir:
    FLAGS.output_dir = os.path.dirname(FLAGS.generation_file)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Load generations.
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)
  captions = load_synthetic_captions(FLAGS.generation_file, vocab)
  images = []
  for image_id, captions in captions.items():
    metadata = image_utils.ImageMetadata(
        image_id=image_id, captions=captions, objects=FLAGS.features)
    images.append(metadata)

  # Dump to sharded TFRecords.
  io_utils.convert_to_tfrecords(
      dataset=images,
      num_shards=FLAGS.num_shards,
      basename=os.path.join(FLAGS.output_dir, "train"),
      example_fn=io_utils.caption_example)


if __name__ == "__main__":
  flags.mark_flag_as_required("features")
  flags.mark_flag_as_required("generation_file")
  tf.disable_v2_behavior()
  app.run(main)

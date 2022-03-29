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
"""Create FROST Finetuning data using Spacy."""

import collections
import os

from absl import app
from absl import flags
from absl import logging

from language.frost import spacy_frost_annotator_lib

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "Input TFDS dataset.")

flags.DEFINE_list("splits", ["test", "validation"], "Data splits to prepare.")

flags.DEFINE_string("output_dir", None, "Output directory to save frost files.")

flags.DEFINE_integer("shard_count", 1, "Number of output shards per split.")

# TFDS target keys for different datasets
TARGET_KEYS = {
    "billsum": "summary",
    "cnn_dailymail": "highlights",
    "samsum": "summary",
    "xsum": "summary"
}


def create_byte_feature(value):
  feature = tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[value.encode() if isinstance(value, str) else value]))
  return feature


def main(_):
  logging.set_verbosity(logging.INFO)

  if FLAGS.dataset not in TARGET_KEYS:
    logging.error("%s is not supported. Supported datasets are %s.",
                  FLAGS.dataset, str(list(TARGET_KEYS.keys())))

  # Load TFDS dataset
  dataset, info = tfds.load(FLAGS.dataset, split=FLAGS.splits, with_info=True)
  logging.info("Dataset info: %s", info)

  # Spacy Forst annotator function
  frost_processor_fn = spacy_frost_annotator_lib.get_spacy_frost_processor_fn()

  for split, dataset_split in zip(FLAGS.splits, dataset):
    logging.info("%s: %s", split, dataset_split)

    # Initialize tfrecord writers
    writers = []
    for shard_index in range(FLAGS.shard_count):
      shard_file_name = (
          FLAGS.dataset + ".spacy_frost_annotated." + split + "." +
          str(shard_index) + "-of-" + str(FLAGS.shard_count) + ".tfrecord")
      writers.append(
          tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir, shard_file_name)))

    writer_index = 0

    total_written = 0
    for example in tfds.as_numpy(dataset_split):
      if TARGET_KEYS[FLAGS.dataset] not in example:
        logging.error("Target key %s is not available in TFDS examples.",
                      TARGET_KEYS[FLAGS.dataset])

      # Create feature
      features = collections.OrderedDict()
      for key in example:
        features[key] = create_byte_feature(example[key])

      target_with_entityplans = frost_processor_fn(
          example[TARGET_KEYS[FLAGS.dataset]].decode())
      features["frost_" + TARGET_KEYS[FLAGS.dataset]] = create_byte_feature(
          target_with_entityplans)
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))

      writers[writer_index].write(tf_example.SerializeToString())
      writer_index = (writer_index + 1) % len(writers)

      total_written += 1

    for writer in writers:
      writer.close()

    logging.info("Wrote %d total examples", total_written)


if __name__ == "__main__":
  flags.mark_flag_as_required("dataset")
  flags.mark_flag_as_required("splits")
  flags.mark_flag_as_required("output_dir")
  app.run(main)

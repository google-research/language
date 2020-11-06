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
"""IO utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import json
import math
import multiprocessing
import string

import h5py
from language.capwap.utils import image_utils
import numpy as np
import tensorflow.compat.v1 as tf

MAX_THREADS = 64

# ------------------------------------------------------------------------------
#
# TF Example helpers.
#
# ------------------------------------------------------------------------------


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def string_feature(value):
  return bytes_feature(value.encode("utf8"))


def int64_feature_list(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature_list(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature_list(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def string_feature_list(values):
  return bytes_feature([v.encode("utf8") for v in values])


def caption_example(image):
  """Convert image caption data into an Example proto.

  Args:
    image: A ImageMetadata instance.

  Returns:
    example: An Example proto with serialized tensor data.
  """
  # Collect image object information from metadata.
  image_features, positions = read_object(image.objects, image.image_id)

  # Serialize multi-dimensional tensor data.
  captions_proto = tf.make_tensor_proto(np.array(image.captions))
  features_proto = tf.make_tensor_proto(image_features)
  positions_proto = tf.make_tensor_proto(positions)

  # Create final features dict.
  features = dict(
      image_id=int64_feature(image.image_id),
      captions=bytes_feature(captions_proto.SerializeToString()),
      object_features=bytes_feature(features_proto.SerializeToString()),
      object_positions=bytes_feature(positions_proto.SerializeToString()))
  return tf.train.Example(features=tf.train.Features(feature=features))


def vqa_example(image):
  """Convert visual qa data into an Example proto.

  Args:
    image: An ImageMetadata instance.

  Returns:
    example: An Example proto with serialized tensor data.
  """
  # Collect image object information from metadata.
  image_features, positions = read_object(image.objects, image.image_id)

  # Serialize multi-dimensional tensor data.
  captions_proto = tf.make_tensor_proto(np.array(image.captions))
  question_ids_proto = tf.make_tensor_proto(np.array(image.question_ids))
  questions_proto = tf.make_tensor_proto(np.array(image.questions))
  features_proto = tf.make_tensor_proto(image_features)
  positions_proto = tf.make_tensor_proto(positions)

  # Take the first answer always for simplicity.
  # This is only used for training and unofficial eval.
  answers = copy.deepcopy(image.answers)
  for i, answer in enumerate(answers):
    answers[i] = answer[0]
  answers_proto = tf.make_tensor_proto(np.array(answers))

  # Create final features dict.
  features = dict(
      image_id=int64_feature(image.image_id),
      question_ids=bytes_feature(question_ids_proto.SerializeToString()),
      questions=bytes_feature(questions_proto.SerializeToString()),
      answers=bytes_feature(answers_proto.SerializeToString()),
      captions=bytes_feature(captions_proto.SerializeToString()),
      object_features=bytes_feature(features_proto.SerializeToString()),
      object_positions=bytes_feature(positions_proto.SerializeToString()))
  return tf.train.Example(features=tf.train.Features(feature=features))


# ------------------------------------------------------------------------------
#
# Data loading helpers.
#
# ------------------------------------------------------------------------------


def load_karpathy_splits(filename):
  """Load Karpathy COCO ids for train, val, and test."""
  splits = {"train": set(), "val": set(), "test": set()}
  with tf.io.gfile.GFile(filename, "r") as f:
    for image in json.load(f)["images"]:
      split = image["split"] if image["split"] != "restval" else "train"
      splits[split].add(image["cocoid"])
  return splits


def filter_question(question, answer):
  """Apply filtering rules to QA pair."""
  question = question.strip(string.punctuation + " ")
  answer = answer.strip(string.punctuation + " ")
  if not question:
    return True
  if not answer:
    return True
  if answer.lower() in ["yes", "no", "none", "unanswerable", "unsuitable"]:
    return True
  if any([c.isnumeric() for c in answer]):
    return True
  return False


def read_object(objects, image_id):
  """Read R-CNN object data from HDF5 file."""
  # Super slow but oh well.
  should_close = False
  if isinstance(objects, str):
    should_close = True
    objects = h5py.File(objects, "r")
  image_id = str(image_id)
  image_features = objects["features-" + image_id][:]
  image_bboxes = objects["bboxes-" + image_id]
  image_dims = objects["dims-" + image_id]
  positions = image_utils.quantize_bbox(
      bboxes=image_bboxes, height=image_dims[0], width=image_dims[1])
  if should_close:
    objects.close()
  return image_features, positions


# ------------------------------------------------------------------------------
#
# Data writing helpers.
#
# ------------------------------------------------------------------------------


def convert_shard(shard, shard_name_fn, example_fn):
  """Multithreading helper to serialize an individual shard to disk.

  Args:
    shard: Tuple of shard id (int) and examples (ImageMetadata list).
    shard_name_fn: Maps shard id to file name.
    example_fn: Maps ImageMetadata to Example proto.
  """
  shard_id, examples = shard

  # The example might have an "objects" attribute, which in this case, points
  # to the filename of a HDF5 file that should be opened. Many examples share
  # the same objects HDF5 file for storing their object data, so we keep the
  # file handle open until we hit an example that points to a different file.
  maybe_open_file = hasattr(examples[0], "objects")
  if maybe_open_file:
    current_file = examples[0].objects
    current_file_handle = h5py.File(current_file, "r")

  output_file = shard_name_fn(shard_id)
  tf.logging.info("Writing shard %s" % output_file)
  with tf.io.TFRecordWriter(output_file) as writer:
    for example in examples:
      # Check if the HDF5 file should be updated.
      if maybe_open_file:
        if example.objects != current_file:
          current_file_handle.close()
          current_file = example.objects
          current_file_handle = h5py.File(current_file, "r")
        example.objects = current_file_handle
      example_proto = example_fn(example)
      writer.write(example_proto.SerializeToString())
  if maybe_open_file:
    current_file_handle.close()


def convert_to_tfrecords(dataset, num_shards, basename, example_fn):
  """Convert a dataset to sharded TFRecords.

  Args:
    dataset: List of ImageMetadata.
    num_shards: Number of randomized shards to write dataset to.
    basename: Base name to write shards to (/path/name-xxxxx-of-yyyyy).
    example_fn: Returns Example proto given example metadata.
  """
  # Shuffle the ordering of images.
  np.random.seed(12345)
  np.random.shuffle(dataset)

  # Break dataset into num_shards.
  size = int(math.ceil(len(dataset) / num_shards))
  shards = [dataset[i:i + size] for i in range(0, len(dataset), size)]

  # Map with multithreading.
  tf.logging.info("Processing %d shards", num_shards)
  num_threads = min([num_shards, MAX_THREADS])
  workers = multiprocessing.pool.ThreadPool(num_threads)
  shard_name = basename + "-%.5d-of-" + "%.5d" % num_shards
  map_fn = functools.partial(
      convert_shard,
      shard_name_fn=lambda i: shard_name % i,
      example_fn=example_fn)
  workers.map(map_fn, enumerate(shards))
  tf.logging.info("Finished %d shards.", num_shards)


def convert_to_rc(dataset, filename):
  """Convert dataset of ImageMetadata items to RC-formatted JSON file.

  Args:
    dataset: List of ImageMetadata.
    filename: Path to write to.
  """
  rc_data = collections.defaultdict(dict)
  for example in dataset:
    image_id = example.image_id
    for i, qid in enumerate(example.question_ids):
      question = example.questions[i]
      answers = example.answers[i]
      rc_data[image_id][qid] = dict(question=question, answers=answers)
  with tf.io.gfile.GFile(filename, "w") as f:
    json.dump(rc_data, f)


class NumpyEncoder(json.JSONEncoder):
  """Helper to encode things with Numpy objects."""

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if np.issubdtype(obj, np.integer):
      return int(obj)
    if np.issubdtype(obj, np.float):
      return float(obj)
    if np.issubdtype(obj, np.bool):
      return bool(obj)
    return json.JSONEncoder.default(self, obj)

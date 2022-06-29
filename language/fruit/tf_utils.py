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
"""Utilities for converting to and from tf.Examples."""
import json

import tensorflow as tf
import tensorflow_datasets as tfds

DEFAULT_FEATURE_DESCRIPTION = {
    "inputs": tf.io.FixedLenFeature([], dtype=tf.string),
    "targets": tf.io.FixedLenFeature([], dtype=tf.string),
    "generatable_surfaces": tf.io.FixedLenFeature([], dtype=tf.string),
    "id": tf.io.FixedLenFeature([], dtype=tf.int64)
}


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@tf.autograph.experimental.do_not_convert
def maybe_decode(x):
  """Decodes bytes to strings if neccessary."""
  return x.decode() if isinstance(x, bytes) else x


def maybe_encode(x):
  """Encodes strings to bytes if neccessary."""
  return x.encode() if isinstance(x, str) else x


def _parse_fn(example_proto):
  """Parses an example from the WikiDiff dataset."""
  return tf.io.parse_single_example(example_proto, DEFAULT_FEATURE_DESCRIPTION)


def to_example(dictionary, serialize_to_string=False):
  """Converts dictionaries to examples."""
  feature = {
      "inputs":
          _bytes_feature(maybe_encode(dictionary["inputs"])),
      "targets":
          _bytes_feature(maybe_encode(dictionary["targets"])),
      "generatable_surfaces":
          _bytes_feature(
              maybe_encode(json.dumps(dictionary["generatable_surfaces"]))),
      "id":
          _int64_feature(dictionary["id"])
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  if serialize_to_string:
    example = example.SerializeToString()
  return example


def postprocess_example(example):
  """Converts examples to dictionaries."""
  return {
      "inputs":
          maybe_decode(example["inputs"]),
      "targets":
          maybe_decode(example["targets"]),
      "generatable_surfaces":
          json.loads(maybe_decode(example["generatable_surfaces"])),
      "id":
          example["id"]
  }


def from_tfrecords(input_pattern):
  """Loads a dataset from TFRecord format to a list of dictionaries."""
  filenames = tf.io.gfile.glob(input_pattern)
  dataset = tf.data.TFRecordDataset(filenames=filenames)
  dataset = dataset.map(_parse_fn)
  dataset = list(map(postprocess_example, tfds.as_numpy(dataset)))
  return dataset

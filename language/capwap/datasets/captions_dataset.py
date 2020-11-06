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
"""Creates a tf.data.Dataset for a dataset with references.

This dataset type is used for pre-training (both out of domain and WSP).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from language.capwap import datasets
from language.capwap.utils import image_utils
from language.capwap.utils import tensor_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

KEYS = frozenset({
    "input_type", "image_id", "token_inputs", "token_outputs",
    "condition_inputs", "object_features"
})


def parse_example(serialized_example):
  """Parse a serialized example proto."""
  features = tf.io.parse_single_example(
      serialized_example,
      features=dict(
          image_id=tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
          image=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          captions=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          question_ids=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          questions=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          answers=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          object_positions=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value=""),
          object_features=tf.io.FixedLenFeature(
              shape=[], dtype=tf.string, default_value="")))
  return features


def preprocess_mapper(features, params, lookup_table, vocab, mode):
  """Model-specific preprocessing of features from the dataset."""
  # Set input type.
  features["input_type"] = tf.constant(datasets.DatasetTypes.REFERENCE)

  if mode != tf.estimator.ModeKeys.PREDICT:
    # Select random caption.
    captions = tf.io.parse_tensor(features["captions"], tf.string)
    num_captions = tensor_utils.shape(captions, 0)
    rid = tf.random.uniform([], maxval=num_captions, dtype=tf.int32)

    caption = text_utils.build_text_inputs(
        text=captions[rid],
        length=params["caption_length"],
        lookup_table=lookup_table,
        segment_id=0,
        start_token=vocab.CLS,
        end_token=vocab.SEP)
    assert isinstance(caption, text_utils.TextInputs)

    features["token_inputs"] = text_utils.TextInputs(
        token_ids=caption.token_ids[:-1],
        mask=caption.mask[:-1],
        segment_ids=caption.segment_ids[:-1],
        positions=caption.positions[:-1])

    features["token_outputs"] = text_utils.TextOutputs(
        token_ids=caption.token_ids[1:], mask=caption.mask[1:])

    if params.get("conditional_decoding"):
      random_span = text_utils.get_random_span(
          text=captions[rid],
          p=params["span_sample_p"],
          max_span_len=params["span_length"])

      features["condition_inputs"] = text_utils.build_text_inputs(
          text=random_span,
          length=params["condition_length"],
          lookup_table=lookup_table,
          segment_id=1,
          start_token=vocab.ANS)

  features["object_features"] = image_utils.parse_object_features(
      features["object_features"], features["object_positions"], params)

  # Remove extra inputs.
  features = {f: features[f] for f in features if f in KEYS}

  # Add dummy inputs for standardization for multi-tasking.
  footprint = datasets.footprint(params)
  assert footprint
  for k, v in footprint.items():
    if k not in features:
      features[k] = v

  return features


def get_dataset(params, mode, file_pattern, vocab):
  """Gets a tf.data.Dataset representing the captions data."""
  data_files = tf.io.gfile.glob(file_pattern)
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  # Create dataset from files.
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))

  # Shuffle.
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=len(data_files))

  # Repeat.
  if mode != tf.estimator.ModeKeys.PREDICT:
    dataset = dataset.repeat()

  # Load TFRecords from files.
  dataset = dataset.interleave(
      functools.partial(tf.data.TFRecordDataset, buffer_size=16 * 1024 * 1024),
      cycle_length=len(data_files),
      num_parallel_calls=len(data_files))

  # Shuffle.
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=1000)

  # Decode TFRecords.
  dataset = dataset.map(
      parse_example, num_parallel_calls=params["num_input_threads"])

  # Preprocess.
  dataset = dataset.map(
      functools.partial(
          preprocess_mapper,
          params=params,
          lookup_table=vocab.get_string_lookup_table(),
          vocab=vocab,
          mode=mode),
      num_parallel_calls=params["num_input_threads"])

  return dataset

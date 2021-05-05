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
"""Performs IO somewhat specific to TensorFlow.

This includes reading/writing `tf.Example`s to/from TF record files and opening
files via `tf.gfile`.
"""

import collections
import gzip
import json
from typing import Iterator, List, Sequence, Text

from absl import logging
from language.canine.tydiqa import data
from language.canine.tydiqa import preproc
from language.canine.tydiqa import tydi_tokenization_interface as tok_interface
import tensorflow.compat.v1 as tf


def read_entries(input_jsonl_pattern: Text,
                 tokenizer: tok_interface.TokenizerWithOffsets,
                 max_passages: int, max_position: int, fail_on_invalid: bool):
  """Reads TyDi QA examples from JSONL files.

  Args:
    input_jsonl_pattern: Glob of the gzipped JSONL files to read.
    tokenizer: Used to create special marker symbols to insert into the text.
    max_passages: see FLAGS.max_passages.
    max_position: see FLAGS.max_position.
    fail_on_invalid: Immediately stop if an error is found?

  Yields:
    tuple:
      input_file: str
      line_no: int
      tydi_entry: "TyDiEntry"s, dicts as returned by `create_entry_from_json`,
        one per line of the input JSONL files.
      debug_info: Dict containing debugging data.
  """
  matches = tf.gfile.Glob(input_jsonl_pattern)
  if not matches:
    raise ValueError(f"No files matched: {input_jsonl_pattern}")
  for input_path in matches:
    with gzip.GzipFile(fileobj=tf.gfile.Open(input_path, "rb")) as input_file:  # pytype: disable=wrong-arg-types
      for line_no, line in enumerate(input_file, 1):
        json_elem = json.loads(line, object_pairs_hook=collections.OrderedDict)
        entry = preproc.create_entry_from_json(
            json_elem,
            tokenizer,
            max_passages=max_passages,
            max_position=max_position,
            fail_on_invalid=fail_on_invalid)

        if not entry:
          tf.logging.info("Invalid Example %d", json_elem["example_id"])
          if fail_on_invalid:
            raise ValueError("Invalid example at {}:{}".format(
                input_path, line_no))

        # Return a `debug_info` dict that methods throughout the codebase
        # append to with debugging information.
        debug_info = {"json": json_elem}
        yield input_path, line_no, entry, debug_info


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  # This needs to be kept in sync with `FeatureWriter`.
  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class FeatureWriter(object):
  """Writes InputFeatures to TF example file."""

  def __init__(self, filename: Text, is_training: bool):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature: preproc.InputFeatures) -> None:
    """Writes InputFeatures to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values: Sequence[int]) -> tf.train.Feature:
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    # This needs to be kept in sync with `input_fn_builder`.
    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["language_id"] = create_int_feature([feature.language_id])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      features["wp_start_offset"] = create_int_feature(feature.wp_start_offset)
      features["wp_end_offset"] = create_int_feature(feature.wp_end_offset)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


class CreateTFExampleFn(object):
  """Functor for creating TyDi `tf.Example`s to be written to a TFRecord file."""

  def __init__(self, is_training, max_question_length, max_seq_length,
               doc_stride, include_unknowns,
               tokenizer: tok_interface.TokenizerWithOffsets):
    self.is_training = is_training
    self.tokenizer = tokenizer
    self.max_question_length = max_question_length
    self.max_seq_length = max_seq_length
    self.doc_stride = doc_stride
    self.include_unknowns = include_unknowns

  def process(self,
              entry,
              errors,
              debug_info=None) -> Iterator[tf.train.Example]:
    """Converts TyDi entries into serialized tf examples.

    Args:
      entry: "TyDi entries", dicts as returned by `create_entry_from_json`.
      errors: A list that this function appends to if errors are created. A
        non-empty list indicates problems.
      debug_info: A dict of information that may be useful during debugging.
        These elements should be used for logging and debugging only. For
        example, we log how the text was tokenized into WordPieces.

    Yields:
      `tf.train.Example` with the features needed for training or inference
      (depending on how `is_training` was set in the constructor).
    """
    if not debug_info:
      debug_info = {}
    tydi_example = data.to_tydi_example(entry, self.is_training)
    debug_info["tydi_example"] = tydi_example
    input_features = preproc.convert_single_example(
        tydi_example,
        tokenizer=self.tokenizer,
        is_training=self.is_training,
        max_question_length=self.max_question_length,
        max_seq_length=self.max_seq_length,
        doc_stride=self.doc_stride,
        include_unknowns=self.include_unknowns,
        errors=errors,
        debug_info=debug_info,
    )  # type: List[preproc.InputFeatures]
    for input_feature in input_features:
      input_feature.example_index = int(entry["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["example_index"] = create_int_feature(
          [input_feature.example_index])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)
      features["language_id"] = create_int_feature([input_feature.language_id])

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        features["wp_start_offset"] = create_int_feature(
            input_feature.wp_start_offset)
        features["wp_end_offset"] = create_int_feature(
            input_feature.wp_end_offset)

      yield tf.train.Example(features=tf.train.Features(feature=features))


def gopen(path):
  """Opens a file object given a (possibly gzipped) `path`."""
  logging.info("*** Loading from: %s ***", path)
  if ".gz" in path:
    return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))  # pytype: disable=wrong-arg-types
  else:
    return tf.gfile.Open(path, "r")

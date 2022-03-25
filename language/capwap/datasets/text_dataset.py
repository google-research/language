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
"""Creates a tf.data.Dataset for a language modelling dataset.

This type of datset is used when there is no image (e.g., raw wiki text).
The output of the question generation model on Wikipedia text is ready to be
read in directly by this dataset type.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from language.capwap import datasets
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


def preprocess_mapper(raw_text, params, lookup_table, vocab):
  """Model-specific preprocessing of features from the dataset."""
  features = dict(input_type=datasets.DatasetTypes.GUIDED)

  splits = tf.strings.split([raw_text], "\t").values
  question, answer, text = splits[1], splits[2], splits[3]

  text = text_utils.build_text_inputs(
      text=text,
      length=params["caption_length"],
      lookup_table=lookup_table,
      segment_id=0,
      start_token=vocab.CLS,
      end_token=vocab.SEP)
  assert isinstance(text, text_utils.TextInputs)

  features["token_inputs"] = text_utils.TextInputs(
      token_ids=text.token_ids[:-1],
      mask=text.mask[:-1],
      segment_ids=text.segment_ids[:-1],
      positions=text.positions[:-1])

  features["token_outputs"] = text_utils.TextOutputs(
      token_ids=text.token_ids[1:], mask=text.mask[1:])

  if params.get("conditional_decoding"):
    features["condition_inputs"] = text_utils.build_planner_inputs(
        question=question,
        answer=answer,
        length=params["condition_length"],
        lookup_table=lookup_table)

  # Add dummy inputs for standardization for multi-tasking.
  for k, v in datasets.footprint(params).items():
    if k not in features:
      features[k] = v

  return features


def get_dataset(params, mode, file_pattern, vocab):
  """Gets a tf.data.Dataset representing LM data."""
  data_files = tf.io.gfile.glob(file_pattern)
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  # Create dataset from files.
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))

  # Shuffle.
  if mode == tf_estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=len(data_files))

  # Repeat.
  if mode != tf_estimator.ModeKeys.PREDICT:
    dataset = dataset.repeat()

  # Load text from files.
  dataset = dataset.interleave(
      functools.partial(tf.data.TextLineDataset, buffer_size=16 * 1024 * 1024),
      cycle_length=len(data_files),
      num_parallel_calls=len(data_files))

  # Shuffle.
  if mode == tf_estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=1000)

  # Preprocess.
  dataset = dataset.map(
      functools.partial(
          preprocess_mapper,
          params=params,
          lookup_table=vocab.get_string_lookup_table(),
          vocab=vocab),
      num_parallel_calls=params["num_input_threads"])

  return dataset

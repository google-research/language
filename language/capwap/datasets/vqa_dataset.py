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
"""Creates a tf.data.Dataset for a visual QA dataset.

This dataset type is used for reinforce training.
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
    "input_type", "image_id", "question_id", "question_inputs", "answer_inputs",
    "answer_outputs", "condition_inputs", "object_features"
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


def expand_example(features, sample_one=True):
  """Expand nested tensor protos into multiple examples."""
  question_ids = tf.io.parse_tensor(features["question_ids"], out_type=tf.int64)
  questions = tf.io.parse_tensor(features["questions"], out_type=tf.string)
  answers = tf.io.parse_tensor(features["answers"], out_type=tf.string)
  num_qas = tensor_utils.shape(questions, 0)

  if sample_one:
    rid = tf.random.uniform([], maxval=num_qas, dtype=tf.int32)
    question_ids = tf.expand_dims(question_ids[rid], 0)
    questions = tf.expand_dims(questions[rid], 0)
    answers = tf.expand_dims(answers[rid], 0)
    num_qas = 1

  image_ids = tf.tile(tf.expand_dims(features["image_id"], 0), [num_qas])
  images = tf.tile(tf.expand_dims(features["image"], 0), [num_qas])
  object_features = tf.tile(
      tf.expand_dims(features["object_features"], 0), [num_qas])
  object_positions = tf.tile(
      tf.expand_dims(features["object_positions"], 0), [num_qas])

  features = dict(
      image_id=image_ids,
      image=images,
      object_features=object_features,
      object_positions=object_positions,
      question_id=question_ids,
      question=questions,
      answer=answers)
  return tf.data.Dataset.from_tensor_slices(features)


def preprocess_mapper(features, params, lookup_table, vocab):
  """Model-specific preprocessing of features from the dataset."""
  # Set input type.
  features["input_type"] = tf.constant(datasets.DatasetTypes.VQA)

  # Fix question id.
  features["question_id"] = tf.ensure_shape(features["question_id"], [])

  features["question_inputs"] = text_utils.build_text_inputs(
      text=features["question"],
      length=params["question_length"],
      lookup_table=lookup_table,
      segment_id=0)

  answer = text_utils.build_text_inputs(
      text=features["answer"],
      length=params["answer_length"],
      lookup_table=lookup_table,
      segment_id=1,
      start_token=vocab.ANS)
  assert isinstance(answer, text_utils.TextInputs)

  features["answer_inputs"] = answer

  features["answer_outputs"] = text_utils.TextOutputs(
      token_ids=answer.token_ids[1:], mask=answer.mask[1:])

  features["object_features"] = image_utils.parse_object_features(
      features["object_features"], features["object_positions"], params)

  if params.get("conditional_decoding"):
    features["condition_inputs"] = text_utils.build_planner_inputs(
        question=features["question"],
        answer=features["answer"],
        length=params["condition_length"],
        lookup_table=lookup_table)

  # Remove extra inputs.
  features = {f: features[f] for f in features if f in KEYS}

  # Add dummy inputs for standardization for multi-tasking.
  for k, v in datasets.footprint(params).items():
    if k not in features:
      features[k] = v

  return features


def get_dataset(params, mode, file_pattern, vocab):
  """Gets a tf.data.Dataset representing the VQA data."""
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
      num_parallel_calls=min(len(data_files), params["num_input_threads"]))

  # Shuffle.
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=1000)

  # Decode TFRecords.
  dataset = dataset.map(
      parse_example, num_parallel_calls=params["num_input_threads"])

  # Expand TFRecords.
  dataset = dataset.flat_map(
      functools.partial(
          expand_example, sample_one=not params.get("expand_by_question")))

  # Preprocess.
  dataset = dataset.map(
      functools.partial(
          preprocess_mapper,
          params=params,
          lookup_table=vocab.get_string_lookup_table(),
          vocab=vocab),
      num_parallel_calls=params["num_input_threads"])

  return dataset

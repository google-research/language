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
"""Dataset from the Natural Questions long answer task.

Fields:
  `question`: <string> [question_len]; tokens in the question.
  `context`: <string> [num_candidates, context_len]; tokens in each candidate.
  'answer_indices': <int32>[num_annotations]: answer indicated by each
  annotator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

flags.DEFINE_string("nq_long_train_pattern", None,
                    "Path to NQ long answer training data.")

flags.DEFINE_string("nq_long_eval_pattern", None,
                    "Path to NQ long answer eval data.")

FLAGS = flags.FLAGS


def split_on_whitespace(str_tensor):
  return tf.string_split(tf.expand_dims(str_tensor, -1)).values


def parse_example(serialized_example):
  """Parse example."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          "question":
              tf.FixedLenFeature([], tf.string),
          "context":
              tf.FixedLenSequenceFeature(
                  dtype=tf.string, shape=[], allow_missing=True),
          "long_answer_indices":
              tf.FixedLenSequenceFeature(
                  dtype=tf.int64, shape=[], allow_missing=True)
      })
  features["question"] = features["question"]
  features["context"] = features["context"]
  features["long_answer_indices"] = tf.to_int32(features["long_answer_indices"])
  return features


def get_dataset(is_train):
  """Gets a tf.data.Dataset representing the NQ data."""
  if is_train:
    data_pattern = FLAGS.nq_long_train_pattern
  else:
    data_pattern = FLAGS.nq_long_eval_pattern

  data_files = tf.gfile.Glob(data_pattern)
  assert data_files

  def _load_records(filenames):
    return tf.data.TFRecordDataset(filenames, buffer_size=16 * 1024)

  if is_train:
    # During training, read from all files in parallel to improve the speed of
    # the input pipeline.
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=len(data_files)))
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _load_records, sloppy=is_train, cycle_length=len(data_files)))
  else:
    dataset = _load_records(data_files)
  dataset = dataset.map(parse_example, num_parallel_calls=6)
  return dataset

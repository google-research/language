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
"""Functions for input pipeline.

The input pipeline should be both GPU and TPU friendly.
"""

import tensorflow as tf


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but tf.int32 can be faster and more
  # memory efficient on certain hardware.
  for name in list(example.keys()):
    tensor = example[name]
    if tensor.dtype == tf.int64:
      tensor = tf.cast(tensor, dtype=tf.int32)
    example[name] = tensor

  return example


def _create_int_feature(length):
  return tf.io.FixedLenFeature([length], tf.int64)


def create_training_dataset(input_file, batch_size, config):
  """Returns `tf.data.Dataset` for training."""
  name_to_features = {}

  name_to_features["wordpiece_ids"] = _create_int_feature(
      config["max_num_wordpieces"])
  name_to_features["num_wordpieces"] = _create_int_feature(1)

  name_to_features["application_span_begin"] = _create_int_feature(
      config["max_num_applications"])
  name_to_features["application_span_end"] = _create_int_feature(
      config["max_num_applications"])
  name_to_features["application_rule_idx"] = _create_int_feature(
      config["max_num_applications"])

  name_to_features["nu_node_type"] = _create_int_feature(
      config["max_num_numerator_nodes"])
  name_to_features["nu_node_1_idx"] = _create_int_feature(
      config["max_num_numerator_nodes"])
  name_to_features["nu_node_2_idx"] = _create_int_feature(
      config["max_num_numerator_nodes"])
  name_to_features["nu_application_idx"] = _create_int_feature(
      config["max_num_numerator_nodes"])
  name_to_features["nu_num_nodes"] = _create_int_feature(1)

  name_to_features["de_node_type"] = _create_int_feature(
      config["max_num_denominator_nodes"])
  name_to_features["de_node_1_idx"] = _create_int_feature(
      config["max_num_denominator_nodes"])
  name_to_features["de_node_2_idx"] = _create_int_feature(
      config["max_num_denominator_nodes"])
  name_to_features["de_application_idx"] = _create_int_feature(
      config["max_num_denominator_nodes"])
  name_to_features["de_num_nodes"] = _create_int_feature(1)

  if "*" in input_file:
    # Potentially match multiple input files.
    files = tf.io.matching_files(input_file)
    files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)
    dataset = shards.interleave(tf.data.TFRecordDataset)
  else:
    # Only using single input file.
    dataset = tf.data.TFRecordDataset(input_file)

  dataset = dataset.repeat()
  dataset = dataset.shuffle(buffer_size=1000)

  decode_fn = lambda record: _decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Send the single file to all workers.
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.OFF)
  dataset = dataset.with_options(options)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1024)
  return dataset


def get_dataset_fn(input_file, config):
  """Gets a closure to create a dataset.."""
  global_batch_size = config["batch_size"]

  def dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = create_training_dataset(input_file, batch_size, config)
    return dataset

  return dataset_fn

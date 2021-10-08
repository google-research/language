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
"""Data utils."""

import functools
import json
import math
import os


from absl import logging
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


def pad_fn(features, bsz):
  """Pads a batch up to specified batch size and adds sample weights."""
  new_features = {}
  for feature_name, feature in features.items():
    actual_bsz = tf.shape(feature)[0]
    pad_size = bsz - actual_bsz
    paddings = [[0, pad_size], [0, 0]]
    new_feature = tf.pad(feature, paddings)
    new_features[feature_name] = new_feature

  sample_weights = tf.ones(actual_bsz, dtype=tf.int64)
  sample_weights = tf.pad(sample_weights, [[0, pad_size]])
  new_features['sample_weights'] = sample_weights

  return new_features


def update_name_to_features(name_to_features,
                            samples_per_example):
  """Adjust feature shapes according to `samples_per_example`."""
  new_name_to_features = {}
  for k, v in name_to_features.items():
    old_shape = v.shape
    if isinstance(v.shape, int):
      old_shape = [old_shape]
    new_shape = old_shape.copy()
    new_shape[0] *= samples_per_example
    new_name_to_features[k] = tf.io.FixedLenFeature(
        new_shape, v.dtype, default_value=v.default_value)
  return new_name_to_features


def make_decode_fn(
    name_to_features,
    samples_per_example):
  """Make function to decode TF examples.

  Args:
    name_to_features: features to decode from a record in binary format.
    samples_per_example: how many TF Examples are packed in a single TF Record.

  Returns:
    A decode_fn function, which given a TF Record in binary format, parses it
      and returns features in the form of a dictionary
  """

  name_to_features = update_name_to_features(name_to_features,
                                             samples_per_example)

  def decode_fn(record):
    """Decodes tf examples."""
    example = tf.io.parse_single_example(record, name_to_features)
    for feature, tensor in example.items():
      new_tensor_shape = (samples_per_example, tensor.shape[0] //
                          samples_per_example) + tensor.shape[1:]
      example[feature] = tf.reshape(tensor, new_tensor_shape)
    return example

  return decode_fn


def _get_input_output_names(
    patterns, decode_fn,
    preprocess_fn
):
  """Records input and output features of decode_fn and preprocess_fn.

  Args:
    patterns: patterns for the input files.
    decode_fn: a function that decodes TF examples from binary format.
    preprocess_fn: a function that preprocess TF examples.

  Returns:
    A tuple of `input_names` and `output_types`.
      `input_names` is a list of features in the original TF records. This is
      determined by running `decode_fn` on the first sample from the data and
      recording all features it decoded.
      `output_types` is a dictionary containing all features in the preprocessed
      TF example and their types. This is determined by running `preprocess_fn`
      on the first TF Example from the data and recording
      all features and their types returned by the function.
  """
  if isinstance(patterns, list):
    input_files = tf.io.gfile.glob(patterns[0])
  else:
    input_files = tf.io.gfile.glob(patterns)
  first_record = next(iter(tf.data.TFRecordDataset(input_files[0])))
  first_example = decode_fn(first_record)
  input_names = list(first_example.keys())
  preprocessed_example = preprocess_fn(
      {k: v.numpy()[0] for k, v in first_example.items()})
  output_types = {k: v.dtype for k, v in preprocessed_example.items()}
  return input_names, output_types


def wrap_numpy_function(
    preprocess_input_names, preprocess_fn, preprocess_output_types
):
  """Wraps python function into Tensorflow op.

  Args:
    preprocess_input_names: names of all input features to the `preprocess_fn`
      function.
    preprocess_fn: a function that preprocesses a dictionary of features using
      numpy library.
    preprocess_output_types: names of all output features and their types
      returned to the `preprocess_fn` by function.

  Returns:
    A function that preprocesses features using tensorflow ops.
  """

  def preprocess_with_list_input_fn(*args):
    input_dict = {k: v for k, v in zip(preprocess_input_names, args)}
    output_dict = preprocess_fn(input_dict)
    output_list = tuple(output_dict[k] for k in preprocess_output_types)
    return output_list

  def f(example):
    input_list = [example[k] for k in preprocess_input_names]
    output_types = [v for k, v in preprocess_output_types.items()]
    output_list = tf.numpy_function(preprocess_with_list_input_fn, input_list,
                                    output_types)
    output_dict = {
        k: v for k, v in zip(preprocess_output_types.keys(), output_list)
    }
    return output_dict

  return f


def load_dataset(
    patterns,
    decode_fn,
    preprocess_fn,
    collater_fn,
    is_training,
    per_device_batch_size,
    local_device_count,
    host_count,
    host_id,
    seed = 0,
    pad_eval = False,
):
  """Create TF dataset from one or more file patterns.

  Args:
    patterns: one or more file pattern strings from which to load data.
    decode_fn: decodes bytes into samples.
    preprocess_fn: performs preprocessing on examples.
    collater_fn: performs preprocessing on the whole batch.
    is_training: if true, shuffles.
    per_device_batch_size: total batch size per host.
    local_device_count: number of devices, used for batching.
    host_count: number of hosts, used for sharding.
    host_id: ID of current host, used for sharding.
    seed: random seed for non-deterministic operations.
    pad_eval: if true, pad rather than discard last batch for evaluation.

  Returns:
    tf.data.Dataset for training.
  """

  dataset = tf.data.Dataset.list_files(patterns, seed=seed)
  num_files = len(dataset)
  if num_files // host_count >= 10:
    shard_first = True
  else:
    shard_first = False

  if is_training:
    if shard_first:
      dataset = dataset.shard(num_shards=host_count, index=host_id)
    dataset = dataset.shuffle(buffer_size=num_files, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_files,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    if not shard_first:
      dataset = dataset.shard(num_shards=host_count, index=host_id)
    dataset = dataset.shuffle(buffer_size=5000, seed=seed)
  else:
    dataset = tf.data.TFRecordDataset(dataset)

  # Decode tf example
  dataset = dataset.map(
      decode_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True,
  )
  # Flatten results to account for multiple samples per example
  dataset = dataset.unbatch()

  # We may not want to discard remainders or repeat for evaluation. Instead we
  # pad. If we shard first, one host might receive fewer batches than others, so
  # we first pad a full-size batch, then shard.
  if not is_training:
    if pad_eval:
      total_bsz = host_count * local_device_count * per_device_batch_size
      dataset = dataset.batch(
          batch_size=total_bsz,
          drop_remainder=False,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=True,
      )
      dataset = dataset.map(
          functools.partial(pad_fn, bsz=total_bsz),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=True,
      )
      dataset = dataset.unbatch()
    dataset = dataset.shard(num_shards=host_count, index=host_id)

  # Preprocess individual samples
  dataset = dataset.map(
      preprocess_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True,
  )

  # Generate device batches
  dataset = dataset.batch(
      batch_size=per_device_batch_size,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True,
  )

  dataset = dataset.map(
      collater_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True,
  )

  # Host batch contains one device batch per local device
  dataset = dataset.batch(
      batch_size=local_device_count,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True,
  )

  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset


def load_multi_dataset(
    datasets_config,
    name_to_features,
    preprocess_fn,
    collater_fn,
    is_training,
    per_device_batch_size,
    local_device_count,
    host_count,
    host_id,
    seed = 0,
    pad_eval = False,
):
  """Create TF dataset from different data sources.

  Args:
    datasets_config: a list of datasets to load.
    name_to_features: features to decode from a record in binary format.
    preprocess_fn: performs preprocessing on examples.
    collater_fn: performs preprocessing on the whole batch.
    is_training: if true, shuffles
    per_device_batch_size: total batch size per host.
    local_device_count: number of devices, used for batching.
    host_count: number of hosts, used for sharding.
    host_id: ID of current host, used for sharding.
    seed: random seed for non-deterministic operations.
    pad_eval: if true, pad rather than discard last batch for evaluation.

  Returns:
    tf.data.Dataset. Input data for training.
  """
  datasets = []
  for dataset_config in datasets_config:
    decode_fn_per_dataset = make_decode_fn(
        name_to_features=name_to_features,
        samples_per_example=dataset_config['samples_per_example'],
    )
    datasets.append(
        load_dataset(
            patterns=dataset_config['patterns'],
            decode_fn=decode_fn_per_dataset,
            preprocess_fn=preprocess_fn,
            collater_fn=collater_fn,
            is_training=is_training,
            per_device_batch_size=per_device_batch_size,
            local_device_count=local_device_count,
            host_count=host_count,
            host_id=host_id,
            seed=seed,
            pad_eval=pad_eval,
        ))
  if len(datasets) > 1:
    dataset = tf.data.experimental.sample_from_datasets(
        datasets, weights=None, seed=seed)
  else:
    dataset = datasets[0]
  return dataset


def load_sharded_array(
    pattern,
    stride,
    offset,
):
  """Load and concatenate numpy arrays according to pattern.

  Args:
    pattern: pattern of array filenames to be loaded.
    stride: stride to traverse paths to be loaded.
    offset: starting offset into paths to be loaded.

  Returns:
    Loaded and concatenated array.
  """
  paths = tf.io.gfile.glob(pattern)
  array_list = []
  for path in paths[offset::stride]:
    logging.info('Loading %s on to process %d', path, jax.process_index())
    with tf.io.gfile.GFile(path, 'rb') as f:
      array_list.append(np.load(f))

  array = np.concatenate(array_list, axis=0)
  return array


def save_sharded_array(array, prefix, num_shards,
                       stride, offset, shard_size_divisible):
  """Save numpy array into multiple shards.

  Input array will be divided and saved into files with names
  "prefix-X-of-num_shards", where `X` is in [offset, offset + stride, ...].

  Args:
    array: numpy array to be saved.
    prefix: output filenames prefix.
    num_shards: total number of shards.
    stride: stride for which shards save to the input array.
    offset: starting offset for which shards save to the input array.
    shard_size_divisible: ensure that the first dimension of each shard is
      divisible by this number via zero padding at the end.
  """
  if num_shards % stride != 0:
    raise ValueError('`num_shards` must to be divisible by the stride.')

  actual_num_shards = num_shards // stride

  actual_data_per_shard = np.full((actual_num_shards),
                                  array.shape[0] // actual_num_shards)
  actual_data_per_shard[:array.shape[0] % actual_num_shards] += 1

  shard_size = (
      math.ceil(actual_data_per_shard.max() / shard_size_divisible) *
      shard_size_divisible)

  array_shard = np.zeros(
      [shard_size] + list(array.shape[1:]), dtype=array.dtype)

  start_index = 0
  for i, shard_index in enumerate(range(offset, num_shards, stride)):
    end_index = min(array.shape[0], start_index + actual_data_per_shard[i])
    actual_size = end_index - start_index
    output_path = '%s-%05d-of-%05d' % (prefix, shard_index, num_shards)

    array_shard[:] = 0
    array_shard[:actual_size] = array[start_index:end_index]

    with tf.io.gfile.GFile(output_path, 'wb') as f:
      np.save(f, array_shard)

    start_index = end_index


def save_samples_to_json(features,
                         config, step):
  """Save samples to a json file."""
  save_samples_for_this_step = (
      config.get('save_samples_every_steps') and
      (step % config.get('save_samples_every_steps') == 0))

  process_index = jax.process_index()
  accepted_processes = config.get('save_samples_process_ids', 0)
  if isinstance(accepted_processes, list):
    save_samples_for_this_process = (process_index in accepted_processes)
  elif accepted_processes == -1:
    save_samples_for_this_process = True
  else:
    save_samples_for_this_process = (process_index == accepted_processes)

  if save_samples_for_this_step and save_samples_for_this_process:
    logging.info('Saving samples at step %d, process %d', step, process_index)
    path = os.path.join(config.model_dir, 'samples',
                        'step_%d.process_%d.json' % (step, process_index))
    tf.io.gfile.makedirs(os.path.dirname(path))
    with tf.io.gfile.GFile(path, 'ab') as fp:
      for batch in features:
        json.dump(batch, fp)
        fp.write('\n')

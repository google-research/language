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
"""Tests for data utils."""

import math
import os
from typing import Dict, Text

from absl.testing import absltest
from absl.testing import parameterized
from language.mentionmemory.utils import data_utils
import numpy as np
import tensorflow.compat.v2 as tf



class DataTest(parameterized.TestCase):

  data_files = ['test1.tfrecord', 'test2.tfrecord']
  dataset_size = 32
  data_dim = 16
  samples_per_example = 4
  name_to_features = {
      'x': tf.io.FixedLenFeature([data_dim], tf.int64),
  }

  @staticmethod
  def preprocess_fn(sample: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    return sample

  def _get_test_data_path(self, file_name):
    path = os.path.join('language/mentionmemory/utils/testdata', file_name)
    return path

  def setUp(self):
    super().setUp()

    self.decode_fn = data_utils.make_decode_fn(self.name_to_features,
                                               self.samples_per_example)

    self.data1 = np.random.randint(
        0,
        10,
        size=(self.dataset_size // self.samples_per_example,
              self.data_dim * self.samples_per_example),
        dtype=np.int64)

    self.data2 = np.random.randint(
        0,
        10,
        size=(self.dataset_size // self.samples_per_example,
              self.data_dim * self.samples_per_example),
        dtype=np.int64)

    self.workdir_obj = self.create_tempdir()
    self.workdir = self.workdir_obj.full_path
    self.data_patterns = [
        os.path.join(self.workdir, filename) for filename in self.data_files
    ]

    for dataset_idx, data in enumerate([self.data1, self.data2]):
      writer = tf.io.TFRecordWriter(self.data_patterns[dataset_idx])
      for row in data:
        features = tf.train.Features(feature={
            'x': tf.train.Feature(int64_list=tf.train.Int64List(value=row)),
        })

        record_bytes = tf.train.Example(features=features).SerializeToString()
        writer.write(record_bytes)

  @parameterized.parameters(
      (1, 1, 1, 0, False),
      (1, 1, 1, 0, True),
      (1, 1, 3, 0, True),
      (1, 1, 3, 2, True),
      (1, 3, 3, 2, True),
      (3, 2, 2, 1, True),
  )
  def test_reproduce_eval_data(
      self,
      per_device_batch_size,
      local_device_count,
      host_count,
      host_id,
      pad_eval,
  ):
    """Check whether loading dataset gets back original data."""

    loaded_data = data_utils.load_dataset(
        patterns=self.data_patterns[0],
        decode_fn=self.decode_fn,
        preprocess_fn=DataTest.preprocess_fn,
        collater_fn=lambda x: x,
        is_training=False,
        per_device_batch_size=per_device_batch_size,
        local_device_count=local_device_count,
        host_count=host_count,
        host_id=host_id,
        pad_eval=pad_eval,
    )
    data_list = list(loaded_data)
    recovered_data = np.array([batch['x'] for batch in data_list])

    # Compute how many non-zero elements the host should have
    host_bsz = per_device_batch_size * local_device_count
    total_bsz = host_bsz * host_count
    total_remainder = self.dataset_size % total_bsz
    remainder_per_host = total_remainder // host_count
    host_remainder = total_remainder % host_count
    n_nonzero_elements = self.dataset_size // total_bsz * host_bsz + remainder_per_host
    if host_id < host_remainder:
      n_nonzero_elements += 1
    n_batches = math.ceil(self.dataset_size / total_bsz)

    # Check data has correct shape
    self.assertEqual(
        recovered_data.shape,
        (n_batches, local_device_count, per_device_batch_size, self.data_dim))
    reshaped_data = recovered_data.reshape(-1, self.data_dim)

    # # Check if all samples in dataset are present in original data
    comparison_data = reshaped_data[:n_nonzero_elements]
    reshaped_original_data = self.data1.reshape(-1, self.data_dim)
    arr_eq = np.expand_dims(comparison_data,
                            1) == np.expand_dims(reshaped_original_data, 0)
    self.assertTrue(np.all(np.any(np.all(arr_eq, axis=-1), axis=1), axis=0))

    if pad_eval:
      # Check if nonzero sample weights correspond to expected nonzero elements
      sample_weights = np.array(
          [batch['sample_weights'] for batch in data_list]).reshape(-1)
      self.assertTrue(np.all(sample_weights[:n_nonzero_elements] == 1))
      self.assertTrue(np.all(sample_weights[n_nonzero_elements:] == 0))

      # Check if values are zero for sample weight zero
      self.assertTrue(np.all(reshaped_data[n_nonzero_elements:] == 0))

  def test_reproduce_train_data(self):
    loaded_data = data_utils.load_dataset(
        patterns=self.data_patterns[0],
        decode_fn=self.decode_fn,
        preprocess_fn=DataTest.preprocess_fn,
        collater_fn=lambda x: x,
        is_training=False,
        per_device_batch_size=1,
        local_device_count=1,
        host_count=1,
        host_id=0,
    )
    data_iter = iter(loaded_data)
    recovered_data = np.array(
        [data_iter.get_next()['x'] for _ in range(self.dataset_size)])

    recovered_data = recovered_data.reshape(
        self.dataset_size // self.samples_per_example,
        self.data_dim * self.samples_per_example)
    # We shuffle, so we just check that all arrays are present without regard
    # to order.
    arr_eq = np.expand_dims(recovered_data, 1) == np.expand_dims(self.data1, 0)
    self.assertTrue(np.all(np.any(np.all(arr_eq, axis=-1), axis=1), axis=0))

  @parameterized.parameters(
      {'is_training': True},
      {'is_training': False},
  )
  def test_training_data_deterministic(self, is_training=True):
    """Make sure loading data is deterministic given seed."""
    data_arrays = []
    for _ in range(2):
      loaded_dataset = data_utils.load_dataset(
          patterns=self.data_patterns,
          decode_fn=self.decode_fn,
          preprocess_fn=DataTest.preprocess_fn,
          collater_fn=lambda x: x,
          is_training=is_training,
          per_device_batch_size=4,
          local_device_count=1,
          host_count=2,
          host_id=0,
          seed=0,
      )
      data_iterator = iter(loaded_dataset)
      data_array = np.array([
          data_iterator.get_next()['x'] for _ in range(self.dataset_size // 4)
      ])
      data_arrays.append(data_array)

    self.assertTrue(np.array_equal(*data_arrays))

  @parameterized.parameters(
      ('eae_paper-00000-of-00001', 1, 1, False),
      ('eae_paper-00000-of-00001', 1, 20, False),
      ('eae_paper-00000-of-00001', 1, 1, True),
      ('eae_paper-00000-of-00001', 1, 4, True),
      ('mtb.v5-00000-of-00001', 4, 1, False),
      ('mtb.v5-00000-of-00001', 4, 10, False),
      ('mtb.v5-00000-of-00001', 4, 1, True),
      ('mtb.v5-00000-of-00001', 4, 2, True),
  )
  def test_load_real_data(self, filename, samples_per_example,
                          per_device_batch_size, is_training):
    num_total_samples = 100
    self.assertEqual(num_total_samples % per_device_batch_size, 0)
    data_path = self._get_test_data_path(filename)

    name_to_features = {
        'text_ids': tf.io.FixedLenFeature(128, tf.int64),
        'text_mask': tf.io.FixedLenFeature(128, tf.int64),
        'dense_span_starts': tf.io.FixedLenFeature(128, tf.int64),
        'dense_span_ends': tf.io.FixedLenFeature(128, tf.int64),
        'dense_mention_mask': tf.io.FixedLenFeature(128, tf.int64),
        'dense_mention_ids': tf.io.FixedLenFeature(128, tf.int64),
    }

    decode_fn = data_utils.make_decode_fn(name_to_features, samples_per_example)

    loaded_dataset = data_utils.load_dataset(
        patterns=data_path,
        decode_fn=decode_fn,
        preprocess_fn=DataTest.preprocess_fn,
        collater_fn=lambda x: x,
        is_training=is_training,
        per_device_batch_size=per_device_batch_size,
        local_device_count=1,
        host_count=1,
        host_id=0,
        seed=0,
    )
    data_list = iter(loaded_dataset)
    for _ in range(num_total_samples // per_device_batch_size):
      batch = next(data_list)
      self.assertIsNotNone(batch)

  @parameterized.parameters(
      (1, False),
      (2, False),
      (20, False),
      (100, False),
      (1, True),
      (2, True),
      (20, True),
      (100, True),
  )
  def test_load_multi_dataset_real_data(self, per_device_batch_size,
                                        is_training):
    self.assertEqual(100 % per_device_batch_size, 0)
    num_total_samples = 200
    datasets_config = [
        {
            'patterns': self._get_test_data_path('eae_paper-00000-of-00001'),
            'samples_per_example': 1,
        },
        {
            'patterns': self._get_test_data_path('mtb.v5-00000-of-00001'),
            'samples_per_example': 4,
        },
    ]
    name_to_features = {
        'text_ids': tf.io.FixedLenFeature(128, tf.int64),
        'text_mask': tf.io.FixedLenFeature(128, tf.int64),
        'dense_span_starts': tf.io.FixedLenFeature(128, tf.int64),
        'dense_span_ends': tf.io.FixedLenFeature(128, tf.int64),
        'dense_mention_mask': tf.io.FixedLenFeature(128, tf.int64),
        'dense_mention_ids': tf.io.FixedLenFeature(128, tf.int64),
    }
    loaded_dataset = data_utils.load_multi_dataset(
        datasets_config=datasets_config,
        name_to_features=name_to_features,
        preprocess_fn=DataTest.preprocess_fn,
        collater_fn=lambda x: x,
        is_training=is_training,
        per_device_batch_size=per_device_batch_size,
        local_device_count=1,
        host_count=1,
        host_id=0,
        seed=0,
    )
    data_list = iter(loaded_dataset)
    for _ in range(num_total_samples // per_device_batch_size):
      batch = next(data_list)
      self.assertIsNotNone(batch)


class LoadTest(absltest.TestCase):
  n_splits = 8
  data_per_split = 16

  def test_loaded_arrays_match_saved(self):
    workdir_obj = self.create_tempdir()
    workdir = workdir_obj.full_path
    pattern = os.path.join(workdir, 'array*')
    array = np.random.rand(self.n_splits * self.data_per_split)
    save_array = array.reshape(self.n_splits, self.data_per_split)
    for split in range(self.n_splits):
      path = os.path.join(workdir, 'array' + str(split))
      np.save(path, save_array[split])

    self.assertTrue(
        np.all(data_utils.load_sharded_array(pattern, 1, 0) == array))
    self.assertTrue(
        np.all(
            data_utils.load_sharded_array(pattern, 1, 1) ==
            save_array[1:].reshape(-1)))
    self.assertTrue(
        np.all(
            data_utils.load_sharded_array(pattern, self.n_splits, 0) ==
            save_array[0]))


class SaveShardedArrayTest(parameterized.TestCase):

  @parameterized.parameters(
      ((100, 3), 16, 1),
      ((100, 3), 16, 8),
      ((100,), 16, 1),
      ((100,), 16, 2),
      ((100, 3, 5), 16, 1),
      ((100, 3, 5), 16, 16),
      ((100, 3), 100, 1),
      ((100, 3), 100, 10),
      ((100,), 99, 1),
      ((100,), 99, 11),
  )
  def test_save_sharded_array(self, array_shape, num_shards, stride):
    shard_size_divisible = 3

    self.assertEqual(num_shards % stride, 0)

    arrays_per_offset = [np.random.random(array_shape) for _ in range(stride)]

    tmp_dir = self.create_tempdir()
    prefix = os.path.join(tmp_dir.full_path, 'test')

    for offset in range(stride):
      data_utils.save_sharded_array(arrays_per_offset[offset], prefix,
                                    num_shards, stride, offset,
                                    shard_size_divisible)

    loaded_array_first_dim = None
    for offset in range(stride):
      loaded_array = data_utils.load_sharded_array(
          prefix + '-?????-of-%05d' % num_shards, stride, offset)
      all_axis_except_first = list(range(1, len(array_shape)))
      sum_all_except_first_axis = np.apply_over_axes(np.sum,
                                                     np.abs(loaded_array),
                                                     all_axis_except_first)
      sum_all_except_first_axis = sum_all_except_first_axis.reshape(-1)
      is_not_pad = sum_all_except_first_axis > 0
      actual_array = loaded_array[is_not_pad]
      self.assertTrue(np.all(actual_array == arrays_per_offset[offset]))

      if loaded_array_first_dim is None:
        loaded_array_first_dim = loaded_array.shape[0]
      else:
        self.assertEqual(loaded_array_first_dim, loaded_array.shape[0])


if __name__ == '__main__':
  absltest.main()

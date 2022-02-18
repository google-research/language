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
"""Grounded SCAN Dataset."""

import functools
import os

import tensorflow as tf

TFRECORD_BUFFER_SIZE = 64 * (1 << 16)


class GSCANDataset(object):
  """Grounded SCAN dataset."""

  def __init__(self,
               data_dir,
               split,
               img_dim = 16,
               grid_size = 6,
               max_seq_len = 10,
               max_target_seq_len = 105,
               **unused_kwargs):
    self._data_dir = data_dir
    self.img_dim = img_dim
    self.grid_size = grid_size
    self.max_seq_len = max_seq_len
    self.max_target_seq_len = max_target_seq_len
    self.split = split

  def _parse(self, example):
    """Parse TFRecord example."""
    features = {
        'index':
            tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
        'token':
            tf.io.FixedLenFeature(shape=[self.max_seq_len], dtype=tf.int64),
        'txt_mask':
            tf.io.FixedLenFeature(shape=[self.max_seq_len], dtype=tf.int64),
        'target_token':
            tf.io.FixedLenFeature(
                shape=[self.max_target_seq_len], dtype=tf.int64),
        'target_txt_mask':
            tf.io.FixedLenFeature(
                shape=[self.max_target_seq_len], dtype=tf.int64),
        'image':
            tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }
    data = tf.io.parse_single_example(example, features=features)
    return data

  def preprocess(self, example):
    example['image'] = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    example['image'] = tf.reshape(
        example['image'], (self.grid_size, self.grid_size, self.img_dim))
    return example

  def as_dataset(self,
                 split,
                 shuffle_files=False,
                 read_config=None,
                 **unused_kwargs):
    """Returns a parsed and preprocessed `tf.data.Dataset`."""
    if tf.io.gfile.exists(os.path.join(self._data_dir, f'{split}.tfrecord')):
      data_filenames = os.path.join(self._data_dir, f'{split}.tfrecord')
    else:
      data_filenames = os.path.join(self._data_dir, f'{split}-*-of-*.tfrecord')
    assert tf.io.gfile.glob(data_filenames), ('No data files matched %s!' %
                                              data_filenames)
    shuffle_seed = None if read_config is None else read_config.shuffle_seed
    files = tf.data.Dataset.list_files(
        data_filenames, shuffle=shuffle_files, seed=shuffle_seed)

    dataset_type = functools.partial(
        tf.data.TFRecordDataset,
        buffer_size=TFRECORD_BUFFER_SIZE,
        num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = files.interleave(
        dataset_type,
        cycle_length=-1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

    dataset = dataset.map(
        self._parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(self.num_examples[split]))

    return dataset

  @property
  def num_examples(self):
    """The number of examples for each split."""
    if self.split == 'compositional_splits':
      example_count = {
          'train': 367933,
          'dev': 3716,
          'test': 19282,
          'visual': 37436,
          'situational_1': 88642,
          'situational_2': 16808,
          'contextual': 11460,
          'adverb_1': 112880,
          'adverb_2': 38582,
          'visual_easier': 18718,
      }
    elif self.split == 'target_length_split':
      example_count = {
          'train': 180301,
          'dev': 1821,
          'test': 37784,
          'target_lengths': 198588
      }
    elif self.split == 'spatial_relation_splits':
      example_count = {
          'train': 259088,
          'dev': 2617,
          'test': 28526,
          'visual': 62250,
          'relation': 6285,
          'referent': 30492,
          'relative_position_1': 41576,
          'relative_position_2': 41529
      }
    elif self.split == 'test':
      example_count = {'train': 8}
    else:
      raise RuntimeError(f'Unknown split {self.split}.')
    return example_count

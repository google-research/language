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
"""Utils for manipulating tf.data.Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.lookup_ops import LookupInterface


def length_mapper(keys_to_map, dims_to_map=None, suffix="_len"):
  """Creates a mapping function to augment a tf.data.Dataset with lengths.

  Suppose we have a `dataset` with outputs `str` with shape [sequence_length].
  Here is an example usage of this function:

  dataset = dataset.map(length_mapper(['str']))

  Now the dataset will also include outputs `str_len` that can be used for
  masking out padding in a model. The 'str_len' feature will be a scalar int32
  Tensor.

  Args:
    keys_to_map: List of strings that are keys for tf.string Tensors to map to
        lengths.
    dims_to_map: List of dimensions that representing the desired length to
        retrieve. Defaults to the zero'th dimension if None.
    suffix: String to append to the given keys to indicate the mapped Tensors.

  Returns:
    _mapper: A mapping function that can be used with the tf.data.Dataset API.
  """
  if dims_to_map is None:
    dims_to_map = [0 for _ in keys_to_map]
  def _mapper(dataset):
    for k, d in zip(keys_to_map, dims_to_map):
      dataset[k + suffix] = tf.shape(dataset[k])[d]
    return dataset
  return _mapper


def string_to_int_mapper(keys_to_map, mapping, num_oov_buckets=1, suffix="_id"):
  """Creates a mapping function to convert strings to ints in a tf.data.Dataset.

  For `dataset` outputs of type `str`, uses the list of strings in the given
  input `mapping` to look up the strings using tf.contrib.lookup and convert
  them to same-shape tensors of size tf.int32.

  Example:
    vocab = ['the', 'fox', 'jumped']
    dataset = dataset.map(string_to_int_mapper(['words'], mapping=vocab))
    dataset['words_id']  # <-- 'the' is mapped to 0, 'fox' to 1, etc...

  Args:
    keys_to_map: List of strings that are keys for tf.string Tensors to lookup.
    mapping: List of strings (or string tensors) to do the lookup. If the
        mapping is already a lookup table, then we directly use it.
    num_oov_buckets: Number of OOV buckets to use (default = 1).
    suffix: String to append to the given keys to indicate the mapped Tensors.

  Returns:
    _mapper: A mapping function that can be used with the tf.data.Dataset API.
  """
  if isinstance(mapping, LookupInterface):
    table = mapping
  else:
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping, num_oov_buckets=num_oov_buckets)

  def _mapper(dataset):
    for k in keys_to_map:
      dataset[k + suffix] = tf.to_int32(table.lookup(dataset[k]))
    return dataset
  return _mapper

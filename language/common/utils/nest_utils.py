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
"""Utilties for dealing with nested structures."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import numpy as np
import tensorflow.compat.v1 as tf


def add_string_feature(key, values,
                       example):
  example.features.feature[key].bytes_list.value.extend(values)


def add_int_feature(key, values,
                    example):
  example.features.feature[key].int64_list.value.extend(values)


def add_float_feature(key, values,
                      example):
  example.features.feature[key].float_list.value.extend(values)


def flat_dict_to_tf_example(inputs,
                            structure):
  """Convert a flat dictionary to a tf.Example.

  Args:
    inputs: A dictionary of flat numpy arrays.
    structure: A nested structure of placeholders that have specified shapes.

  Returns:
    example: An example with the flattened inputs.
  """
  flat_structure = nest_to_flat_dict(structure)
  tf.nest.assert_same_structure(
      nest1=flat_structure, nest2=inputs, check_types=False)
  example = tf.train.Example()
  dtype_map = {
      tf.string: add_string_feature,
      tf.int32: add_int_feature,
      tf.int64: add_int_feature,
      tf.float32: add_float_feature,
      tf.float64: add_float_feature
  }
  for k, v in inputs.items():
    placeholder = flat_structure[k]
    assert placeholder.shape == v.shape
    add_fn = dtype_map[placeholder.dtype]
    add_fn(k, v.flatten(), example)
  return example


def tf_example_to_structure(serialized_example,
                            structure):
  """Convert a serialized tf.Example into a nested structure of Tensors.

  Args:
    serialized_example: String tensor containing a serialized example.
    structure: A nested structure of placeholders that have specified shapes.

  Returns:
    features: A nested structure of Tensors consistent with `structure`.
  """
  flat_structure = nest_to_flat_dict(structure)
  dtype_map = {
      tf.string: tf.string,
      tf.int32: tf.int64,
      tf.int64: tf.int64,
      tf.float32: tf.float32,
      tf.float64: tf.float32
  }

  def _placeholder_to_feature(placeholder):
    return tf.FixedLenFeature(
        shape=np.prod(placeholder.shape, dtype=np.int32),
        dtype=dtype_map[placeholder.dtype])

  flat_feature_spec = {
      k: _placeholder_to_feature(v) for k, v in flat_structure.items()
  }
  flat_features = tf.parse_single_example(serialized_example, flat_feature_spec)
  flat_features = {
      k: tf.reshape(tf.cast(flat_features[k], v.dtype), v.shape)
      for k, v in flat_structure.items()
  }
  features = flat_dict_to_nest(flat_features, structure)
  return features


def nest_to_flat_dict(nest):
  """Convert a nested structure into a flat dictionary.

  Args:
    nest: A nested structure.

  Returns:
    flat_dict: A dictionary with strings keys that can be converted back into
               the original structure via `flat_dict_to_nest`.
  """
  flat_sequence = tf.nest.flatten(nest)
  return {str(k): v for k, v in enumerate(flat_sequence)}


def flat_dict_to_nest(flat_dict, structure):
  """Convert a nested structure into a flat dictionary.

  Args:
    flat_dict: A dictionary with string keys.
    structure: A nested structure.

  Returns:
    nest: A nested structure that inverts `nest_to_flat_dict`.
  """
  flat_sequence = [flat_dict[str(i)] for i in range(len(flat_dict))]
  return tf.nest.pack_sequence_as(
      structure=structure, flat_sequence=flat_sequence)


def assert_same(nest1, nest2):
  """Assert that both structures are equivalent.

  This function is more strict than tf.nest.assert_same_structure since it also
  requires that Tensors have the same dtype and shape.

  Args:
    nest1: A nested structure.
    nest2: A nested structure.
  """
  tf.nest.assert_same_structure(nest1, nest2)
  for t1, t2 in zip(tf.nest.flatten(nest1), tf.nest.flatten(nest2)):
    assert t1.dtype == t2.dtype
    if isinstance(t1, tf.Tensor) and isinstance(t2, tf.Tensor):
      assert t1.shape == t2.shape

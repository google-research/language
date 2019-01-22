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
"""Utilities specific to TPUs.

These functions should only be called inside the model_fn of a TPUEstimator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.utils import tensor_utils
import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function


def num_tpu_shards():
  """Get the number of TPU shards."""
  return tpu_function.get_tpu_context().number_of_shards


def shard_id():
  """Get an integer scalar Tensor indicating the index of the current shard."""
  # Prevent the TPU compiler from rewriting this part of the graph.
  with tf.control_dependencies(None):
    return tpu_ops.tpu_replicated_input(list(range(num_tpu_shards())))


def cross_shard_pad(input_tensor):
  """Cross shard pad.

  Assuming `input_tensor` is replicated over different TPU cores across the
  zeroth dimension, this creates a global tensor with unique chunks per replica.
  This function only fills in the local `input_tensor` and pads the non-local
  part of the tensor with zeros. Does not actually do any cross-shard
  communication.

  Args:
    input_tensor: <int32|float32> [local_batch_size, dim1, dim2, ...]

  Returns:
    padded_tensor: <int32|float32>
        [local_batch_size * num_shards, dim1, dim2, ...]
  """
  num_shards = num_tpu_shards()

  # [num_shards]
  local_mask = tf.equal(tf.range(num_shards), shard_id())
  local_mask = tf.cast(local_mask, input_tensor.dtype)

  tensor_shape = tensor_utils.shape(input_tensor)
  local_batch_size = tensor_shape[0]
  global_batch_size = num_shards * local_batch_size

  # [num_shards, 1, 1, ...]
  for _ in tensor_shape:
    local_mask = tf.expand_dims(local_mask, -1)

  # [num_shards, local_batch_size, input_tensor_dim1, ...]
  padded_tensor = local_mask * tf.expand_dims(input_tensor, 0)

  # [global_batch_size, input_tensor_dim1, ...]
  padded_tensor = tf.reshape(padded_tensor,
                             [global_batch_size] + tensor_shape[1:])
  return padded_tensor


def cross_shard_concat(input_tensor):
  """Concatenates all replicas of `input_tensor` across all shardss.

  Args:
    input_tensor: <int32|float32> [local_batch_size, dim1, dim2, ...]

  Returns:
    padded_tensor: <int32|float32>
        [local_batch_size * num_shards, dim1, dim2, ...]
  """
  padded_tensor = cross_shard_pad(input_tensor)
  concat_tensor = tf.contrib.tpu.cross_replica_sum(padded_tensor)
  return concat_tensor

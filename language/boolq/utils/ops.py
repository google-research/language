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
"""Some generally useful tensorflow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.data.ops import dataset_ops

# pylint: enable=g-direct-tensorflow-import


def affine(x, output_size, weight_name, bias_name=None, weight_init=None):
  """Affine transformation of the input `x`.

  Args:
    x: <float32>[..., x_dim]
    output_size: size of the last output dimension
    weight_name: Name of the weight variable to use
    bias_name: Name for the bias variable, if one should be used
    weight_init: Initializer of the weight variable

  Returns:
    transformed <float32>[..., `output_size`]
  """
  dim = x.shape.as_list()[-1]
  w = tf.get_variable(
      weight_name, (dim, output_size), tf.float32, initializer=weight_init)
  out = tf.tensordot(x, w, [[len(x.shape) - 1], [0]])
  if bias_name:
    b = tf.get_variable(
        bias_name, (output_size,),
        tf.float32,
        initializer=tf.zeros_initializer())
    for _ in range(len(out.shape) - 1):
      b = tf.expand_dims(b, 0)
    out += b
  return out


def last_dim_weighted_sum(x, weight_name, weight_init=None, keepdims=False):
  """Computes a weighted sum of the last dimension of `x`.

  Args:
    x: <float32>[..., x_dim]
    weight_name: Name of the weight variable to use
    weight_init: Initializer of the weight variable
    keepdims: Whether the output should hav an ending size one dim

  Returns:
    summed: <float32>[...] or <float32>[..., 1] iff `keepdims`
  """

  dim = x.shape.as_list()[-1]
  w = tf.get_variable(weight_name, dim, initializer=weight_init)
  out = tf.tensordot(x, w, [[len(x.shape) - 1], [0]])
  if keepdims:
    return tf.expand_dims(out, len(out.shape))
  else:
    return out


def mask_logits(vec, mask):
  """Mask `vec` in log-space.

  Elements in `vec` that are not in `mask` will be set to be very
  negative, so that when no longer in log-space (e.i., after `tf.exp(vec)`)
  their values will be very close to zero.

  Args:
    vec: <float32>[...] tensor to mask
    mask: Either a None (in which case this is a no-op), a boolean or 0/1 float
      mask that matches all, or all but the last, dimensions of `vec`, or 1-D
      integer length mask

  Raises:
    ValueError: If `mask` cannot be matched to `vec`
  Returns:
    masked: vec:<float32>[...]
  """
  if mask is None:
    return vec

  if mask.dtype == tf.int32:
    # Assume `mask` holds sequence lengths
    if len(vec.shape) not in [2, 3]:
      raise ValueError("Can't use a length mask on tensor of rank>3")
    mask = tf.sequence_mask(mask, tf.shape(vec)[1], tf.float32)
  else:
    mask = tf.to_float(mask)

  if len(mask.shape) == (len(vec.shape) - 1):
    mask = tf.expand_dims(mask, len(vec.shape) - 1)

  return vec * mask - (1 - mask) * 1E20


def _lowercase(x):
  # Specify `np.object` to avoid incorrectly returning a np.string type arr
  return np.array([w.lower() for w in x], np.object)


def lowercase_op(string_tensor):
  """Lowercase an arbitrarily sized string tensor."""

  shape = tf.shape(string_tensor)
  lc = tf.py_func(_lowercase, [tf.reshape(string_tensor, [-1])], tf.string,
                  False)
  return tf.reshape(lc, shape)


def bucket_by_quantiles(len_fn, batch_size, n_buckets, hist_bounds):
  """Dynamically bucket a `tf.data.Dataset` based on the element's length.

  Keeps track of a histogram of the input elements lengths, and yields batches
  of examples that belong to the same length quantile.

  Useful in cases where you want to bucket data, but don't know what the
  optimal bucketing ranges should be

  Args:
    len_fn: Function mapping elements in the dataset to an integer length
    batch_size: Maximum size of the output batches
    n_buckets: Number of quantiles to break the data into
    hist_bounds: List of integer bounds to use when building the histograms,
      should cover a range so that at most a single quantile of elements are
      lower/higher then the bucket range. More fine-grained buckets will make
      the histogram more precise, but adds to the computational overhead

  Raises:
    ValueError: If `hist_bounds` or `len_fn` are invalid
  Returns:
    A function that can be used with tf.data.Dataset.apply to batch a dataset
  """
  n_hist_binds = len(hist_bounds)

  if n_hist_binds < n_buckets:
    raise ValueError("Requested %d buckets, but only have %d histogram bins" %
                     (n_buckets, n_hist_binds))
  if any(hist_bounds[i] >= hist_bounds[i + 1] for i in range(n_hist_binds - 1)):
    raise ValueError("Bins must be descending")

  # Need to use `use_resource = True` to make this work correctly
  # within tf.data.Dataset
  hist_counts = tf.get_local_variable(
      "hist-counts",
      n_hist_binds + 1,
      tf.int64,
      tf.zeros_initializer(),
      use_resource=True)
  hist_bounds = tf.constant(hist_bounds, tf.int64)

  def bucket_fn(x):
    """Compute the element bucket and update the histogram."""
    ix = len_fn(x)
    if ix.dtype == tf.int32:
      ix = tf.to_int64(ix)
    elif ix.dtype != tf.int64:
      raise ValueError("Len function returned a non-int")

    adds_to_bins = tf.to_int64(tf.greater(hist_bounds, ix))
    # pad with a 1 for the "larger than all" bin
    adds_to_bins = tf.pad(adds_to_bins, [[0, 1]], constant_values=1)
    new_counts = tf.assign_add(hist_counts, adds_to_bins)
    bin_ix = n_hist_binds - tf.reduce_sum(adds_to_bins)
    # Computes the quantile based on the counts of the exammple's bucket
    bucket_ix = tf.floordiv(((n_buckets - 1) * new_counts[bin_ix]),
                            new_counts[-1])
    return bucket_ix

  def reduce_fn(_, x):
    return x.padded_batch(batch_size, dataset_ops.get_legacy_output_shapes(x))

  return tf.data.experimental.group_by_window(bucket_fn, reduce_fn, batch_size)

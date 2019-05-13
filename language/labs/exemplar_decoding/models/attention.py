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
"""This is mostly based on tf.contrib.seq2seq.

   It includes several customized features that allow to pass a mask to compute
   the attention, which is useful for the attending over exeplar baselines.
   I disabled many of the warnings, since they really come from the original
   file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from language.labs.exemplar_decoding.models.common import dimension_value
from language.labs.exemplar_decoding.models.linear import HyperDense
import numpy as np
import tensorflow as tf

# pylint: disable=g-long-ternary
# pylint: disable=unused-argument

__all__ = [
    "HyperAttention",
    "safe_cumprod",
]


def _prepare_memory(
    memory,
    memory_sequence_length,
    mask,
    check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    mask: To mask out some of the elements.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = tf.contrib.framework.nest.map_structure(
      lambda m: tf.convert_to_tensor(m, name="memory"), memory)
  if memory_sequence_length is not None:
    memory_sequence_length = tf.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))

    tf.contrib.framework.nest.map_structure(_check_dims, memory)

  seq_len_mask = tf.cast(mask,
                         tf.contrib.framework.nest.flatten(memory)[0].dtype)
  seq_len_batch_size = (dimension_value(mask.shape[0]) or tf.shape(mask)[0])

  def _maybe_mask(m, seq_len_mask):
    """Mask the sequence with m."""
    rank = m.get_shape().ndims
    rank = rank if rank is not None else tf.rank(m)
    extra_ones = tf.ones(rank - 2, dtype=tf.int32)
    m_batch_size = dimension_value(m.shape[0]) or tf.shape(m)[0]
    with tf.control_dependencies(
        [tf.assert_equal(seq_len_batch_size, m_batch_size, message="batch")]):
      seq_len_mask = tf.reshape(
          seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
      return m * seq_len_mask

  return tf.contrib.framework.nest.map_structure(
      lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score,
                      memory_sequence_length,
                      score_mask,
                      score_mask_value):
  score_mask_values = score_mask_value * tf.ones_like(score)
  return tf.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(tf.contrib.seq2seq.AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               mask=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=None,
               name=None):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      probability_fn: A `callable`.  Converts the score and previous alignments
        to probabilities. Its signature should be:
        `probabilities = probability_fn(score, state)`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      mask: To mask out some of the elements.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if (query_layer is not None and
        not isinstance(query_layer, tf.layers.Layer)):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    # if (memory_layer is not None
    #     and not isinstance(memory_layer, tf.layers.Layer)):
    #   raise TypeError(
    #       "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    self._mask = mask
    self.dtype = memory.dtype
    if not callable(probability_fn):
      raise TypeError("probability_fn must be callable, saw type: %s" %
                      type(probability_fn).__name__)
    if score_mask_value is None:
      score_mask_value = tf.dtypes.as_dtype(
          memory.dtype).as_numpy_dtype(-np.inf)
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(
                score, memory_sequence_length, mask, score_mask_value),
            prev))
    with tf.name_scope(name, "BaseAttentionMechanismInit",
                       tf.contrib.framework.nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length, mask=mask,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
      self._batch_size = (
          dimension_value(self._keys.shape[0]) or tf.shape(self._keys)[0])
      self._alignments_size = (
          dimension_value(self._keys.shape[1]) or tf.shape(self._keys)[1])

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  @property
  def state_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step.

    The default behavior is to return a tensor of all zeros.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return tf.zeros([max_time, batch_size], dtype=dtype)

  def initial_state(self, batch_size, dtype):
    """Creates the initial state values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step.

    The default behavior is to return the same output as initial_alignments.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A structure of all-zero tensors with shapes as described by `state_size`.
    """
    return self.initial_alignments(batch_size, dtype)


def _luong_score(query, keys, scale):
  """Implements Luong-style (multiplicative) scoring function.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, call this function with `scale=True`.

  Args:
    query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    scale: Whether to apply a scale to the score function.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.

  Raises:
    ValueError: If `key` and `query` depths do not match.
  """
  depth = query.get_shape()[-1]
  key_units = keys.get_shape()[-1]
  if depth != key_units:
    raise ValueError(
        "Incompatible or unknown inner dimensions between query and keys.  "
        "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
        "Perhaps you need to set num_units to the keys' dimension (%s)?"
        % (query, depth, keys, key_units, key_units))
  dtype = query.dtype

  # Reshape from [batch_size, depth] to [batch_size, 1, depth]
  # for matmul.
  query = tf.expand_dims(query, 1)

  # Inner product along the query units dimension.
  # matmul shapes: query is [batch_size, 1, depth] and
  #                keys is [batch_size, max_time, depth].
  # the inner product is asked to **transpose keys' inner shape** to get a
  # batched matmul on:
  #   [batch_size, 1, depth] . [batch_size, depth, max_time]
  # resulting in an output shape of:
  #   [batch_size, 1, max_time].
  # we then squeeze out the center singleton dimension.
  score = tf.matmul(query, keys, transpose_b=True)
  score = tf.squeeze(score, [1])

  if scale:
    # Scalar used in weight scaling
    g = tf.get_variable(
        "attention_g", dtype=dtype, initializer=tf.ones_initializer, shape=())
    score = g * score
  return score


class MyAttention(_BaseAttentionMechanism):
  """Based on wrapper of tf.contrib.seq2seq.LuongAttention.

     Allows to use a customized mask.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               mask=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="MyAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      mask: To mask out some of the elements.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is `tf.nn.softmax`. Other options include
        `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional) The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the memory layer of the attention mechanism.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = tf.nn.softmax
    if dtype is None:
      dtype = tf.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(MyAttention, self).__init__(
        query_layer=None,
        memory_layer=None,
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        mask=mask,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """

    with tf.variable_scope(None, "my_attention", [query]):
      score = _luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


class HyperAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, construct the object with parameter
  `scale=True`.
  """

  def __init__(self,
               num_units,
               mem_input,
               hps,
               memory,
               use_beam=False,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="HyperAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      mem_input: mem_input.
      hps: hyperparameters.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      use_beam: Use beam search or not.
      memory_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is `tf.nn.softmax`. Other options include
        `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional) The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the memory layer of the attention mechanism.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = tf.nn.softmax
    if dtype is None:
      dtype = tf.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(HyperAttention, self).__init__(
        query_layer=None,
        memory_layer=HyperDense(
            num_units,
            mem_input=mem_input,
            hps=hps,
            use_beam=use_beam,
            name="memory_layer",
            use_bias=False,
            dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """

    with tf.variable_scope(None, "hyper_attention", [query]):
      score = _luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def safe_cumprod(x, *args, **kwargs):
  """Computes cumprod of x in logspace using cumsum to avoid underflow.

  The cumprod function and its gradient can result in numerical instabilities
  when its argument has very small and/or zero values.  As long as the argument
  is all positive, we can instead compute the cumulative product as
  exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.

  Args:
    x: Tensor to take the cumulative product of.
    *args: Passed on to cumsum; these are identical to those in cumprod.
    **kwargs: Passed on to cumsum; these are identical to those in cumprod.
  Returns:
    Cumulative product of x.
  """
  with tf.name_scope(None, "SafeCumprod", [x]):
    x = tf.convert_to_tensor(x, name="x")
    tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
    return tf.exp(
        tf.cumsum(tf.log(tf.clip_by_value(x, tiny, 1)), *args, **kwargs))


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = tf.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = tf.matmul(expanded_alignments, attention_mechanism.values)
  context = tf.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(tf.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments, next_attention_state

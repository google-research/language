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
"""Custom attention wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


__all__ = [
    "FixedMemoryLuongAttention",
    "FixedMemoryAttentionWrapper",
]


def _fixed_memory_luong_score(query, keys, scale):
  """Implements Luong-style (multiplicative) scoring function.

  Assumes that keys have batch dimension of 1 (i.e., fixed memory bank).

  Args:
    query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[1, max_time, num_units]`.
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

  # Reshape from [1, memory_size, depth] to [memory_size, depth] for matmul.
  keys = array_ops.squeeze(keys, 0)

  # Inner product along the query units dimension.
  # matmul shapes: query is [batch_size, depth] and
  #                keys is [max_time, depth].
  # the inner product is asked to **transpose keys' inner shape** to get a
  # matmul on:
  #   [batch_size, depth] . [depth, max_time]
  # resulting in an output shape of:
  #   [batch_size, max_time].
  # we then squeeze out the center singleton dimension.
  score = math_ops.matmul(query, keys, transpose_b=True)

  if scale:
    # Scalar used in weight scaling
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=init_ops.ones_initializer, shape=())
    score = g * score
  return score


class FixedMemoryLuongAttention(attention_wrapper.LuongAttention):
  """LuongAttention that acts on a fixed memory."""

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
    with variable_scope.variable_scope(None, "luong_attention", [query]):
      score = _fixed_memory_luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)

  # Reshape from [1, memory_time, memory_size] to [memory_time, memory_size]
  memory = array_ops.squeeze(attention_mechanism.values, 0)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, memory_time]
  # memory shape is
  #   [memory_time, memory_size]
  # the matmul is over memory_time, so the output shape is
  #   [batch_size, memory_size].
  context = math_ops.matmul(alignments, memory)

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments, next_attention_state


class FixedMemoryAttentionWrapper(attention_wrapper.AttentionWrapper):
  """AttentionWrapper that works with a fixed memory array.

  Essentially, ensures that memory batch size is always 1.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None,
               attention_layer=None):
    super(FixedMemoryAttentionWrapper, self).__init__(
        cell=cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=attention_layer_size,
        alignment_history=True,  # used to compute logits.
        cell_input_fn=cell_input_fn,
        output_attention=output_attention,
        initial_cell_state=initial_cell_state,
        name=name,
        attention_layer=attention_layer)

  def _batch_size_checks(self, batch_size, error_message):
    del batch_size  # Unused.
    # Attention batch size must be always 1.
    return [check_ops.assert_equal(1, attention_mechanism.batch_size,
                                   message=error_message)
            for attention_mechanism in self._attention_mechanisms]

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      TypeError: If `state` is not an instance of `AttentionWrapperState`.
    """
    if not isinstance(state, attention_wrapper.AttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = "Memory batch size must be equal to 1."
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    if self._is_multi:
      previous_attention_state = state.attention_state
      previous_alignment_history = state.alignment_history
    else:
      previous_attention_state = [state.attention_state]
      previous_alignment_history = [state.alignment_history]

    all_alignments = []
    all_attentions = []
    all_attention_states = []
    maybe_all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      attention, alignments, next_attention_state = _compute_attention(
          attention_mechanism, cell_output, previous_attention_state[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_attention_states.append(next_attention_state)
      all_alignments.append(alignments)
      all_attentions.append(attention)
      maybe_all_histories.append(alignment_history)

    attention = array_ops.concat(all_attentions, 1)
    next_state = attention_wrapper.AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        attention_state=self._item_or_tuple(all_attention_states),
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(maybe_all_histories))

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state

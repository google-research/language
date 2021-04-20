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
"""A fork of BERT's transformer implementation that performs local attention."""



from language.canine import bert_modeling
from language.canine import tensor_contracts as tc
import tensorflow.compat.v1 as tf


@tc.contract(
    tc.Require("a", shape=["batch", "seq", "dim"]),
    tc.Require("b", shape=["batch", "seq", "dim"]),
    tc.NamedDim("batch", "a", 0),
    tc.NamedDim("seq", "a", 1),
    tc.NamedDim("dim", "a", 2))
def _safe_add(a, b):
  return a + b


@tc.contract(
    tc.Require("from_tensor", shape=["batch", "from_len", "from_dim"]),
    tc.Require("to_tensor", shape=["batch", "to_len", "to_dim"]),
    tc.Require("attention_mask", shape=["batch", "from_len", "to_len"]),
    tc.Ensure(
        tc.RESULT,
        shape=[
            "batch", "from_len",
            tc.Unchecked("heads"),
            tc.Unchecked("size_per_head")
        ]), tc.NamedDim("batch", "from_tensor", 0),
    tc.NamedDim("from_len", value_of="from_seq_length"),
    tc.NamedDim("from_dim", "from_tensor", 2),
    tc.NamedDim("to_len", value_of="to_seq_length"),
    tc.NamedDim("to_dim", "to_tensor", 2))
def local_attention_layer(from_tensor,
                          to_tensor,
                          from_seq_length,
                          to_seq_length,
                          attention_mask,
                          num_attention_heads = 1,
                          size_per_head = 512,
                          query_act = None,
                          key_act = None,
                          value_act = None,
                          attention_probs_dropout_prob = 0.0,
                          initializer_range = 0.02,
                          batch_size = None,
                          always_attend_to_first_position = True,
                          first_position_attends_to_all = True,
                          attend_from_chunk_width = 128,
                          attend_from_chunk_stride = 128,
                          attend_to_chunk_width = 128,
                          attend_to_chunk_stride = 128):
  """A fork of BERT's `attention_layer` that performs local attention.

  This attention is local in that attention happens only within each block
  (as defined by the length of the stides).

  Function parameters specific to local attention (i.e. added from BERT's
  `attention_layer`) are at the bottom of the argument list.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    from_seq_length: Length of the padded 'from' sequence in this model.
    to_seq_length: Length of the padded 'to' sequence in this model.
    attention_mask: int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    always_attend_to_first_position: Should all blocks be able to attend to the
      `to_tensor`'s first position (e.g. a [CLS] position)?
    first_position_attends_to_all: Should the `from_tensor`'s first position
      be able to attend to all positions within the `from_tensor`?
    attend_from_chunk_width: The width of each block-wise chunk in
      `from_tensor`.
    attend_from_chunk_stride: The number of elements to skip when moving to the
      next block in `from_tensor`.
    attend_to_chunk_width: The width of each block-wise chunk in `to_tensor`.
    attend_to_chunk_stride: The number of elements to skip when moving to the
      next block in `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  if attend_from_chunk_width < attend_from_chunk_stride:
    raise ValueError("`attend_from_chunk_width` < `attend_from_chunk_stride`"
                     "would cause sequence positions to get skipped.")
  if attend_to_chunk_width < attend_to_chunk_stride:
    raise ValueError("`attend_to_chunk_width` < `attend_to_chunk_stride`"
                     "would cause sequence positions to get skipped.")

  # Create chunks that we will attend *from* and then concatenate them.
  from_chunks = []
  if first_position_attends_to_all:
    from_chunks.append((0, 1))
    # We must skip this first position so that our output sequence is the
    # correct length (this matters in the *from* sequence only).
    from_start = 1
  else:
    from_start = 0
  for chunk_start in range(from_start, from_seq_length,
                           attend_from_chunk_stride):
    chunk_end = min(from_seq_length, chunk_start + attend_from_chunk_width)
    from_chunks.append((chunk_start, chunk_end))

  # Determine the chunks that will will attend *to*.
  to_chunks = []
  if first_position_attends_to_all:
    to_chunks.append((0, to_seq_length))
  for chunk_start in range(0, to_seq_length, attend_to_chunk_stride):
    chunk_end = min(to_seq_length, chunk_start + attend_to_chunk_width)
    to_chunks.append((chunk_start, chunk_end))

  if len(from_chunks) != len(to_chunks):
    raise ValueError(
        f"Expected to have same number of `from_chunks` ({from_chunks}) and "
        f"`to_chunks` ({from_chunks}). Check strides.")

  # TODO(jhclark): Can we save a bit of extra compute by slicing the Q/K/V
  # projected versions of these instead of recomputing those projections?
  # This only helps when the Q stride isn't the same as the K/V stride.
  # Length of `attention_output_chunks` and therefore `attention_output` is
  # determined by `from_chunks` to ensure correctness. The correspondence with
  # `to_chunks` is somewhat best effort. We need to do more to enforce this.
  attention_output_chunks = []
  for (from_start, from_end), (to_start, to_end) in zip(from_chunks, to_chunks):
    from_tensor_chunk = from_tensor[:, from_start:from_end, :]
    to_tensor_chunk = to_tensor[:, to_start:to_end, :]
    # `attention_mask`: <float>[batch_size, from_seq, to_seq]
    # `attention_mask_chunk`: <float>[batch_size, from_seq_chunk, to_seq_chunk]
    attention_mask_chunk = (
        attention_mask[:, from_start:from_end, to_start:to_end])
    if always_attend_to_first_position:
      cls_attention_mask = attention_mask[:, from_start:from_end, 0:1]
      attention_mask_chunk = tf.concat(
          [cls_attention_mask, attention_mask_chunk], axis=2)

      cls_position = to_tensor[:, 0:1, :]
      to_tensor_chunk = tf.concat([cls_position, to_tensor_chunk], axis=1)
    attention_output_chunk = bert_modeling.attention_layer(
        from_tensor=from_tensor_chunk,
        to_tensor=to_tensor_chunk,
        attention_mask=attention_mask_chunk,
        num_attention_heads=num_attention_heads,
        size_per_head=size_per_head,
        query_act=query_act,
        key_act=key_act,
        value_act=value_act,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
        batch_size=batch_size)
    attention_output_chunks.append(attention_output_chunk)
  return tf.concat(attention_output_chunks, axis=1)


# TODO(jhclark): Write dynamic contract for `input_kv_tensor` case.
@tc.contract(
    tc.Require("input_tensor", shape=["batch", "seq_len", "dim"],
               static_dims=[1]),
    tc.Require("attention_mask", shape=["batch", "seq_len", "seq_len"]),
    tc.Require("input_kv_tensor", optional=True,
               shape=["batch", tc.Unchecked("kv_len"), tc.Unchecked("dim")]),
    tc.Require("init_kv_attention_mask", optional=True,
               shape=["batch", "seq_len", tc.Unchecked("kv_len")]),
    tc.Ensure(tc.RESULT, shape=["batch", "seq_len", "dim"]),
    tc.NamedDim("batch", "input_tensor", 0),
    tc.NamedDim("seq_len", "input_tensor", 1),
    tc.NamedDim("dim", value_of="hidden_size"))
def local_transformer_model(input_tensor,
                            attention_mask,
                            input_kv_tensor = None,
                            init_kv_attention_mask = None,
                            hidden_size = 768,
                            num_hidden_layers = 12,
                            num_attention_heads = 12,
                            intermediate_size = 3072,
                            intermediate_act_fn = None,
                            hidden_dropout_prob = 0.1,
                            attention_probs_dropout_prob = 0.1,
                            initializer_range = 0.02,
                            do_return_all_layers = False,
                            num_layers_to_update = None,
                            always_attend_to_first_position = True,
                            first_position_attends_to_all = True,
                            attend_from_chunk_width = 128,
                            attend_from_chunk_stride = 128,
                            attend_to_chunk_width = 128,
                            attend_to_chunk_stride = 128,
                            init_attend_to_chunk_width = 128,
                            init_attend_to_chunk_stride = 128):
  """Fork of BERT's `transformer_model` that performs local attention.

  This attention is local in that attention happens only within each block
  (as defined by the length of the stides).

  Function parameters specific to local attention (i.e. added from BERT's
  `attention_layer`) are at the bottom of the argument list.

  IMPORTANT: Both `input_tensor` and `init_kv_tensor` must have a static
  sequence length dimension, such that it can be extracted as a python integer
  at graph-building time. Dynamic sequence lengths are not supported by
  `local_transformer_model` and `local_attention_layer` (because doing so would
  greatly limit XLA's ability to create a highly optimized program on TPU).

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    input_kv_tensor: float Tensor of shape
      [batch_size, seq_length_kv, seq_dim_kv]. If specified, this will be used
      for the initial layer of keys and values for self-attention.
      `input_tensor` will still be used for queries and resnet connections.
    init_kv_attention_mask: (optional) int32 Tensor of shape [batch_size,
      seq_length, seq_length_kv], with 1 for positions that can be attended to
      and 0 in positions that should not be. i.e. It indicates which items we
      can attend *from* in `input_tensor` (`seq_length`) and *to* in
      `input_kv_tensor` (`seq_length_kv`).
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    num_layers_to_update: (optional) Number of layers to update during
      training (a `tf.stop_gradient` is applied beyond this point). This is
      useful for gradual layer unfreezing during fine-tuning to prevent
      catastrophic forgetting.
    always_attend_to_first_position: Should all blocks be able to attend to the
      `to_tensor`'s first position (e.g. a [CLS] position)?
    first_position_attends_to_all: Should the query ("from") tensor's first
      position be able to attend to all positions within the key-value tensor?
    attend_from_chunk_width: The width of each block-wise chunk in
      the query ("from") tensor.
    attend_from_chunk_stride: The number of elements to skip when moving to the
      next block in the query ("from") tensor.
    attend_to_chunk_width: The width of each block-wise chunk in the key-value
      ("to") tensor.
    attend_to_chunk_stride: The number of elements to skip when moving to the
      next block in the key-value ("to") tensor.
    init_attend_to_chunk_width: `attend_to_chunk_width` for first layer when
      `init_kv_tensor` is specified.
    init_attend_to_chunk_stride: `attend_to_chunk_stride` for first layer when
      `init_kv_tensor` is specified.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """

  if num_hidden_layers == 0:
    return input_tensor

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = bert_modeling.get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  from_shape = bert_modeling.get_shape_list(input_tensor, expected_rank=3)
  # This is enforced as a static int in the contract above.
  from_seq_length: int = from_shape[1]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  prev_output = input_tensor
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output
      if layer_idx == 0 and input_kv_tensor is not None:
        layer_kv_input = input_kv_tensor
        layer_attention_mask = init_kv_attention_mask
        layer_attend_to_chunk_width = init_attend_to_chunk_width
        layer_attend_to_chunk_stride = init_attend_to_chunk_stride
        if init_kv_attention_mask is None:
          raise ValueError("`init_kv_attention_mask` must be specified when "
                           "`input_kv_tensor` is specified.")
      else:
        layer_kv_input = layer_input
        layer_attention_mask = attention_mask
        layer_attend_to_chunk_width = attend_to_chunk_width
        layer_attend_to_chunk_stride = attend_to_chunk_stride

      to_shape = bert_modeling.get_shape_list(layer_kv_input, expected_rank=3)
      to_seq_length: int = to_shape[1]
      assert isinstance(to_seq_length, int)

      with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("self"):
          attention_output = local_attention_layer(
              from_tensor=layer_input,  # Queries.
              to_tensor=layer_kv_input,
              from_seq_length=from_seq_length,
              to_seq_length=to_seq_length,
              attention_mask=layer_attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              # Parameters specific to local attention:
              always_attend_to_first_position=always_attend_to_first_position,
              first_position_attends_to_all=first_position_attends_to_all,
              attend_from_chunk_width=attend_from_chunk_width,
              attend_from_chunk_stride=attend_from_chunk_stride,
              attend_to_chunk_width=layer_attend_to_chunk_width,
              attend_to_chunk_stride=layer_attend_to_chunk_stride)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = bert_modeling.dense_layer_3d_proj(
              attention_output, hidden_size, num_attention_heads,
              attention_head_size,
              bert_modeling.create_initializer(initializer_range), None,
              "dense")
          attention_output = bert_modeling.dropout(attention_output,
                                                   hidden_dropout_prob)
          attention_output = bert_modeling.layer_norm(
              _safe_add(attention_output, layer_input))

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = bert_modeling.dense_layer_2d(
            attention_output, intermediate_size,
            bert_modeling.create_initializer(initializer_range),
            intermediate_act_fn, "dense")

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = bert_modeling.dense_layer_2d(
            intermediate_output, hidden_size,
            bert_modeling.create_initializer(initializer_range), None, "dense")
        layer_output = bert_modeling.dropout(layer_output, hidden_dropout_prob)
        layer_output = bert_modeling.layer_norm(
            _safe_add(layer_output, attention_output))

        if num_layers_to_update is not None:
          num_layers_remaining = num_hidden_layers - layer_idx - 1
          if num_layers_remaining == num_layers_to_update:
            layer_output = tf.stop_gradient(layer_output)

        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    return all_layer_outputs
  else:
    return all_layer_outputs[-1]

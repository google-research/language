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
"""Common building blocks for TF graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_shape_list(tensor):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def embedding_lookup(embedding_table, input_ids, use_one_hot_embeddings=False):
  """Looks up embeddings for id tensor.

  Args:
    embedding_table: float Tensor of shape [vocab_size, embedding_size]
    input_ids: int32 Tensor of shape [batch_size, seq_length] or [batch_size,
      seq_length, attributes_size] containing ids.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size] or
    [batch_size, seq_length, attributes_size, embedding_size].
  """
  if not use_one_hot_embeddings:
    return tf.nn.embedding_lookup(embedding_table, input_ids)

  vocab_size, embedding_size = embedding_table.get_shape()
  flat_input_ids = tf.reshape(input_ids, [-1])
  one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
  output = tf.matmul(one_hot_input_ids, embedding_table)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output, input_shape + [embedding_size])
  return output


def linear_transform(x, output_size, scope, bias=False, input_size=None):
  """Simple linear transform of x.

  Args:
    x: <float>[batch_size, length, input_size]
    output_size: Integer specifying output size.
    scope: String name for variable scope.
    bias: If True, adds a learned bias term.
    input_size: Explicitly specify input_size if not set as static shape.

  Returns:
    <float>[batch_size, length, output_size]
  """
  input_size = input_size or x.get_shape()[-1]
  with tf.variable_scope(scope):
    batch_size = tf.shape(x)[0]
    weights = tf.get_variable("weights", shape=(input_size, output_size))
    weights = tf.expand_dims(weights, 0)
    weights = tf.tile(weights, [batch_size, 1, 1])
    x = tf.matmul(x, weights)
    if bias:
      bias = tf.get_variable(
          "bias", shape=(output_size), initializer=tf.zeros_initializer())
      x += bias
    return x


def apply_norm(x, epsilon=1e-6):
  """Applies layer normalization to x.

  Based on "Layer Normalization":
  https://arxiv.org/abs/1607.06450

  Args:
    x: <float>[..., input_size]
    epsilon: Used to avoid division by 0.

  Returns:
    <float>[..., input_size]
  """
  input_size = x.get_shape()[-1]
  with tf.variable_scope("layer_norm", values=[x]):
    scale = tf.get_variable(
        "layer_norm_scale", [input_size], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [input_size], initializer=tf.zeros_initializer())
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    result = norm_x * scale + bias
    return result


def layer_postprocess(layer_input, layer_output, model_params, perform_dropout):
  """Applies dropout, residual connection, and layer normalization.

  Based on layer postprocessing from "Attention Is All You Need":
  https://arxiv.org/abs/1706.03762

  Args:
    layer_input: <float>[batch_size, length, emb_size]
    layer_output: <float>[batch_size, length, emb_size]
    model_params: ModelParameters proto.
    perform_dropout: If True, perform dropout. Should be True for training.

  Returns:
    <float>[batch_size, length, emb_size]
  """
  x = layer_output
  # Apply dropout if training.
  if perform_dropout:
    x = tf.nn.dropout(
        x, keep_prob=(1.0 - model_params.training_options.layer_dropout_rate))
  # Residual connection.
  x = x + layer_input
  # Layer normalization.
  x = apply_norm(x)
  return x


def _split_heads(x, num_heads):
  """Split dimension 3 into multiple heads.

  Attempts to preserve static shape information.

  Args:
    x: a Tensor with shape [batch, length, emb_size]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, emb_size / num_heads]
  """
  old_shape = x.get_shape().dims
  new_shape = old_shape[:-1] + [num_heads] + [old_shape[-1] // num_heads]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [num_heads, -1]], 0))
  ret.set_shape(new_shape)
  return tf.transpose(ret, [0, 2, 1, 3])


def _combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, emb_size / num_heads]

  Returns:
    a Tensor with shape [batch, length, emb_size]
  """
  x = tf.transpose(x, [0, 2, 1, 3])
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def _prepare_attention_inputs(query_antecedent, memory_antecedent, query_size,
                              memory_size, num_heads):
  """Prepare multi-head attention inputs.

  Args:
    query_antecedent: <float>[batch_size, query_length, query_input_size]
    memory_antecedent: <float>[batch_size, memory_length, memory_input_size]
    query_size: Integer specifying number of dimensions for q.
    memory_size: Integer specifying number of deimsnions for k and v.
    num_heads: Integer specifying number of attention heads.

  Returns:
    q: <float>[batch_size, num_heads, query_length, query_size / num_heads]
    k: <float>[batch_size, num_heads, memory_length, memory_size / num_heads]
    v: <float>[batch_size, num_heads, memory_length, memory_size / num_heads]
  """
  q = linear_transform(query_antecedent, query_size, "q_transform")
  k = linear_transform(memory_antecedent, memory_size, "k_transform")
  v = linear_transform(memory_antecedent, memory_size, "v_transform")
  q = _split_heads(q, num_heads)
  k = _split_heads(k, num_heads)
  v = _split_heads(v, num_heads)
  # Scale query inputs.
  key_depth_per_head = query_size // num_heads
  q *= key_depth_per_head**-0.5
  return q, k, v


def _edge_vectors_attention_inner(x,
                                  y,
                                  adjacency_matrix,
                                  num_labels,
                                  transpose,
                                  name,
                                  use_one_hot_embeddings=False):
  """Edge vector attention inner calculation.

  Args:
    x: <float>[batch_size, heads, length, length or emb_size].
    y: <float>[batch_size, heads, length, emb_size].
    adjacency_matrix: <int>[batch, length, length]
    num_labels: Number of unique labels in adjacency matrix.
    transpose: Whether to transpose inner matrices of y and z. Should be true if
      last dimension of x is emb_size, not length.
    name: Name for variable scope.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    <float>[batch_size, heads, length, length or emb_size].
  """
  with tf.variable_scope(name, values=[x, y]):
    emb_size = y.get_shape()[-1]
    xy = tf.matmul(x, y, transpose_b=transpose)
    # Generates embedding for each relative position.
    embeddings_table = tf.get_variable("embeddings", [num_labels, emb_size])
    graph_embeddings = embedding_lookup(embeddings_table, adjacency_matrix,
                                        use_one_hot_embeddings)
    # embeddings is <float>[batch, length, length, emb_size].
    x_t = tf.transpose(x, [0, 2, 1, 3])
    # x_t is <float>[batch, length, heads, length or emb_size].
    xz = tf.matmul(x_t, graph_embeddings, transpose_b=transpose)
    # xz is <float>[batch, length, heads, length or emb_size].
    xz_t = tf.transpose(xz, [0, 2, 1, 3])
    # xz_t is <float>[batch, heads, length, length or emb_size].
    return xy + xz_t


def graph_attention(query_antecedent,
                    memory_antecedent,
                    bias,
                    model_size,
                    num_heads,
                    adjacency_matrix,
                    num_labels,
                    use_one_hot_embeddings=False):
  """Multihead attention with edge vectors.

  Args:
    query_antecedent: <float>[batch_size, length, emb_size]
    memory_antecedent: <float>[batch_size, length, emb_size]
    bias: <float>[?, ?, length, length] tensor that can be broadcast to have
      shape [batch_size, num_heads, length, length].
    model_size: Integer specifying number of dimensions.
    num_heads: Integer specifying number of attention heads.
    adjacency_matrix: <int>[batch_size, length, length] specifying edge labels.
    num_labels: Integer specifying number of unique edge labels.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    <float>[batch_size, length, model_size]
  """
  q, k, v = _prepare_attention_inputs(query_antecedent, memory_antecedent,
                                      model_size, model_size, num_heads)
  logits = _edge_vectors_attention_inner(
      q,
      k,
      adjacency_matrix,
      num_labels,
      transpose=True,
      name="keys",
      use_one_hot_embeddings=use_one_hot_embeddings)
  logits += bias
  weights = tf.nn.softmax(logits, name="attention_weights")
  x = _edge_vectors_attention_inner(
      weights,
      v,
      adjacency_matrix,
      num_labels,
      transpose=False,
      name="values",
      use_one_hot_embeddings=use_one_hot_embeddings)
  # Combine heads and perform output transformation.
  x = _combine_heads(x)
  x = linear_transform(x, model_size, "output_transform")
  return x


def attention(query_antecedent, memory_antecedent, bias, model_size, num_heads):
  """Multihead dot-product attention.

  Args:
    query_antecedent: <float>[batch_size, query_length, query_emb_size]
    memory_antecedent: <float>[batch_size, memory_length, memory_emb_size]
    bias: <float>[?, ?, ?, ?] tensor that can be broadcast to have shape
      [batch_size, num_heads, query_length, memory_length].
    model_size: Integer specifying number of dimensions.
    num_heads: Integer specifying number of attention heads.

  Returns:
    <float>[batch_size, length, model_size]
  """
  q, k, v = _prepare_attention_inputs(query_antecedent, memory_antecedent,
                                      model_size, model_size, num_heads)
  # q is <float>[batch_size, num_heads, query_length, model_size / num_heads]
  # k, v <float>[batch_size, num_heads, memory_length, model_size / num_heads]
  logits = tf.matmul(q, k, transpose_b=True)
  # logits is <float>[batch_size, num_heads, query_length, memory_length]
  logits += bias
  weights = tf.nn.softmax(logits, name="attention_weights")
  x = tf.matmul(weights, v)
  # Combine heads and perform output transformation.
  x = _combine_heads(x)
  x = linear_transform(x, model_size, "output_transform")
  return x


def attention_bias_lower_triangle(length):
  """Create a bias tensor to be added to attention logits.

  Masked elements will have an effective negative-infinity value.

  Args:
    length: Integer specifying maximum sequence length in batch.

  Returns:
    <float>[length, length]
  """
  # First, create a sequence mask, e.g.:
  # [1, 0, ..., 0]
  # ...
  # [1, 1, ..., 1]
  sequence_mask = tf.sequence_mask(tf.range(1, length + 1), length)
  # Invert to transform to attention biases.
  attention_bias = tf.to_float(tf.logical_not(sequence_mask)) * -1e9
  return attention_bias


def attention_bias_ignore_padding(source_len, max_length):
  """Create a bias tensor to be added to attention logits.

  Out-of-range elements will have an effective negative-infinity value.

  Args:
    source_len: <int>[batch_size]
    max_length: Integer specifying maxmimum sequence length in batch.

  Returns:
    <float>[batch_size, 1, 1, max_length]
  """
  memory_padding = tf.to_float(
      tf.logical_not(tf.sequence_mask(source_len, maxlen=max_length)))
  ret = memory_padding * -1e9
  # Expand so tensor can be broadcast across heads and query length.
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def ff_layer(x, hidden_size, output_size, nonlinearity=tf.nn.relu):
  """Single hidden-layer feed-forward network.

  Args:
    x: <float>[batch_size, length, input_size]
    hidden_size: Integer number of dimensions for hidden layer.
    output_size: Integer number of dimensions for output.
    nonlinearity: Function to use for non-linearity.

  Returns:
    <float>[batch_size, length, output_size]
  """
  x = linear_transform(x, hidden_size, "ffn_1", bias=True)
  x = nonlinearity(x)
  x = linear_transform(x, output_size, "ffn_2", bias=True)
  return x


def relative_positions_adjacency_matrix(length, batch_size,
                                        max_relative_position):
  """Generates batch of relative position matrices.

  Args:
    length: Integer specifying maximum sequence length in batch.
    batch_size: Integer specifying batch size.
    max_relative_position: Integer specifying the distance at which the relative
      position betwen targets is clipped.

  Returns:
    <int>[batch_size, length, length]
  """
  range_vec = tf.range(length)
  range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
  distance_mat = range_mat - tf.transpose(range_mat)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  shifted_mat = distance_mat_clipped + max_relative_position

  # Tile to batch_size, duplicating the same adjacency matrix for each target.
  shifted_mat = tf.expand_dims(shifted_mat, 0)
  shifted_mat = tf.tile(shifted_mat, [batch_size, 1, 1])

  return shifted_mat

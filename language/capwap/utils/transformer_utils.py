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
"""Utilities for computing Transformer model for image captioning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math

from bert import modeling
from language.capwap.utils import tensor_utils
from language.capwap.utils import text_utils
from tensor2tensor.utils import beam_search
import tensorflow.compat.v1 as tf

TransformerCache = collections.namedtuple("TransformerCache",
                                          ["keys", "values"])

DecodeOutput = collections.namedtuple("DecodeOutput",
                                      ["token_ids", "mask", "scores"])

DecodeState = collections.namedtuple(
    "DecodeState", ["encoder_cache", "encoder_cache_mask", "output_cache"])

# ------------------------------------------------------------------------------
#
# Transformer container classes.
#
# ------------------------------------------------------------------------------


class TransformerConfig(object):
  """Configuration for TransformerModel."""

  def __init__(
      self,
      vocab_size,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072,
      hidden_act="gelu",
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_positions=512,
      max_segments=3,
      max_conditions=64,
      max_image_regions=256,
      initializer_range=0.02,
  ):
    """Constructs TransformerConfig."""
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_positions = max_positions
    self.max_segments = max_segments
    self.max_conditions = max_conditions
    self.max_image_regions = max_image_regions
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `TransformerConfig` from a Python dict of parameters."""
    config = TransformerConfig(vocab_size=None)
    for key, value in json_object.items():
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `TransformerConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dict."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TransformerModel(object):
  """Transformer model for image captioning container."""
  # Segment embedding types.
  CAP = 0
  Q = 1
  A = 2
  IMG = 3

  def __init__(self, config, is_training, scope_prefix=""):
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    self._config = config
    self._initializer = modeling.create_initializer(config.initializer_range)
    self.scope_prefix = scope_prefix + "/" if scope_prefix else ""
    with tf.variable_scope(self.scope_prefix + "bert/embeddings"):
      self._embedding_table = tf.get_variable(
          name="word_embeddings",
          shape=[config.vocab_size, config.hidden_size],
          initializer=self.initializer)
      self._segment_table = tf.get_variable(
          name="segment_embeddings",
          shape=[config.max_segments, config.hidden_size],
          initializer=self.initializer)
      self._position_table = tf.get_variable(
          name="position_embeddings",
          shape=[config.max_positions, self.config.hidden_size],
          initializer=self.initializer)
      self._condition_position_table = tf.get_variable(
          name="condition_position_embeddings",
          shape=[config.max_conditions, self.config.hidden_size],
          initializer=self.initializer)
      self._image_region_table = tf.get_variable(
          name="image_region_embeddings",
          shape=[config.max_image_regions, self.config.hidden_size],
          initializer=self.initializer)
      self._image_order_table = tf.get_variable(
          name="image_order_embeddings",
          shape=[config.max_image_regions, self.config.hidden_size],
          initializer=self.initializer)

  @property
  def config(self):
    return self._config

  @property
  def embedding_table(self):
    return self._embedding_table

  @property
  def segment_table(self):
    return self._segment_table

  @property
  def position_table(self):
    return self._position_table

  @property
  def condition_position_table(self):
    return self._condition_position_table

  @property
  def image_region_table(self):
    return self._image_region_table

  @property
  def image_order_table(self):
    return self._image_order_table

  @property
  def initializer(self):
    return self._initializer

  def compute_transformer(
      self,
      input_ids,
      input_segment_id,
      input_positions,
      attention_mask,
      input_cache=None,
      reuse=None,
      conditional=False,
  ):
    """Build the full text transformer."""
    with tf.variable_scope(self.scope_prefix + "transformer", reuse=reuse):
      with tf.variable_scope("embeddings"):
        token_emb = tf.gather(self.embedding_table, input_ids)
        segment_embed = tf.gather(self.segment_table, input_segment_id)
        if conditional:
          position_table = self.condition_position_table
        else:
          position_table = self.position_table
        position_emb = tf.gather(position_table, input_positions)
        input_emb = token_emb + segment_embed + position_emb
        input_emb = modeling.layer_norm_and_dropout(
            input_emb, self.config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        sequence_output, output_cache = compute_transformer(
            input_tensor=input_emb,
            attention_mask=attention_mask,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(self.config.hidden_act),
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=(
                self.config.attention_probs_dropout_prob),
            initializer_range=self.config.initializer_range,
            input_cache=input_cache)
      return sequence_output, output_cache

  def compute_image_transformer(
      self,
      input_ids,
      input_image,
      input_image_mask,
      input_positions,
      reuse=None,
  ):
    """Build the image transformer."""
    with tf.variable_scope(self.scope_prefix + "transformer", reuse=reuse):
      with tf.variable_scope("bridge"):
        image_emb = tf.layers.dense(
            inputs=input_image,
            units=self.config.hidden_size,
            activation=tf.nn.relu,
            kernel_initializer=modeling.create_initializer(
                self.config.initializer_range),
            reuse=reuse)

      with tf.variable_scope("embeddings"):
        input_emb = tf.gather(self.embedding_table, input_ids)
        image_emb = tf.concat([input_emb, image_emb], axis=1)
        batch_size = tensor_utils.shape(image_emb, 0)
        sequence_length = tensor_utils.shape(image_emb, 1)
        position_emb = tf.gather(self.image_region_table, input_positions)
        position_emb = tf.pad(position_emb, [[0, 0], [1, 0], [0, 0]])
        input_order = tf.range(tensor_utils.shape(image_emb, 1))
        input_order = tf.tile(
            tf.expand_dims(input_order, 0),
            [tensor_utils.shape(image_emb, 0), 1])
        order_emb = tf.gather(self.image_order_table, input_order)
        input_segment_id = tf.fill([batch_size, sequence_length], self.IMG)
        segment_emb = tf.gather(self.segment_table, input_segment_id)
        input_emb = image_emb + position_emb + order_emb + segment_emb
        input_emb = modeling.layer_norm_and_dropout(
            input_emb, self.config.hidden_dropout_prob)

      with tf.variable_scope("image/encoder"):
        sequence_output, output_cache = compute_transformer(
            input_tensor=input_emb,
            attention_mask=tf.expand_dims(input_image_mask, 1),
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(self.config.hidden_act),
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=(
                self.config.attention_probs_dropout_prob),
            initializer_range=self.config.initializer_range,
            input_cache=None)
      return sequence_output, output_cache

  def compute_logits(self, target_emb, reuse=None):
    """Compute logits for word prediction."""
    with tf.variable_scope(self.scope_prefix + "cls/predictions", reuse=reuse):
      with tf.variable_scope("transform"):
        target_emb = tf.layers.dense(
            target_emb,
            units=self.config.hidden_size,
            activation=modeling.get_activation(self.config.hidden_act),
            kernel_initializer=self.initializer)
        target_emb = modeling.layer_norm(target_emb)
      output_bias = tf.get_variable(
          "output_bias",
          shape=[self.config.vocab_size],
          initializer=tf.zeros_initializer())
    logits = tf.matmul(target_emb, self.embedding_table, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits


# ------------------------------------------------------------------------------
#
# Transformer helper functions.
#
# ------------------------------------------------------------------------------


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   size_per_head,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    num_attention_heads: Number of attention heads.
    size_per_head: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """

  last_dim = modeling.get_shape_list(input_tensor)[-1]

  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[last_dim, num_attention_heads * size_per_head],
        initializer=initializer)
    w = tf.reshape(w, [last_dim, num_attention_heads, size_per_head])
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * size_per_head],
        initializer=tf.zeros_initializer)
    b = tf.reshape(b, [num_attention_heads, size_per_head])
    ret = tf.einsum("abc,cde->abde", input_tensor, w)
    ret += b
    if activation is not None:
      return activation(ret)
    else:
      return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        num_attention_heads,
                        head_size,
                        initializer,
                        activation,
                        name=None):
  """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    num_attention_heads: The size of output dimension.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  head_size = hidden_size // num_attention_heads
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, hidden_size],
        initializer=initializer)
    w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
    b = tf.get_variable(
        name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)

  ret = tf.einsum("BFNH,NHD->BFD", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 2D kernel.

  Args:
    input_tensor: Float tensor with rank 3.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  last_dim = modeling.get_shape_list(input_tensor)[-1]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel", shape=[last_dim, output_size], initializer=initializer)
    b = tf.get_variable(
        name="bias", shape=[output_size], initializer=tf.zeros_initializer)

  ret = tf.einsum("abc,cd->abd", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def compute_attention_mask(token_mask, input_mask):
  """Compute attention mask."""
  batch_size = tensor_utils.shape(token_mask, 0)
  num_tokens = tensor_utils.shape(token_mask, 1)
  token_to_token = tf.ones([batch_size, num_tokens, num_tokens], dtype=tf.int32)
  token_to_token = tf.matrix_band_part(token_to_token, -1, 0)
  if input_mask is not None:
    token_to_input = tf.expand_dims(input_mask, 1)
    token_to_input = tf.tile(token_to_input, [1, num_tokens, 1])
    attention_mask = tf.concat([token_to_input, token_to_token], axis=-1)
  else:
    attention_mask = token_to_token
  return attention_mask


def attention_layer(
    from_tensor,
    to_tensor,
    attention_mask,
    num_attention_heads,
    size_per_head,
    attention_probs_dropout_prob,
    initializer_range,
    input_cache=None,
):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`."""

  def _project(t, name):
    return dense_layer_3d(
        input_tensor=t,
        num_attention_heads=num_attention_heads,
        size_per_head=size_per_head,
        initializer=modeling.create_initializer(initializer_range),
        activation=None,
        name=name)

  query_layer = _project(from_tensor, "query")
  key_layer = _project(to_tensor, "key")
  value_layer = _project(to_tensor, "value")
  output_cache = TransformerCache(keys=key_layer, values=value_layer)

  if input_cache is not None:
    key_layer = tf.concat([input_cache.keys, key_layer], 1)
    value_layer = tf.concat([input_cache.values, value_layer], 1)

  attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_layer, query_layer)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  # [B, 1, F, T]
  attention_mask = tf.expand_dims(attention_mask, axis=[1])
  attention_scores += (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  attention_probs = tf.nn.softmax(attention_scores)
  attention_probs = modeling.dropout(attention_probs,
                                     attention_probs_dropout_prob)

  # [B, F, N, H]
  context_layer = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_layer)

  return context_layer, output_cache


def compute_transformer(
    input_tensor,
    attention_mask,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    intermediate_size,
    intermediate_act_fn,
    hidden_dropout_prob,
    attention_probs_dropout_prob,
    initializer_range,
    input_cache,
):
  """Multi-headed, multi-layer Transformer."""
  attention_mask = tf.cast(attention_mask, tf.float32)
  attention_head_size = int(hidden_size / num_attention_heads)
  prev_output = input_tensor
  if input_cache is not None:
    input_cache = TransformerCache(
        keys=tf.unstack(input_cache.keys, axis=2),
        values=tf.unstack(input_cache.values, axis=2))
  output_cache = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output
      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          if input_cache is not None:
            layer_input_cache = TransformerCache(
                keys=input_cache.keys[layer_idx],
                values=input_cache.values[layer_idx])
          else:
            layer_input_cache = None
          attention_output, layer_output_cache = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              input_cache=layer_input_cache)
          output_cache.append(layer_output_cache)
        with tf.variable_scope("output"):
          attention_output = dense_layer_3d_proj(
              attention_output, hidden_size,
              num_attention_heads, attention_head_size,
              modeling.create_initializer(initializer_range), None, "dense")
          attention_output = modeling.dropout(attention_output,
                                              hidden_dropout_prob)
          attention_output = modeling.layer_norm(attention_output + layer_input)
      with tf.variable_scope("intermediate"):
        intermediate_output = dense_layer_2d(
            attention_output, intermediate_size,
            modeling.create_initializer(initializer_range), intermediate_act_fn,
            "dense")
      with tf.variable_scope("output"):
        layer_output = dense_layer_2d(
            intermediate_output, hidden_size,
            modeling.create_initializer(initializer_range), None, "dense")
        layer_output = modeling.dropout(layer_output, hidden_dropout_prob)
        layer_output = modeling.layer_norm(layer_output + attention_output)
        prev_output = layer_output

  # [batch_size, seq_len, num_layers, num_heads, head_size]
  output_cache = TransformerCache(
      keys=tf.stack([c.keys for c in output_cache], 2),
      values=tf.stack([c.values for c in output_cache], 2))
  return prev_output, output_cache


# ------------------------------------------------------------------------------
#
# Beam search interface for tensor2tensor.
#
# ------------------------------------------------------------------------------


def beam_search_decode(
    model,
    encoder_cache,
    encoder_cache_mask,
    start_id,
    stop_id,
    segment_id,
    num_steps,
    beam_size,
    alpha=0,
    reuse=tf.AUTO_REUSE,
):
  """Decode for a given number of steps."""
  true_batch_size = tensor_utils.shape(encoder_cache_mask, 0)
  num_layers = model.config.num_hidden_layers
  num_heads = model.config.num_attention_heads
  head_size = int(model.config.hidden_size / num_heads)

  def symbols_to_logits_fn(input_ids, i, state):
    """Go from ids to logits for next symbol."""
    # Size of expanded tensor (expanded by beam size).
    batch_size = tensor_utils.shape(input_ids, 0)

    # [batch_size, 1]
    current_step_mask = tf.ones([batch_size, 1], tf.int32)

    # [batch_size, num_steps]
    written_mask = tf.cast(tf.less(tf.range(num_steps), i), tf.int32)
    written_mask = tf.tile(tf.expand_dims(written_mask, 0), [batch_size, 1])
    is_written = tf.cast(written_mask, tf.bool)

    # [batch_size, cache_size + num_steps, num_layers, num_heads, head_size]
    input_cache = TransformerCache(
        keys=tf.concat([state.encoder_cache.keys, state.output_cache.keys], 1),
        values=tf.concat(
            [state.encoder_cache.values, state.output_cache.values], 1))

    # [batch_size, 1, cache_size + num_steps]
    masks = [state.encoder_cache_mask, written_mask, current_step_mask]
    attention_mask = tf.concat(masks, axis=1)
    attention_mask = tf.expand_dims(attention_mask, 1)

    # sequence_output: [batch_size, 1, hidden_size],
    # step_cache: [batch_size, 1, num_layers, num_heads, head_size]
    sequence_output, step_cache = model.compute_transformer(
        input_ids=input_ids,
        input_segment_id=tf.fill(tensor_utils.shape(input_ids), segment_id),
        input_positions=tf.fill(tensor_utils.shape(input_ids), i),
        attention_mask=attention_mask,
        input_cache=input_cache,
        reuse=reuse)

    # [batch_size, 1, vocab_size]
    logits = model.compute_logits(sequence_output, reuse=reuse)

    def update_values(old_values, current_value):
      """Update stored values with this time step."""
      shape = [1] * len(tensor_utils.shape(old_values))
      shape[:2] = [batch_size, num_steps]
      tile = tensor_utils.shape(old_values)
      tile[:2] = [1, 1]
      condition = tf.tile(tf.reshape(is_written, shape), tile)
      tile = [1] * len(tensor_utils.shape(old_values))
      tile[1] = num_steps
      current_value = tf.tile(current_value, tile)
      return tf.where(condition, old_values, current_value)

    # [batch_size, num_steps, num_layers, num_heads, head_size]
    beam_output_cache = TransformerCache(
        keys=update_values(state.output_cache.keys, step_cache.keys),
        values=update_values(state.output_cache.values, step_cache.values))

    # Return new state.
    state = DecodeState(
        encoder_cache=state.encoder_cache,
        encoder_cache_mask=state.encoder_cache_mask,
        output_cache=beam_output_cache)

    return tf.squeeze(logits, 1), state

  # Initialize output cache with zeros.
  shape = [true_batch_size, num_steps, num_layers, num_heads, head_size]
  output_cache = TransformerCache(keys=tf.zeros(shape), values=tf.zeros(shape))

  # Initialize state.
  state = DecodeState(
      encoder_cache=encoder_cache,
      encoder_cache_mask=encoder_cache_mask,
      output_cache=output_cache)

  # Decode using beam search.
  decoded_ids, scores, state = beam_search.beam_search(
      symbols_to_logits_fn=symbols_to_logits_fn,
      initial_ids=tf.fill([true_batch_size], start_id),
      eos_id=stop_id,
      beam_size=beam_size,
      alpha=alpha,
      decode_length=num_steps,
      vocab_size=model.config.vocab_size,
      states=state,
      use_tpu=True)

  # Postprocess.
  flat_mask = text_utils.get_token_mask(
      tf.reshape(decoded_ids, [-1, num_steps + 1]), stop_id)
  mask = tf.reshape(flat_mask, [true_batch_size, beam_size, num_steps + 1])
  decoded_ids *= mask

  return DecodeOutput(decoded_ids, mask, scores)

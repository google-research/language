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
"""The common model layers."""

import functools
import typing

from flax import linen as nn
import jax.numpy as jnp

default_kernel_init = nn.initializers.normal(1e-2)
default_bias_init = nn.initializers.zeros
default_embedding_init = nn.initializers.normal(1e-2)

InitFn = typing.Callable[[jnp.ndarray, typing.Any, typing.Optional[typing.Any]],
                         jnp.ndarray]
ActFn = typing.Callable[[jnp.ndarray], jnp.ndarray]


class TransformerEmbeddings(nn.Module):
  """Transformer embedding layer.

  Attributes:
    hidden_size: <int32>[1].
    vocab_size: <int32>[1].
    type_vocab_size: <int32>[1].
    max_position_embeddings: <int32>[1].
    hidden_dropout_rate: <float32>[1].
    layer_norm_eps: <float32>[1].
    embedding_init: Initializer for embeddings.
    deterministic: bool, deterministic or not (to apply dropout).
    decode: bool, whether deocde (to use cache) or not.
  """

  hidden_size: int
  vocab_size: int
  type_vocab_size: int
  max_position_embeddings: int = 512
  hidden_dropout_rate: float = 0.1
  layer_norm_eps: float = 1e-12
  embedding_init: InitFn = default_embedding_init
  deterministic: bool = True
  decode: bool = False

  @nn.compact
  def __call__(self, input_ids, pos_ids=None, token_type_ids=None):
    """Construct the Transformer embeddings.

    Args:
      input_ids: <int32>[batch_size, len].
      pos_ids: <int32>[batch_size, len].
      token_type_ids: <int32>[batch_size, len].

    Returns:
      embs: <float32>[batch_size, len, hidden_size].
    """
    batch_size, seq_length = input_ids.shape
    if pos_ids is None:
      pos_ids = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, 1))
      if self.decode:
        # cache position index for tracking decoding position.
        is_initialized = self.has_variable('cache', 'cache_index')
        cache_index = self.variable('cache', 'cache_index',
                                    lambda: jnp.array(0, dtype=jnp.uint32))
        if is_initialized:
          i = cache_index.value
          cache_index.value = i + 1
          pos_ids = jnp.full(input_ids.shape, i, dtype=jnp.int32)
    if token_type_ids is None:
      token_type_ids = jnp.zeros_like(input_ids)
    input_embs = nn.Embed(
        self.vocab_size,
        self.hidden_size,
        embedding_init=self.embedding_init,
        name='word_embeddings')(
            input_ids)
    position_embs = nn.Embed(
        self.max_position_embeddings,
        self.hidden_size,
        embedding_init=self.embedding_init,
        name='position_embeddings')(
            pos_ids)
    token_type_embs = nn.Embed(
        self.type_vocab_size,
        self.hidden_size,
        embedding_init=self.embedding_init,
        name='token_type_embeddings')(
            token_type_ids)
    embs = input_embs + position_embs + token_type_embs
    embs = nn.LayerNorm(epsilon=self.layer_norm_eps, name='LayerNorm')(embs)
    embs = nn.Dropout(rate=self.hidden_dropout_rate)(
        embs, deterministic=self.deterministic)
    return embs


class TransformerAttention(nn.Module):
  """Transformer attention layer.

  Attributes:
    num_heads: <int32>[1].
    hidden_size: <int32>[1].
    attention_dropout_rate: <float32>[1].
    hidden_dropout_rate: <float32>[1].
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
    decode: bool, whether deocde (to use cache) or not.
  """

  num_heads: int
  hidden_size: int
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True
  decode: bool = False

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, mask=None):
    """Apply attention block including attention and an output layer.

    Args:
      inputs_q: <float32>[batch_size, len, dim].
      inputs_kv: <float32>[batch_size, len, dim].
      mask: <int32>[batch_size, len].

    Returns:
      output: <float32>[batch_size, len, hidden_size].
    """
    output = nn.MultiHeadDotProductAttention(
        qkv_features=self.hidden_size,
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dropout_rate=self.attention_dropout_rate,
        decode=self.decode,
        name='attention',
    )(inputs_q, inputs_kv, mask=mask, deterministic=self.deterministic)
    output = TransformerOutput(
        hidden_size=self.hidden_size,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output')(output, inputs_q)
    return output


class TransformerCrossAttention(nn.Module):
  """Transformer cross-attention layer.

  Attributes:
    num_heads: <int32>[1].
    bi_hidden_size: <int32>[1].
    hidden_size1: <int32>[1].
    hidden_size2: <int32>[1].
    attention_dropout_rate: <float32>[1].
    hidden_dropout_rate: <float32>[1].
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
  """

  num_heads: int
  bi_hidden_size: int
  hidden_size1: int
  hidden_size2: int
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True

  @nn.compact
  def __call__(self, input1, input2, mask1=None, mask2=None):
    """Apply attention block including cross-attention and an output layer.

    Args:
      input1: <float32>[batch_size, len, dim].
      input2: <float32>[batch_size, len, dim].
      mask1: <int32>[batch_size, len].
      mask2: <int32>[batch_size, len].

    Returns:
      output: <float32>[batch_size, len, hidden_size].
    """
    context1 = nn.MultiHeadDotProductAttention(
        qkv_features=self.bi_hidden_size,
        out_features=self.bi_hidden_size,
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dropout_rate=self.attention_dropout_rate,
        name='cross1')(
            input2, input1, mask=mask1, deterministic=self.deterministic)
    context2 = nn.MultiHeadDotProductAttention(
        qkv_features=self.bi_hidden_size,
        out_features=self.bi_hidden_size,
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dropout_rate=self.attention_dropout_rate,
        name='cross2')(
            input1, input2, mask=mask2, deterministic=self.deterministic)
    output1, output2 = TransformerBiOutput(
        hidden_size1=self.hidden_size1,
        hidden_size2=self.hidden_size2,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        deterministic=self.deterministic,
        name='bi_output')(context2, input1, context1, input2)
    return output1, output2


class TransformerIntermediate(nn.Module):
  """Transformer interdediate layer after self-attention.

  Attributes:
    intermediate_size: <int32>[1].
    hidden_act: <int32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
  """

  intermediate_size: int
  hidden_act: ActFn = nn.gelu
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init

  @nn.compact
  def __call__(self, hidden):
    """Apply the Transformer intermediate layer.

    Args:
      hidden: <float32>[batch_size, len, dim].

    Returns:
      output: <float32>[batch_size, len, intermediate_size].
    """
    hidden = nn.Dense(
        self.intermediate_size,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense')(
            hidden)
    hidden = self.hidden_act(hidden)
    return hidden


class TransformerOutput(nn.Module):
  """Transformer output layer.

  Attributes:
    hidden_size: <int32>[1].
    hidden_dropout_rate: <int32>[1].
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
  """

  hidden_size: int
  hidden_dropout_rate: float
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True

  @nn.compact
  def __call__(self, hidden, x):
    """Apply the Transformer output layer.

    Args:
      hidden: <float32>[batch_size, len, dim].
      x: <float32>[batch_size, len, dim].

    Returns:
      output: <float32>[batch_size, len, hidden_size].
    """
    hidden = nn.Dense(
        self.hidden_size,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense')(
            hidden)
    hidden = nn.Dropout(rate=self.hidden_dropout_rate)(
        hidden, deterministic=self.deterministic)
    hidden = nn.LayerNorm(
        epsilon=self.layer_norm_eps, name='LayerNorm')(
            hidden + x)
    return hidden


class TransformerBiOutput(nn.Module):
  """Transformer bi-output layer.

  Attributes:
    hidden_size1: <int32>[1].
    hidden_size2: <int32>[1].
    hidden_dropout_rate: <float32>[1].
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
  """

  hidden_size1: int
  hidden_size2: int
  hidden_dropout_rate: float = 0.1
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True

  @nn.compact
  def __call__(self, hidden1, x1, hidden2, x2):
    """Apply the Transformer bi-output layer.

    Args:
      hidden1: <float32>[batch_size, len, dim].
      x1: <float32>[batch_size, len, dim].
      hidden2: <float32>[batch_size, len, dim].
      x2: <float32>[batch_size, len, dim].

    Returns:
      output: <float32>[batch_size, len, hidden_size1], <float32>[batch_size,
      len, hidden_size2]
    """
    transformer_output = functools.partial(
        TransformerOutput,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)
    hidden1 = transformer_output(
        hidden_size=self.hidden_size1, name='output1')(hidden1, x1)
    hidden2 = transformer_output(
        hidden_size=self.hidden_size2, name='output2')(hidden2, x2)
    return hidden1, hidden2


class TransformerPooler(nn.Module):
  """Transformer pooler.

  Attributes:
    hidden_size: The size of hidden layer.
    output_act: The output activation function.
    pooling_operator: The pooling operator, can be mean, first, or max.
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
  """

  hidden_size: int
  output_act: ActFn = nn.tanh
  pooling_operator: str = 'max'
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init

  @nn.compact
  def __call__(self, x, mask=None):
    """Apply the Transformer pooler.

    Args:
      x: <float32>[batch_size, len, dim].
      mask: <int32>[batch_size, len].

    Returns:
      pooled: <float32>[batch_size, hidden_size].
    """
    if mask is None:
      mask = jnp.ones(x.shape[:2], dtype=jnp.int32)
    mask = mask[:, :, None]
    if self.pooling_operator == 'mean':
      masked_x = x * mask
      pooled = masked_x.sum(1) / jnp.maximum(mask.sum(1), 1e-9)
    elif self.pooling_operator == 'first':
      pooled = x[:, 0]
    elif self.pooling_operator == 'max':
      masked_x = x * mask + -1e10 * (1 - mask)
      pooled = jnp.max(x, axis=1)
    else:
      raise NotImplementedError
    pooled = nn.Dense(
        self.hidden_size,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense')(
            pooled)
    pooled = self.output_act(pooled)
    return pooled


class TransformerLayer(nn.Module):
  """Transformer Layer including self-attention and feed-forward.

  Attributes:
    num_heads: <int32>[1].
    hidden_size: <int32>[1].
    intermediate_size: <int32>[1].
    attention_dropout_rate: <float32>[1].
    hidden_dropout_rate: <float32>[1].
    hidden_act: The hidden activation function.
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
  """

  num_heads: int
  hidden_size: int
  intermediate_size: int
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  hidden_act: ActFn = nn.gelu
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, mask=None):
    """Apply the Transformer layer.

    Args:
      inputs_q: <float32>[batch_size, len, dim].
      inputs_kv: <float32>[batch_size, len, dim].
      mask: <int32>[batch_size, len].

    Returns:
      layer_output: <float32>[batch_size, len, hidden_size].
    """
    attn_output = TransformerAttention(
        hidden_size=self.hidden_size,
        attention_dropout_rate=self.attention_dropout_rate,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        num_heads=self.num_heads,
        name='attention')(inputs_q, inputs_kv, mask)
    intermediate_output = TransformerIntermediate(
        intermediate_size=self.intermediate_size,
        hidden_act=self.hidden_act,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='intermediate')(
            attn_output)
    layer_output = TransformerOutput(
        hidden_size=self.hidden_size,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output')(intermediate_output, attn_output)
    return layer_output


class TransformerCrossLayer(nn.Module):
  """Transformer connection layer including cross-attention and feed-forward.

  Attributes:
    bi_num_heads: <int32>[1].
    bi_hidden_size: <int32>[1].
    hidden_size1: <int32>[1].
    hidden_size2: <int32>[1].
    intermediate_size1: <int32>[1].
    intermediate_size2: <int32>[1].
    attention_dropout_rate: <float32>[1].
    hidden_dropout_rate: <float32>[1].
    hidden_act: The hidden activation function.
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
  """

  bi_num_heads: int
  bi_hidden_size: int
  hidden_size1: int
  hidden_size2: int
  intermediate_size1: int
  intermediate_size2: int
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  hidden_act: ActFn = nn.gelu
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True

  @nn.compact
  def __call__(self, input1, input2, mask1=None, mask2=None):
    """Apply the Transformer connection layer.

    Args:
      input1: <float32>[batch_size, len, dim]
      input2: <float32>[batch_size, len, dim]
      mask1: <int32>[batch_size, len]
      mask2: <int32>batch_size, len].

    Returns:
      The output representation for input1 and input2 with shape
      (batch_size, len, hidden_size).
    """
    attn_output1, attn_output2 = TransformerCrossAttention(
        num_heads=self.bi_num_heads,
        bi_hidden_size=self.bi_hidden_size,
        hidden_size1=self.hidden_size1,
        hidden_size2=self.hidden_size2,
        attention_dropout_rate=self.attention_dropout_rate,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        deterministic=self.deterministic,
        name='bi_attention')(input1, input2, mask1, mask2)
    intermediate_layer = functools.partial(
        TransformerIntermediate,
        hidden_act=self.hidden_act,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)
    output_layer = functools.partial(
        TransformerOutput,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        deterministic=self.deterministic)
    intermediate_output1 = intermediate_layer(
        intermediate_size=self.intermediate_size1, name='intermediate1')(
            attn_output1)
    layer_output1 = output_layer(
        hidden_size=self.hidden_size1, name='output1')(intermediate_output1,
                                                       attn_output1)
    intermediate_output2 = intermediate_layer(
        intermediate_size=self.intermediate_size2, name='intermediate2')(
            attn_output2)
    layer_output2 = output_layer(
        hidden_size=self.hidden_size2, name='output2')(intermediate_output2,
                                                       attn_output2)
    return layer_output1, layer_output2


class TransformerEncoderDecoderLayer(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    num_heads: <int32>[1].
    hidden_size: <int32>[1].
    intermediate_size: <int32>[1].
    attention_dropout_rate: <float32>[1].
    hidden_dropout_rate: <float32>[1].
    hidden_act: The hidden activation function.
    layer_norm_eps: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
    deterministic: bool, deterministic or not (to apply dropout).
    decode: bool, whether deocde (to use cache) or not.
  """

  num_heads: int
  hidden_size: int
  intermediate_size: int
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  hidden_act: ActFn = nn.gelu
  layer_norm_eps: float = 1e-12
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = True
  decode: bool = False

  @nn.compact
  def __call__(self, x, encoded, decoder_mask=None, encoder_decoder_mask=None):
    """Apply the transfomer encoder-decoder layer.

    Args:
      x: <float32>[batch_size, len,  dim].
      encoded: <float32>[batch_size, len, dim].
      decoder_mask: <int32>[batch_size, len].
      encoder_decoder_mask: <int32>[batch_size, len].

    Returns:
      layer_output: <float32>[batch_size, len, hidden_size].
    """
    transformer_attention = functools.partial(
        TransformerAttention,
        hidden_size=self.hidden_size,
        attention_dropout_rate=self.attention_dropout_rate,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        num_heads=self.num_heads)
    attn_output = transformer_attention(
        name='attention', decode=self.decode)(
            x, x, mask=decoder_mask)
    attn_output = transformer_attention(name='encoder_decoder_attention')(
        attn_output, encoded, mask=encoder_decoder_mask)
    intermediate_output = TransformerIntermediate(
        intermediate_size=self.intermediate_size,
        hidden_act=self.hidden_act,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='intermediate')(
            attn_output)
    layer_output = TransformerOutput(
        hidden_size=self.hidden_size,
        hidden_dropout_rate=self.hidden_dropout_rate,
        layer_norm_eps=self.layer_norm_eps,
        deterministic=self.deterministic,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output')(intermediate_output, attn_output)
    return layer_output


class MLP(nn.Module):
  """The simple two-layer MLP.

  Attributes:
    hidden_size: <int32>[1].
    output_size: <int32>[1].
    hidden_act: The hidden activation function.
    dropout_rate: <float32>[1].
    kernel_init: Initializer for kernel of Dense layers.
    bias_init: Initializer for bias of Dense layers.
  """

  hidden_size: int
  output_size: int
  hidden_act: ActFn = nn.gelu
  dropout_rate: float = 0.5
  kernel_init: InitFn = default_kernel_init
  bias_init: InitFn = default_bias_init
  deterministic: bool = False

  @nn.compact
  def __call__(self, x):
    """Apply the two-layer MLP.

    Args:
      x: <float32>[batch_size, ..., dim].

    Returns:
      output: <float32>[batch_size, ..., output_size].
    """
    x = nn.Dense(
        self.hidden_size,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense')(
            x)
    x = self.hidden_act(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = nn.Dense(
        self.output_size,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense2')(
            x)
    return x

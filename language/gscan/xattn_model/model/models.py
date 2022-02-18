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
"""The Transformer-based models."""

import typing
import flax
from flax import linen as nn
import jax.numpy as jnp
from language.gscan.xattn_model.model import layers
from language.gscan.xattn_model.model import model_utils


@flax.struct.dataclass
class TransformerConfig:
  """Global model hyperparameters."""
  vocab_size: int
  target_vocab_size: int
  type_vocab_size: int = 2
  dtype: typing.Any = jnp.float32
  bi_hidden_dim: int = 128
  l_hidden_dim: int = 128
  v_hidden_dim: int = 128
  l_intermediate_dim: int = 256
  v_intermediate_dim: int = 256
  bi_num_heads: int = 8
  l_num_heads: int = 8
  v_num_heads: int = 8
  decode_num_heads: int = 8
  l_num_layers: int = 6
  v_num_layers: int = 6
  bi_num_layers: int = 6
  decode_num_layers: int = 6
  max_position_embeddings: int = 512
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  hidden_act: layers.ActFn = nn.gelu
  deterministic: bool = True
  kernel_init: layers.InitFn = layers.default_kernel_init
  bias_init: layers.InitFn = layers.default_bias_init
  embedding_init: layers.InitFn = layers.default_embedding_init
  layer_norm_eps: float = 1e-12
  cross_attn: bool = True
  num_conv_channels: int = 50
  conv_kernel_sizes: typing.Sequence[int] = (1, 5, 7)
  max_decode_step: int = 50
  decode: bool = False
  beam_size: int = 1


class CNNImageEncoder(nn.Module):
  """CNN-based image encoder."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, x):
    cfg = self.config
    feats = []
    for i, kernel_size in enumerate(cfg.conv_kernel_sizes):
      feat = nn.Conv(
          cfg.num_conv_channels,
          kernel_size=(kernel_size, kernel_size),
          name=f'conv_{i}')(
              x)
      feats.append(feat)
    img = jnp.concatenate(feats, axis=-1)
    img = img.reshape(img.shape[0], -1, img.shape[-1])
    img = nn.Dense(
        cfg.v_hidden_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        name='dense')(
            img)
    img = nn.relu(img)
    img = nn.Dropout(rate=cfg.dropout_rate)(
        img, deterministic=cfg.deterministic)
    return img


class TransformerEncoder(nn.Module):
  """The generatic transformer-based input encoder.

  It should be inherited with other transformer-based encoders, e.g. the
  encoder with or without cross-modal attention.
  """

  config: TransformerConfig

  def encode_txt(self, batch):
    cfg = self.config
    x = batch['token']
    mask = batch.get('txt_mask', jnp.ones(x.shape[:2], dtype=jnp.int32))
    assert x.ndim == 2, 'Inputs shape must be (batch_size, seq_len).'
    x = layers.TransformerEmbeddings(
        hidden_size=cfg.l_hidden_dim,
        vocab_size=cfg.vocab_size,
        type_vocab_size=cfg.type_vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_dropout_rate=cfg.dropout_rate,
        layer_norm_eps=cfg.layer_norm_eps,
        deterministic=cfg.deterministic,
        embedding_init=cfg.embedding_init,
        name='embeddings')(x, batch.get('pos_ids'), batch.get('seg_ids'))
    mask = mask[:, None, None, :]
    return x, mask

  def encode_image(self, batch):
    img = CNNImageEncoder(self.config, name='img_enc')(batch['image'])
    img_mask = jnp.ones(img.shape[:2], dtype=jnp.int32)
    img_mask = img_mask[:, None, None, :]
    return img, img_mask


class CrossModalEncoder(TransformerEncoder):
  """Transformer-based encoder with cross-modal attention."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, batch):
    cfg = self.config
    txt, txt_mask = self.encode_txt(batch)
    img, img_mask = self.encode_image(batch)

    for i in range(cfg.bi_num_layers):
      txt, img = layers.TransformerCrossLayer(
          bi_num_heads=cfg.bi_num_heads,
          bi_hidden_size=cfg.bi_hidden_dim,
          hidden_size1=cfg.l_hidden_dim,
          hidden_size2=cfg.v_hidden_dim,
          intermediate_size1=cfg.l_intermediate_dim,
          intermediate_size2=cfg.v_intermediate_dim,
          attention_dropout_rate=cfg.attention_dropout_rate,
          hidden_dropout_rate=cfg.dropout_rate,
          layer_norm_eps=cfg.layer_norm_eps,
          deterministic=cfg.deterministic,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          hidden_act=cfg.hidden_act,
          name=f'transformer_cross_layer_{i}')(txt, img, txt_mask, img_mask)
    encoded = jnp.concatenate((txt, img), axis=1)
    encoded_mask = jnp.concatenate(
        (txt_mask.squeeze(1).squeeze(1), img_mask.squeeze(1).squeeze(1)),
        axis=1)
    encoded = img
    encoded_mask = img_mask.squeeze(1).squeeze(1)
    return encoded, encoded_mask


class NonCrossModalEncoder(TransformerEncoder):
  """Transformer-based encoder without cross-modal attention."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, batch):
    cfg = self.config
    txt, txt_mask = self.encode_txt(batch)
    img, img_mask = self.encode_image(batch)

    for i in range(cfg.l_num_layers):
      txt = layers.TransformerLayer(
          num_heads=cfg.l_num_heads,
          hidden_size=cfg.l_hidden_dim,
          intermediate_size=cfg.l_intermediate_dim,
          attention_dropout_rate=cfg.attention_dropout_rate,
          hidden_dropout_rate=cfg.dropout_rate,
          layer_norm_eps=cfg.layer_norm_eps,
          deterministic=cfg.deterministic,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          hidden_act=cfg.hidden_act,
          name=f'txt_transformer_layer_{i}')(
              txt, txt, mask=txt_mask)
    for i in range(cfg.v_num_layers):
      img = layers.TransformerLayer(
          num_heads=cfg.v_num_heads,
          hidden_size=cfg.v_hidden_dim,
          intermediate_size=cfg.v_intermediate_dim,
          attention_dropout_rate=cfg.attention_dropout_rate,
          hidden_dropout_rate=cfg.dropout_rate,
          layer_norm_eps=cfg.layer_norm_eps,
          deterministic=cfg.deterministic,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          hidden_act=cfg.hidden_act,
          name=f'img_transformer_layer_{i}')(
              img, img, mask=img_mask)
    encoded = jnp.concatenate((txt, img), axis=1)
    encoded_mask = jnp.concatenate(
        (txt_mask.squeeze(1).squeeze(1), img_mask.squeeze(1).squeeze(1)),
        axis=1)
    return encoded, encoded_mask


class TransformerDecoder(nn.Module):
  """Transformer decoder."""

  config: TransformerConfig

  @nn.compact
  def __call__(self,
               x,
               encoded,
               pos_ids=None,
               token_type_ids=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    cfg = self.config
    x = layers.TransformerEmbeddings(
        hidden_size=cfg.l_hidden_dim,
        vocab_size=cfg.target_vocab_size,
        type_vocab_size=cfg.type_vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_dropout_rate=cfg.dropout_rate,
        layer_norm_eps=cfg.layer_norm_eps,
        deterministic=cfg.deterministic,
        embedding_init=cfg.embedding_init,
        decode=cfg.decode,
        name='embeddings')(x, pos_ids, token_type_ids)
    for i in range(cfg.decode_num_layers):
      x = layers.TransformerEncoderDecoderLayer(
          num_heads=cfg.decode_num_heads,
          hidden_size=cfg.l_hidden_dim,
          intermediate_size=cfg.l_intermediate_dim,
          attention_dropout_rate=cfg.attention_dropout_rate,
          hidden_dropout_rate=cfg.dropout_rate,
          layer_norm_eps=cfg.layer_norm_eps,
          deterministic=cfg.deterministic,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          hidden_act=cfg.hidden_act,
          decode=cfg.decode,
          name=f'transformer_encoder_decoder_layer_{i}')(x, encoded,
                                                         decoder_mask,
                                                         encoder_decoder_mask)
    x = nn.Dense(
        cfg.target_vocab_size,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        name='dense')(
            x)
    return x

  def get_attention_masks(self, inputs, targets):
    cfg = self.config
    if cfg.decode:
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          nn.make_causal_mask(targets, dtype=cfg.dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs > 0, dtype=cfg.dtype)
    return decoder_mask, encoder_decoder_mask


class Model(nn.Module):
  """The main model class."""

  config: TransformerConfig

  def setup(self):
    cfg = self.config
    if cfg.cross_attn:
      self.encoder = CrossModalEncoder(cfg)
    else:
      self.encoder = NonCrossModalEncoder(cfg)
    self.decoder = TransformerDecoder(cfg)

  def encode(self, batch):
    return self.encoder(batch)

  def decode(self, targets, encoded, targets_mask, inputs_mask):
    if not self.config.decode:
      targets = model_utils.shift_left(targets)
      targets_mask = model_utils.shift_left(targets_mask)
    decoder_mask, encoder_decoder_mask = self.decoder.get_attention_masks(
        inputs_mask, targets_mask)
    decoder_logits = self.decoder(
        targets,
        encoded,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask)
    return decoder_logits

  @nn.compact
  def __call__(self, batch):
    encoded, encoded_mask = self.encode(batch)
    decoder_logits = self.decode(batch['target_token'], encoded,
                                 batch['target_txt_mask'], encoded_mask)
    return decoder_logits

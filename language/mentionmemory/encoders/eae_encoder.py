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
"""Contains Entities as Experts encoder."""

from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp

from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import embedding
from language.mentionmemory.modules import entity_attention_layer
from language.mentionmemory.modules import transformer
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import


@encoder_registry.register_encoder('eae')
class EaEEncoder(base_encoder.BaseEncoder):
  """Entities as Experts (EaE) encoder.

  The Entities as Experts encoder (as in https://arxiv.org/abs/2004.07202)
  incorporates information from a corpus in the form of entity embeddings into a
  Transformer model.

  Attributes:
    vocab_size: size of token vocabulary.
    entity_vocab_size: size of entity vocabulary.
    hidden_size: dimensionality of token representations.
    intermediate_dim: dimensionality of intermediate representations in MLP.
    entity_dim: dimensionality of entity embeddings.
    num_attention_heads: number of attention heads in Transformer layers.
    num_initial_layers: number of layers in first Transformer block.
    num_final_layers: number of layers in second Transformer block.
    dtype: data type of encoding (bfloat16 or float32). Parameters and certain
      parts of computation (i.e. loss) are always in float32.
    max_positions: number of positions (for positional embeddings).
    max_length: maximal number of tokens for pre-training.
    dropout_rate: dropout rate in Transformer layers.
    num_segments: number of possible token types (for token type embeddings).
    kernel_init: initialization function for model kernels.
    bias_init: initialization function for model biases.
    layer_norm_epsilon: layer norm constant for numerical stability.
    no_entity_attention: if true, do not incorporate retrieved entity embeddings
      into Transformer model.
  """
  vocab_size: int
  entity_vocab_size: int
  hidden_size: int
  intermediate_dim: int
  entity_dim: int
  num_attention_heads: int
  num_initial_layers: int
  num_final_layers: int
  dtype: Dtype
  max_positions: int
  # TODO(urikz): Move this argument out of model parameters
  max_length: int
  dropout_rate: float

  num_segments: int = 2
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon
  no_entity_attention: bool = False

  def setup(self):

    self.embedder = embedding.DictEmbed({
        'token_ids':
            embedding.Embed(
                num_embeddings=self.vocab_size,
                embedding_dim=self.hidden_size,
                dtype=self.dtype,
                embedding_init=self.kernel_init,
            ),
        'position_ids':
            embedding.Embed(
                num_embeddings=self.max_positions,
                embedding_dim=self.hidden_size,
                dtype=self.dtype,
                embedding_init=self.kernel_init,
            ),
        'segment_ids':
            embedding.Embed(
                num_embeddings=self.num_segments,
                embedding_dim=self.hidden_size,
                dtype=self.dtype,
                embedding_init=self.kernel_init,
            )
    })

    self.embeddings_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
    self.embeddings_dropout = nn.Dropout(rate=self.dropout_rate)

    self.initial_encoder = transformer.TransformerBlock(
        num_layers=self.num_initial_layers,
        model_dim=self.hidden_size,
        intermediate_dim=self.intermediate_dim,
        num_heads=self.num_attention_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

    self.entity_embeddings = self.param(
        'entity_embeddings', self.kernel_init,
        (self.entity_vocab_size, self.entity_dim))

    if self.no_entity_attention:
      self.intermediate_mention_projector = nn.Dense(
          features=self.entity_dim,
          dtype=self.dtype,
      )
    else:
      self.entity_attention_layer = entity_attention_layer.EntityAttentionLayer(
          hidden_size=self.hidden_size,
          entity_dim=self.entity_dim,
          dtype=self.dtype,
          layer_norm_epsilon=self.layer_norm_epsilon,
      )

    self.final_encoder = transformer.TransformerBlock(
        num_layers=self.num_final_layers,
        model_dim=self.hidden_size,
        intermediate_dim=self.intermediate_dim,
        num_heads=self.num_attention_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

    self.mention_projector = nn.Dense(
        features=self.entity_dim,
        dtype=self.dtype,
    )

  def forward(
      self,
      batch: Dict[str, Array],
      deterministic: bool,
  ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:

    loss_helpers = {}
    logging_helpers = {}

    embedded_input = self.embedder({
        'token_ids': batch['text_ids'],
        'position_ids': batch['position_ids'],
        'segment_ids': batch['segment_ids']
    })

    embedded_input = self.embeddings_layer_norm(embedded_input)
    embedded_input = self.embeddings_dropout(
        embedded_input, deterministic=deterministic)

    loss_helpers['word_embeddings'] = self.embedder.variables['params'][
        'embedders_token_ids']['embedding']

    attention_mask = batch['text_mask']
    encoding = self.initial_encoder(
        encoding=embedded_input,
        attention_mask=attention_mask,
        deterministic=deterministic)

    if self.no_entity_attention:
      if 'mention_target_batch_positions' in batch:
        mention_start_encodings = jut.matmul_2d_index_select(
            encoding, (batch['mention_target_batch_positions'],
                       batch['mention_target_start_positions']))
        mention_end_encodings = jut.matmul_2d_index_select(
            encoding, (batch['mention_target_batch_positions'],
                       batch['mention_target_end_positions']))
        loss_helpers[
            'im_target_mention_encodings'] = self.intermediate_mention_projector(
                jnp.concatenate(
                    (mention_start_encodings, mention_end_encodings), axis=-1))
        loss_helpers['entity_embeddings'] = jnp.asarray(
            self.entity_embeddings, dtype=self.dtype)
    else:
      entity_attention_output = self.entity_attention_layer(
          encoded_input=encoding,
          mention_batch_positions=batch['mention_batch_positions'],
          mention_start_positions=batch['mention_start_positions'],
          mention_end_positions=batch['mention_end_positions'],
          mention_mask=batch['mention_mask'],
          entity_embeddings=jnp.asarray(
              self.entity_embeddings, dtype=self.dtype),
      )
      encoding = entity_attention_output['encoded_output']
      loss_helpers['intermediate_mention_encodings'] = entity_attention_output[
          'mention_encodings']
      loss_helpers['intermediate_entity_attention'] = entity_attention_output[
          'attention_weights']
      loss_helpers['intermediate_entity_cos_sim'] = entity_attention_output[
          'cosine_similarity']

    encoding = self.final_encoder(
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=deterministic)

    if 'mention_target_batch_positions' in batch:
      mention_start_encodings = jut.matmul_2d_index_select(
          encoding, (batch['mention_target_batch_positions'],
                     batch['mention_target_start_positions']))
      mention_end_encodings = jut.matmul_2d_index_select(
          encoding, (batch['mention_target_batch_positions'],
                     batch['mention_target_end_positions']))
      loss_helpers['target_mention_encodings'] = self.mention_projector(
          jnp.concatenate((mention_start_encodings, mention_end_encodings),
                          axis=-1))
      loss_helpers['entity_embeddings'] = jnp.asarray(
          self.entity_embeddings, dtype=self.dtype)

    return encoding, loss_helpers, logging_helpers

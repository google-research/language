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
"""Contains mention auto-encoder implementation."""

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import embedding
from language.mentionmemory.modules import retrieval_update_layers
from language.mentionmemory.modules import transformer
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import
import ml_collections


@encoder_registry.register_encoder('mauto')
class MautoEncoder(base_encoder.BaseEncoder):
  """Mention autoencoder.

  More precisely, this model is designed to evaluate information stored by a
  mention encoder, as well as varying methods for incorporating information into
  a language model. During pre-training the model has access to oracle memory
  entries corresponding to a subset of linked mentions in the passage.

  Attributes:
    vocab_size: size of token vocabulary.
    hidden_size: dimensionality of token representations.
    intermediate_dim: dimensionality of intermediate representations in MLP.
    retrieval_dim: dimensionality of memory values.
    retrieval_update_type: means by which retrieved memory vectors are
      incorporated into input representation, such as simple addition or
      concatenation + MLP.
    retrieval_update_config: hyperparameters for the update layer, beyond input
      dimension and datatype.
    num_attention_heads: number of attention heads in Transformer layers.
    num_initial_layers: number of layers in first Transformer block.
    num_final_layers: number of layers in second Transformer block.
    dtype: data type of encoding (bfloat16 or float32). Parameters and certain
      parts of computation (i.e. loss) are always in float32.
    max_positions: number of positions (for positional embeddings).
    max_length: maximal number of tokens for pre-training.
    dropout_rate: dropout rate in Transformer layers.
    no_retrieval: if true, do not incorporate retrieved mentions into model.
    num_segments: number of possible token types (for token type embeddings).
    kernel_init: initialization function for model kernels.
    bias_init: initialization function for model biases.
    layer_norm_epsilon: layer norm constant for numerical stability.
  """

  vocab_size: int
  hidden_size: int
  intermediate_dim: int
  retrieval_dim: int
  retrieval_update_type: str
  retrieval_update_config: ml_collections.FrozenConfigDict
  num_attention_heads: int
  num_initial_layers: int
  num_final_layers: int
  dtype: Dtype
  max_positions: int
  max_length: int
  dropout_rate: float
  num_segments: int = 2
  no_retrieval: bool = False
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

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

    self.retrieval_update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        self.retrieval_update_type](
            input_dim=self.hidden_size,
            dtype=self.dtype,
            layer_norm_epsilon=self.layer_norm_epsilon,
            **self.retrieval_update_config,
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
        features=self.retrieval_dim,
        dtype=self.dtype,
    )

  def forward(self, batch: Dict[str, Array], deterministic: bool):
    loss_helpers = {}
    logging_helpers = {}

    embedded_input = self.embedder({
        'token_ids': batch['text_ids'],
        'position_ids': batch['position_ids'],
        'segment_ids': batch['segment_ids']
    })

    embedded_input = self.embeddings_layer_norm(embedded_input)
    embedded_input = self.embeddings_dropout(embedded_input, deterministic)

    loss_helpers['word_embeddings'] = self.embedder.variables['params'][
        'embedders_token_ids']['embedding']

    attention_mask = batch['text_mask']
    encoding = self.initial_encoder(
        encoding=embedded_input,
        attention_mask=attention_mask,
        deterministic=deterministic)

    if not self.no_retrieval:
      encoding = self.retrieval_update_layer(
          encoded_input=encoding,
          retrieval_values=jnp.expand_dims(
              # [max_retrieval_indices, retrieval_dim]
              batch['retrieval_mention_values'],
              -2),
          retrieval_scores=jnp.expand_dims(
              # [max_retrieval_indices]
              batch['retrieval_mention_scores'],
              -1),
          mention_batch_positions=batch['retrieval_mention_batch_positions'],
          mention_start_positions=batch['retrieval_mention_start_positions'],
          mention_end_positions=batch['retrieval_mention_end_positions'],
          mention_mask=batch['retrieval_mention_mask'],
          deterministic=deterministic)

    encoding = self.final_encoder(
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=deterministic)

    mention_target_batch_positions = jut.matmul_slice(
        batch['mention_batch_positions'], batch['mention_target_indices'])
    mention_target_start_positions = jut.matmul_slice(
        batch['mention_start_positions'], batch['mention_target_indices'])
    mention_target_end_positions = jut.matmul_slice(
        batch['mention_end_positions'], batch['mention_target_indices'])

    mention_start_final_encodings = jut.matmul_2d_index_select(
        encoding,
        (mention_target_batch_positions, mention_target_start_positions))
    mention_end_final_encodings = jut.matmul_2d_index_select(
        encoding,
        (mention_target_batch_positions, mention_target_end_positions))
    loss_helpers['target_mention_encodings'] = self.mention_projector(
        jnp.concatenate(
            (mention_start_final_encodings, mention_end_final_encodings),
            axis=-1))

    return encoding, loss_helpers, logging_helpers

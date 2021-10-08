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
"""Contains readtwice encoder implementation."""



import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import batch_memory_attention_layer
from language.mentionmemory.modules import embedding
from language.mentionmemory.modules import memory_extraction_layer
from language.mentionmemory.modules import transformer
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils import mention_utils
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import
import ml_collections


@encoder_registry.register_encoder('read_twice')
class ReadTwiceEncoder(base_encoder.BaseEncoder):
  """Implementation of ReadTwice model (see https://arxiv.org/abs/2105.04241).

  The ReadTwice model first encodes the passage with a standard Transformer and
  generates a global mention memory from mentions across the batch (including
  those on different devices). ReadTwice then encodes the passage again,
  attending to the mention memory and cheaply allowing interactions across
  large or different documents.

  Attributes:
    vocab_size: size of token vocabulary.
    memory_key_dim: dimensionality of memory keys.
    memory_value_dim: dimensionality of memory values.
    memory_update_type: name of layer used to incorporate retrieved memories
      into Transformer.
    memory_update_config: hyperparameters for for memory update layer.
    hidden_size: dimensionality of token representations.
    intermediate_dim: dimensionality of intermediate representations in MLP.
    num_attention_heads: number of attention heads in Transformer layers.
    num_initial_layers: number of layers in first Transformer block.
    num_final_layers: number of layers in second Transformer block.
    dtype: data type of encoding (bfloat16 or float32). Parameters and certain
      parts of computation (i.e. loss) are always in float32.
    max_positions: number of positions (for positional embeddings).
    max_length: maximal number of tokens for pre-training.
    dropout_rate: dropout rate in Transformer layers.
    no_retrieval: if true, skip retrieval layer.
    same_passage_retrieval_policy: whether the memory layer attends to mentions
      from the same passage. Possible choices are allow, disallo, and only.
    extract_unlinked_mentions: if true, also extract memories from unlinked
      mentions and place in batch mention memory.
    no_retrieval_for_masked_mentions: if true, disables retrieval for masked
      mentions.
    shared_initial_encoder: if true, use same weights for initial encoder in
      first and second read.
    shared_intermediate_encoder: if true, use same weights for intermediate
      encoder in first and second read.
    shared_final_encoder: if true, use same weights for final encoder in first
      and second read.
    num_initial_layers_second: number of layers to use for second initial
      encoder if not shared.
    num_intermediate_layers_second: number of layers to use for second
      intermediate encoder if not shared.
    num_final_layers_second: number of layers to use for second final encoder if
      not shared.
    k_top: number of retrievals. if None, retrieve entire memory table.
    rows: governs speed-recall trade-off in approximate top-k operation in
      memory attention layer. Only active if k_top is not None.
    num_segments: number of possible token types (for token type embeddings).
    kernel_init: initialization function for model kernels.
    bias_init: initialization function for model biases.
    layer_norm_epsilon: layer norm constant for numerical stability.
  """

  vocab_size: int
  memory_key_dim: int
  memory_value_dim: int
  memory_update_type: str
  memory_update_config: ml_collections.FrozenConfigDict
  hidden_size: int
  intermediate_dim: int
  num_attention_heads: int
  num_initial_layers: int
  num_final_layers: int
  dtype: Dtype
  max_positions: int
  max_length: int
  dropout_rate: float

  num_intermediate_layers: Optional[int] = None
  no_retrieval: bool = False
  same_passage_retrieval_policy: str = 'allow'
  extract_unlinked_mentions: bool = False
  no_retrieval_for_masked_mentions: bool = False
  shared_initial_encoder: bool = True
  shared_intermediate_encoder: bool = True
  shared_final_encoder: bool = True
  num_initial_layers_second: Optional[int] = None
  num_intermediate_layers_second: Optional[int] = None
  num_final_layers_second: Optional[int] = None
  k_top: Optional[int] = None
  rows: Optional[int] = None
  num_segments: int = 2
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

  def setup(self):

    def make_transformer_block(n_layers):
      return transformer.TransformerBlock(
          num_layers=n_layers,
          model_dim=self.hidden_size,
          intermediate_dim=self.intermediate_dim,
          num_heads=self.num_attention_heads,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          layer_norm_epsilon=self.layer_norm_epsilon,
      )

    def make_memory_attention_layer():
      return batch_memory_attention_layer.BatchMemoryAttentionLayer(
          memory_key_dim=self.memory_key_dim,
          input_dim=self.hidden_size,
          memory_update_type=self.memory_update_type,
          memory_update_config=self.memory_update_config,
          dtype=self.dtype,
          k_top=self.k_top,
          rows=self.rows,
          layer_norm_epsilon=self.layer_norm_epsilon,
      )

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

    self.initial_encoder = make_transformer_block(self.num_initial_layers)
    if not self.shared_initial_encoder:
      self.second_initial_encoder = make_transformer_block(
          self.num_initial_layers_second)

    if self.num_intermediate_layers is not None:
      self.intermediate_memory_attention_layer = make_memory_attention_layer()
      self.final_memory_attention_layer = make_memory_attention_layer()
      self.intermediate_encoder = make_transformer_block(
          self.num_intermediate_layers)
      if self.shared_intermediate_encoder:
        self.second_intermediate_encoder = self.intermediate_encoder
      else:
        self.second_intermediate_encoder = make_transformer_block(
            self.num_intermediate_layers_second)
    else:
      self.memory_attention_layer = make_memory_attention_layer()

    self.final_encoder = make_transformer_block(self.num_final_layers)
    if self.shared_final_encoder:
      self.second_final_encoder = self.final_encoder
    else:
      self.second_final_encoder = make_transformer_block(
          self.num_final_layers_second)

    self.memory_extraction_layer = memory_extraction_layer.MemoryExtractionLayer(
        memory_key_dim=self.memory_key_dim,
        memory_value_dim=self.memory_value_dim,
        dtype=self.dtype,
    )

    self.mention_projector = nn.Dense(
        features=self.memory_value_dim,
        dtype=self.dtype,
    )

  def forward(
      self, batch,
      deterministic):
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

    def apply_encoder_block(encoder_block, encoding):
      return encoder_block(
          encoding=encoding,
          attention_mask=attention_mask,
          deterministic=deterministic,
      )

    # First read
    initial_encoding_first = apply_encoder_block(self.initial_encoder,
                                                 embedded_input)
    if self.num_intermediate_layers is not None:
      encoding_first = apply_encoder_block(self.intermediate_encoder,
                                           initial_encoding_first)
    else:
      encoding_first = initial_encoding_first
    encoding_first = apply_encoder_block(self.final_encoder, encoding_first)

    loss_helpers['final_encoding_first'] = encoding_first

    mention_batch_positions = batch['mention_batch_positions']
    mention_target_batch_positions = batch['mention_target_batch_positions']

    # Generate memory table
    if self.extract_unlinked_mentions:
      local_memory_entity_ids = jnp.zeros(
          dtype=jnp.int32, shape=batch['mention_mask'].shape[0])
      local_memory_entity_ids = local_memory_entity_ids.at[
          batch['mention_target_indices']].set(batch['mention_target_ids'])
      extracted_memory_dict = self.memory_extraction_layer(
          encoding=encoding_first,
          mention_batch_positions=mention_batch_positions,
          mention_start_positions=batch['mention_start_positions'],
          mention_end_positions=batch['mention_end_positions'],
          mention_mask=batch['mention_mask'],
          mention_entity_ids=local_memory_entity_ids,
      )
    else:
      extracted_memory_dict = self.memory_extraction_layer(
          encoding=encoding_first,
          mention_batch_positions=mention_target_batch_positions,
          mention_start_positions=batch['mention_target_start_positions'],
          mention_end_positions=batch['mention_target_end_positions'],
          mention_mask=batch['mention_target_weights'],
          mention_entity_ids=batch['mention_target_ids'],
      )
    memory_keys = extracted_memory_dict['memory_keys']
    memory_values = extracted_memory_dict['memory_values']
    memory_mask = extracted_memory_dict['memory_mask']
    memory_entity_ids = extracted_memory_dict['memory_entity_ids']
    local_memory_keys = extracted_memory_dict['local_memory_keys']
    local_memory_values = extracted_memory_dict['local_memory_values']

    loss_helpers['memory_keys'] = local_memory_keys
    loss_helpers['memory_values'] = local_memory_values

    if self.same_passage_retrieval_policy == 'only':
      memory_keys = local_memory_keys
      memory_values = local_memory_values
      memory_mask = mention_utils.all_compare(mention_batch_positions,
                                              mention_target_batch_positions)
      memory_entity_ids = batch['mention_target_ids']
    elif self.same_passage_retrieval_policy == 'disallow':
      # Transform local batch positions to global batch positions
      bsz = encoding_first.shape[0]
      (global_mention_batch_positions,
       _) = mention_utils.get_globally_consistent_batch_positions(
           mention_batch_positions, bsz)

      (_, all_global_mention_target_batch_positions
      ) = mention_utils.get_globally_consistent_batch_positions(
          mention_target_batch_positions, bsz)

      # Set mask to true only for mentions from different samples
      memory_mask = mention_utils.all_compare(
          global_mention_batch_positions,
          all_global_mention_target_batch_positions)
    elif self.same_passage_retrieval_policy != 'allow':
      raise ValueError('Unknown value for same_passage_retrieval_policy: %s' %
                       self.same_passage_retrieval_policy)

    mention_mask = batch['mention_mask']
    if self.no_retrieval_for_masked_mentions:
      mention_mask = mention_mask * (1 - batch['mention_is_masked'])

    def apply_memory_layer(attention_layer, encoding, prefix=''):
      encoding, mem_loss_helpers, mem_logging_helpers = attention_layer(
          encoding=encoding,
          mention_batch_positions=mention_batch_positions,
          mention_start_positions=batch['mention_start_positions'],
          mention_end_positions=batch['mention_end_positions'],
          mention_mask=mention_mask,
          memory_keys=memory_keys,
          memory_values=memory_values,
          memory_mask=memory_mask,
          memory_entity_ids=memory_entity_ids,
          deterministic=deterministic,
      )
      loss_helpers.update(
          {prefix + key: value for key, value in mem_loss_helpers.items()})
      logging_helpers.update(
          {prefix + key: value for key, value in mem_logging_helpers.items()})
      return encoding

    # If second read shares initial encoder, just reuse initial representation
    if self.shared_initial_encoder:
      encoding_second = initial_encoding_first
    else:
      encoding_second = apply_encoder_block(self.second_initial_encoder,
                                            embedded_input)

    if self.num_intermediate_layers is not None:
      if not self.no_retrieval:
        encoding_second = apply_memory_layer(
            attention_layer=self.intermediate_memory_attention_layer,
            encoding=encoding_second)
      encoding_second = apply_encoder_block(self.second_intermediate_encoder,
                                            encoding_second)
      if not self.no_retrieval:
        encoding_second = apply_memory_layer(
            attention_layer=self.final_memory_attention_layer,
            encoding=encoding_second,
            prefix='second_')
    else:
      if not self.no_retrieval:
        encoding_second = apply_memory_layer(
            attention_layer=self.memory_attention_layer,
            encoding=encoding_second)
    encoding_second = apply_encoder_block(self.second_final_encoder,
                                          encoding_second)

    if 'mention_target_batch_positions' in batch:
      mention_start_final_encodings = jut.matmul_2d_index_select(
          encoding_second, (batch['mention_target_batch_positions'],
                            batch['mention_target_start_positions']))
      mention_end_final_encodings = jut.matmul_2d_index_select(
          encoding_second, (batch['mention_target_batch_positions'],
                            batch['mention_target_end_positions']))
      loss_helpers['target_mention_encodings'] = self.mention_projector(
          jnp.concatenate(
              (mention_start_final_encodings, mention_end_final_encodings),
              axis=-1))

    return encoding_second, loss_helpers, logging_helpers

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
"""Contains mention memory encoder implementation."""



from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import embedding
from language.mentionmemory.modules import memory_attention_layer
from language.mentionmemory.modules import memory_retrieval_layer
from language.mentionmemory.modules import transformer
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import
import ml_collections
import numpy as np


class MemoryRetrievalResultProcessor:
  """Processes retrieval result for manual analysis."""

  def __init__(self, config):
    self.memory_reduction = config.memory_reduction
    self.memory_entity_id_pattern = config.memory_entity_id_pattern
    self.memory_text_pattern = config.memory_text_pattern
    self.memory_positions_pattern = config.memory_positions_pattern
    self.save_k_retrieval = config.get('save_k_retrieval', 10)
    if self.save_k_retrieval is not None and config.model_config.encoder_config.get(
        'k_top_post_selection') is None:
      raise Exception('save_k_retrieval only allowed with k_top_post_selection')

    # Lazy load for memory
    self.memory_entity_id = None
    self.memory_text = None
    self.memory_positions = None

    # TODO(urikz): Move `memory_prop` to `data_utils.load_sharded_array`,
    # e.g. array = array[:int(memory_prop * array.shape[0])]
    assert config.memory_prop is None

  def load_array(self, pattern):
    """Load sharded array as if it was loaded from multiple processes."""
    process_count = jax.process_count()
    arrays = []
    for process_index in range(process_count):
      arrays.append(
          data_utils.load_sharded_array(pattern,
                                        process_count * self.memory_reduction,
                                        process_index))
    array = np.stack(arrays, axis=0)
    shape = (-1,) + arrays[0].shape[1:]
    array = array.reshape(shape)
    return array

  def load_memory(self):
    """Load all necessary memory arrays."""
    self.memory_entity_id = self.load_array(self.memory_entity_id_pattern)
    self.memory_text = self.load_array(self.memory_text_pattern)
    self.memory_positions = self.load_array(self.memory_positions_pattern)

  def maybe_load_memory(self):
    """Load memory (passages and mention positions) if it's not loaded yet."""
    if self.memory_entity_id is None:
      self.load_memory()

  def __call__(self, batch,
               auxiliary_output):
    """Produces memory texts and mention positions given memory IDs."""
    # We want to load memory lazily, so we do that here and not in the
    # constructor. First, we check if memory components have been loaded yet
    # (`self.memory_entity_id` is not None) and actually load them only if not.
    self.maybe_load_memory()

    def process_retrievals(prefix=''):
      top_entity_ids = auxiliary_output[prefix + 'top_entity_ids']
      top_memory_ids = auxiliary_output[prefix + 'top_memory_ids']
      if self.save_k_retrieval is not None:
        top_entity_ids = top_entity_ids[:, :, :self.save_k_retrieval]
        top_memory_ids = top_memory_ids[:, :, :self.save_k_retrieval]
      n_devices = top_entity_ids.shape[0]
      n_mentions = top_entity_ids.shape[1]
      n_retrievals = top_entity_ids.shape[2]
      assert top_entity_ids.shape == top_memory_ids.shape
      logging.info(
          'Saving %sretrievals: n_devices=%d, n_mentions=%d, n_retrievals=%d',
          prefix, n_devices, n_mentions, n_retrievals)

      features = {
          'memory_text': [],
          'memory_positions': [],
          'memory_entity_id': [],
      }
      for device_index in range(n_devices):
        memory_text_per_device = []
        memory_positions_per_device = []
        memory_entity_id_per_device = []
        for mention_index in range(n_mentions):
          memory_text_per_mention = []
          memory_positions_per_mention = []
          memory_entity_id_per_mention = []
          for r_index in range(n_retrievals):
            memory_index = top_memory_ids[device_index, mention_index, r_index]
            memory_text_per_mention.append(
                self.memory_text[memory_index].tolist())
            memory_positions_per_mention.append(
                self.memory_positions[memory_index].tolist())
            memory_entity_id_per_mention.append(
                self.memory_entity_id[memory_index].tolist())
          memory_text_per_device.append(memory_text_per_mention)
          memory_positions_per_device.append(memory_positions_per_mention)
          memory_entity_id_per_device.append(memory_entity_id_per_mention)
        features['memory_text'].append(memory_text_per_device)
        features['memory_positions'].append(memory_positions_per_device)
        features['memory_entity_id'].append(memory_entity_id_per_device)

      n_mismatch = (np.array(features['memory_entity_id']) !=
                    top_entity_ids).sum()
      if n_mismatch > 0:
        raise ValueError('Found %d mismatches amongst %d IDs in total' %
                         (n_mismatch, top_entity_ids.size))
      return {prefix + key: value for key, value in features.items()}

    retrieval_features = process_retrievals()
    if 'second_top_entity_ids' in auxiliary_output:
      retrieval_features.update(process_retrievals('second_'))
    return retrieval_features


@encoder_registry.register_encoder('mention_memory')
class MentionMemoryEncoder(base_encoder.BaseEncoder):
  """Mention Memory Encoder.

  The Mention Memory encoder incorporates information from an external mention
  memory into a Transformer model, through similar architecture as Entities as
  Experts.

  Attributes:
    vocab_size: size of token vocabulary.
    hidden_size: dimensionality of token representations.
    intermediate_dim: dimensionality of intermediate representations in MLP.
    memory_key_dim: dimensionality of memory keys.
    separate_memory_values: if true, use separate keys and values for memory.
    memory_update_type: means by which retrieved memory vectors are incorporated
      into input representation, such as simple addition or concatenation + MLP.
    memory_update_config: hyperparameters for the update layer, beyond input
      dimension and datatype.
    k_top_device: top-k retrieved memory vectors per device.
    rows: number of rows in memory table, governs tradeoff between recall and
      speed.
    splits: governs a tradeoff between speed and memory usage in topk similarity
      search layer and has no effect on actual search results. A higher number
      of splits is slower but uses less memory.
    num_attention_heads: number of attention heads in Transformer layers.
    num_initial_layers: number of layers in first Transformer block.
    num_final_layers: number of layers in second Transformer block.
    dtype: data type of encoding (bfloat16 or float32). Parameters and certain
      parts of computation (i.e. loss) are always in float32.
    max_positions: number of positions (for positional embeddings).
    max_length: maximal number of tokens for pre-training.
    dropout_rate: dropout rate in Transformer layers.
    memory_value_dim: dimensionality of memory values.
    k_top_post_selection: Select Top-k memories after retrieving `k_top_device`
      top memories from every device.
    num_segments: number of possible token types (for token type embeddings).
    final_k_top_device: Parameter for the final retrieval layer. top-k retrieved
      memory vectors per device.
    final_splits: Parameter for the final retrieval layer. Governs a tradeoff
      between speed and memory usage in topk similarity search layer and has no
      effect on actual search results. A higher number of splits is slower but
      uses less memory.
    final_k_top_post_selection: Parameter for the final retrieval layer. Select
      Top-k memories after retrieving `k_top_device` top memories from every
      device.
    kernel_init: initialization function for model kernels.
    bias_init: initialization function for model biases.
    layer_norm_epsilon: layer norm constant for numerical stability.
    same_passage_memory_policy: how to treat mentions from the same passage.
        Possible options: `allow`, `disallow` and `only`.
  """
  vocab_size: int
  hidden_size: int
  intermediate_dim: int
  memory_key_dim: int
  separate_memory_values: bool
  memory_update_type: str
  memory_update_config: ml_collections.FrozenConfigDict
  k_top_device: int
  rows: int
  splits: int
  num_attention_heads: int
  num_initial_layers: int
  num_final_layers: int
  dtype: Dtype
  max_positions: int
  max_length: int
  dropout_rate: float
  same_passage_memory_policy: str

  num_intermediate_layers: Optional[int] = None
  memory_value_dim: Optional[int] = None
  k_top_post_selection: Optional[int] = None
  n_memory_text_entities: Optional[int] = None
  num_segments: int = 2
  final_k_top_device: Optional[int] = None
  final_splits: Optional[int] = None
  final_k_top_post_selection: Optional[int] = None

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

    def make_attention_layer():
      return memory_attention_layer.MemoryAttentionLayer(
          memory_key_dim=self.memory_key_dim,
          input_dim=self.hidden_size,
          memory_update_type=self.memory_update_type,
          memory_update_config=self.memory_update_config,
          k_top_device=self.k_top_device,
          splits=self.splits,
          dtype=self.dtype,
          k_top_post_selection=self.k_top_post_selection,
          layer_norm_epsilon=self.layer_norm_epsilon,
      )

    self.memory_keys = self.variable(
        'constants',
        'memory_keys',
        self.bias_init,
        None,
        (self.rows, self.rows, self.memory_key_dim),
        self.dtype,
    )

    if self.separate_memory_values:
      self.memory_values = self.variable(
          'constants',
          'memory_values',
          self.bias_init,
          None,
          (self.rows * self.rows, self.memory_value_dim),
          self.dtype,
      )

    self.memory_identifiers = self.variable(
        'constants',
        'memory_identifiers',
        self.bias_init,
        None,
        (self.rows,),
        jnp.int32,
    )

    self.memory_entity_ids = self.variable(
        'constants',
        'memory_entity_ids',
        self.bias_init,
        None,
        (self.rows,),
        jnp.int32,
    )

    if self.n_memory_text_entities is not None:
      self.memory_text_entities = self.variable(
          'constants',
          'memory_text_entities',
          self.bias_init,
          None,
          (self.rows, self.n_memory_text_entities),
          jnp.int32,
      )
    else:
      self.memory_text_entities = None

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
    self.final_encoder = make_transformer_block(self.num_final_layers)

    if self.num_intermediate_layers is not None:
      self.intermediate_encoder = make_transformer_block(
          self.num_intermediate_layers)
      self.intermediate_memory_attention_layer = make_attention_layer()
      self.final_memory_attention_layer = make_attention_layer()
    else:
      self.memory_attention_layer = make_attention_layer()

    self.mention_projector = nn.Dense(
        features=self.memory_value_dim
        if self.separate_memory_values else self.memory_key_dim,
        dtype=self.dtype,
    )

    self.apply_final_retrieval = (
        self.final_k_top_device is not None or self.final_splits is not None)

    if self.apply_final_retrieval:
      if self.final_k_top_device is None:
        raise ValueError('`final_k_top_device` must be specified for the final '
                         'retrieval')
      if self.final_splits is None:
        raise ValueError('`final_splits` must be specified for '
                         'the final retrieval')

      self.final_query_projector = nn.Dense(
          features=self.memory_key_dim,
          dtype=self.dtype,
      )

      self.final_memory_retrieval_layer = memory_retrieval_layer.MemoryRetrievalLayer(
          k_top_device=self.final_k_top_device,
          splits=self.final_splits,
          k_top_post_selection=self.final_k_top_post_selection,
      )

  def forward(
      self,
      batch,
      deterministic,
  ):
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

    memory_values = jnp.asarray(
        self.memory_values.value,
        dtype=self.dtype) if self.separate_memory_values else None
    memory_keys = jnp.asarray(self.memory_keys.value, dtype=self.dtype)
    memory_entity_ids = self.memory_entity_ids.value
    memory_identifiers = self.memory_identifiers.value

    loss_helpers['memory_values'] = memory_values
    loss_helpers['memory_keys'] = memory_keys
    loss_helpers['memory_entity_ids'] = memory_entity_ids
    loss_helpers['memory_identifiers'] = memory_identifiers

    def apply_memory_attention(memory_layer, encoding, prefix=''):
      encoding, mem_loss_helpers, mem_logging_helpers = memory_layer(
          encoded_input=encoding,
          mention_batch_positions=batch['mention_batch_positions'],
          mention_start_positions=batch['mention_start_positions'],
          mention_end_positions=batch['mention_end_positions'],
          mention_mask=batch['mention_mask'],
          memory_keys=memory_keys,
          memory_identifiers=memory_identifiers,
          memory_entity_ids=memory_entity_ids,
          deterministic=deterministic,
          memory_values=memory_values,
          text_identifiers=batch.get('text_identifiers', None),
          memory_text_entities=(self.memory_text_entities.value
                                if self.memory_text_entities is not None else
                                None),
          same_passage_memory_policy=self.same_passage_memory_policy,
      )
      loss_helpers.update(
          {prefix + key: value for key, value in mem_loss_helpers.items()})
      logging_helpers.update(
          {prefix + key: value for key, value in mem_logging_helpers.items()})
      return encoding

    if self.num_intermediate_layers is None:
      encoding = apply_memory_attention(self.memory_attention_layer, encoding)
    else:
      encoding = apply_memory_attention(
          self.intermediate_memory_attention_layer, encoding)
      encoding = self.intermediate_encoder(
          encoding=encoding,
          attention_mask=attention_mask,
          deterministic=deterministic)
      encoding = apply_memory_attention(self.final_memory_attention_layer,
                                        encoding, 'second_')
    encoding = self.final_encoder(
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=deterministic)

    if 'mention_target_batch_positions' in batch:
      mention_start_final_encodings = jut.matmul_2d_index_select(
          encoding, (batch['mention_target_batch_positions'],
                     batch['mention_target_start_positions']))
      mention_end_final_encodings = jut.matmul_2d_index_select(
          encoding, (batch['mention_target_batch_positions'],
                     batch['mention_target_end_positions']))

      loss_helpers['intermediate_target_mention_encodings'] = jut.matmul_slice(
          loss_helpers['memory_attention_mention_encodings'],
          batch['mention_target_indices'])
      if self.num_intermediate_layers is not None:
        loss_helpers[
            'second_intermediate_target_mention_encodings'] = jut.matmul_slice(
                loss_helpers['second_memory_attention_mention_encodings'],
                batch['mention_target_indices'])

      loss_helpers['target_mention_encodings'] = self.mention_projector(
          jnp.concatenate(
              (mention_start_final_encodings, mention_end_final_encodings),
              axis=-1))

      # Final retrieval layer is only applied over target mentions.
      if self.apply_final_retrieval:
        queries = self.final_query_projector(
            loss_helpers['target_mention_encodings'])

        retrieval_result = self.final_memory_retrieval_layer(
            queries=queries,
            memory_keys=memory_keys,
            memory_identifiers=memory_identifiers,
            memory_entity_ids=memory_entity_ids,
            memory_values=memory_values,
            text_identifiers=None,
            memory_text_entities=None,
            same_passage_memory_policy='disallow',
        )

        loss_helpers.update(
            {'final_' + k: v for k, v in retrieval_result.items()})

    return encoding, loss_helpers, logging_helpers

  @staticmethod
  def load_weights(config):
    """Load model weights and mention memory."""

    if config.load_weights == 'memory_only':
      model_variables = {}
    else:
      model_variables = base_encoder.BaseEncoder.load_weights(config)
    memory_variables = MentionMemoryEncoder.load_memory(config)
    model_variables['constants'] = memory_variables

    return model_variables

  @staticmethod
  def load_memory(config):
    """Load mention memory."""
    model_config = config.model_config
    encoder_config = model_config.encoder_config

    process_count = jax.process_count()
    # Reduce number of loaded memory shards by this proportion. Total shards
    # must be divisible by memory_reduction * process_count.
    memory_reduction = config.get('memory_reduction', 1)
    process_index = jax.process_index()
    local_devices = jax.local_devices()

    memory_prop = config.get('memory_prop', None)
    rows = encoder_config.rows
    memory_key_dim = encoder_config.memory_key_dim

    memory_arrays = {}
    memory_component_names = [
        'memory_keys', 'memory_identifiers', 'memory_entity_ids'
    ]
    # The following arrays should be converted to integer 32 type. The rest of
    # the arrays will converted to model type (typically, bfloat16 of float32).
    memory_component_int_dtypes = {
        'memory_identifiers', 'memory_entity_ids', 'memory_text_entities'
    }
    patterns = [
        config.memory_key_pattern, config.memory_id_pattern,
        config.memory_entity_id_pattern
    ]

    if encoder_config.separate_memory_values:
      memory_component_names.append('memory_values')
      patterns.append(config.memory_value_pattern)

    if config.get('same_entity_set_retrieval_weight', 0) > 0:
      memory_component_names.append('memory_text_entities')
      patterns.append(config.memory_text_entities_pattern)

    for key, pattern in zip(memory_component_names, patterns):
      memory_arrays[key] = data_utils.load_sharded_array(
          pattern, process_count * memory_reduction, process_index)

    memory_variables = {}

    cpu_device = jax.local_devices(backend='cpu')[0]
    dtype = encoder_config.dtype
    for key, array in memory_arrays.items():
      if memory_prop is not None:
        array = array[:int(memory_prop * array.shape[0])]
      if key == 'memory_keys':
        array = array.reshape(len(local_devices), rows, -1, memory_key_dim)
      else:
        array = array.reshape((len(local_devices), -1) + array.shape[1:])
      array = jax.device_put(array, cpu_device)
      if key in memory_component_int_dtypes:
        array = jnp.asarray(array, dtype=jnp.int32)
      else:
        array = jnp.asarray(array, dtype=dtype)
      array = jax.device_put_sharded(list(array), local_devices)
      memory_variables[key] = array
    return memory_variables

  @classmethod
  def make_output_postprocess_fn(
      cls,
      config  # pylint: disable=unused-argument
  ):
    """Postprocess task samples (input and output). See BaseTask."""

    return MemoryRetrievalResultProcessor(config)

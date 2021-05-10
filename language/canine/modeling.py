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
"""The main CANINE model and related functions."""

from typing import Optional, Sequence, Text

import dataclasses
from language.canine import bert_modeling
from language.canine import config_utils
from language.canine import local_attention
from language.canine import tensor_contracts as tc
import tensorflow.compat.v1 as tf

# Support up to 16 hash functions.
_PRIMES = [
    31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223
]

# While this should generally match `sys.maxunicode`, we want to provide this
# as a constant to avoid architecture/system-dependent array overruns.
LARGEST_CODEPOINT = 0x10ffff  # Decimal: 1,114,111


@dataclasses.dataclass
class CanineModelConfig(config_utils.Config):
  """Configuration for `CanineModel`."""

  # Character config:
  downsampling_rate: int = 4
  downsampling_kernel_size: int = 4
  upsampling_kernel_size: int = 4
  num_hash_functions: int = 8
  num_hash_buckets: int = 16384
  local_transformer_stride: int = 128  # Good TPU/XLA memory alignment.

  # Vanilla BERT config:
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_act: Text = "gelu"
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  type_vocab_size: int = 2
  max_positions: int = 16384
  initializer_range: float = 0.02


@tc.contract(
    tc.Require("a", shape=["batch", "seq", "dim"]),
    tc.Require("b", shape=["batch", "seq", "dim"]),
    tc.NamedDim("batch", "a", 0),
    tc.NamedDim("seq", "a", 1),
    tc.NamedDim("dim", "a", 2))
def _safe_add(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  return a + b


def _is_valid_codepoint(codepoints: tf.Tensor) -> tf.Tensor:
  return tf.logical_and(codepoints >= 0, codepoints <= LARGEST_CODEPOINT)


class CanineModel:
  """Main model for CANINE. See constructor for details."""

  def __init__(self,
               config: CanineModelConfig,
               atom_input_ids: tf.Tensor,
               atom_input_mask: tf.Tensor,
               atom_segment_ids: tf.Tensor,
               is_training: bool,
               final_seq_char_positions: Optional[tf.Tensor] = None):
    """Creates a `CanineModel`.

    This interface mirrors the `BertModel` class from the public BERT code, but
    abstracts away what type of input is passed (tokens, characters, etc.).

    A config file can be loaded like so:
    ```
    config = CanineModelConfig.from_json_file("/path/to.json")
    ```

    Args:
      config: Instance of `CanineModelConfig`.
      atom_input_ids: <int32>[batch_size, atom_seq_len] Vocabulary ids of the
        inputs.
      atom_input_mask: <int32>[batch_size, atom_seq_len] Indicates which input
        ids are non-padding.
      atom_segment_ids: <int32>[batch_size, atom_seq_len] Indicates the type of
        each feature. For a traditional BERT model with two segments, this would
        contain segment ids (0 and 1).
      is_training: Are we training? If not, disable dropout.
      final_seq_char_positions: Optional indices within each character sequence
        to be predicted by MLM. If specified, causes `get_sequence_output` to
        return only those positions, and, more importantly, when using a
        transformer for the `final_char_encoding`, only those sequence positions
        will be used as query positions for the transformer, giving a
        substantial boost in pre-training speed.
        <int32>[batch_size, max_predictions_per_seq]
    """

    self.config: CanineModelConfig = config
    self._is_training: bool = is_training

    if final_seq_char_positions is not None:
      batch_size, predictions_len = bert_modeling.get_shape_list(
          final_seq_char_positions)
      self._final_char_seq_length: tf.Tensor = predictions_len
    else:
      batch_size, char_seq_length = bert_modeling.get_shape_list(atom_input_ids)
      self._final_char_seq_length: tf.Tensor = char_seq_length
    self._batch_size = batch_size

    config.validate()

    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    batch_size, char_seq_length = bert_modeling.get_shape_list(atom_input_ids)
    del batch_size  # Unused.

    # `molecule_seq_length`: scalar int.
    molecule_seq_length = char_seq_length // config.downsampling_rate

    # Create attention masks...
    # `char_attention_mask`: <float>[batch, char_seq, char_seq]
    char_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
        atom_input_ids, atom_input_mask)

    # ...for attending from deep BERT molecule stack back to initial characters:
    # `molecule_to_char_attention_mask`: <float>[batch, molecule_seq, char_seq]
    molecule_to_char_attention_mask = self.downsample_attention_mask(
        char_attention_mask, config.downsampling_rate, dim=-2)

    # ...for attending from final character encoder to deep BERT stack:
    # `char_to_molecule_attention_mask`: <float>[batch, char_seq, molecule_seq]
    char_to_molecule_attention_mask = self.downsample_attention_mask(
        char_attention_mask, config.downsampling_rate, dim=-1)

    # ...for self-attention within deep BERT molecule stack:
    # `molecule_attention_mask`: <float>[batch, molecule_seq, molecule_seq]
    molecule_attention_mask = self.downsample_attention_mask(
        molecule_to_char_attention_mask, config.downsampling_rate, dim=-1)

    # The following lines have dimensions: <float>[batch, char_seq, char_dim].
    input_char_embedddings = self._embed_chars(
        codepoints=atom_input_ids, segment_ids=atom_segment_ids)

    # Contextualize character embeddings.
    input_char_encoding = self._encode_initial_chars(input_char_embedddings,
                                                     char_attention_mask)

    # Downsample chars to molecules.
    # The following lines have dimensions: [batch, molecule_seq, molecule_dim].
    # In this transformation, we change the dimensionality from `char_dim` to
    # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
    # the resnet connections (a) from the final char transformer stack back into
    # the original char transformer stack and (b) the resnet connections from
    # the final char transformer stack back into the deep BERT stack of
    # molecules.
    #
    # Empirically, it is critical to use a powerful enough transformation here:
    # mean pooling causes training to diverge with huge gradient norms in this
    # region of the model; using a convolution here resolves this issue. From
    # this, it seems that molecules and characters require a very different
    # feature space; intuitively, this makes sense.
    with tf.variable_scope("initial_char_encoder"):
      init_molecule_encoding = self._chars_to_molecules(
          input_char_encoding,
          expected_molecule_seq_length=molecule_seq_length)

    bert_layers: Sequence[tf.Tensor] = self._bert_stack(
        molecules_in=init_molecule_encoding,
        attention_mask=molecule_attention_mask)
    bert_molecule_encoding = bert_layers[-1]

    init_output_char_encoding = input_char_encoding

    self.final_char_encoding = self._encode_final_chars(
        init_output_char_encoding,
        char_attention_mask=char_attention_mask,
        full_molecules=bert_molecule_encoding,
        char_to_molecule_attention_mask=char_to_molecule_attention_mask,
        molecule_seq_length=molecule_seq_length,
        final_seq_char_positions=final_seq_char_positions)

    # For pooling (sequence-level tasks), we use only the output of the deep
    # BERT stack since we would end up with reduced dimensionality at each
    # character position.
    self.pooled = self._pool(bert_molecule_encoding)

    self.molecule_seq_length = molecule_seq_length
    self.downsampled_layers = bert_layers

  @tc.contract(
      tc.Require(
          "codepoints", shape=["batch", "char_seq"], dtype=tf.int32),
      tc.RequireTrue(_is_valid_codepoint, tensors=["codepoints"],
                     error="Expected `codepoints` to contain valid Unicode "
                           "codepoints."),
      tc.Ensure(
          tc.RESULT,
          dtype=tf.float32,
          shape=["batch", "char_seq", "char_dim"]),
      tc.NamedDim("batch", "codepoints", 0),
      tc.NamedDim("char_seq", "codepoints", 1),
      tc.NamedDim("char_dim", value_of="self.config.hidden_size"))
  def _embed_chars(self, codepoints: tf.Tensor,
                   segment_ids: tf.Tensor) -> tf.Tensor:
    """Lookup character embeddings given integer Unicode codepoints."""

    with tf.variable_scope("char_embeddings"):
      embed_seq = self._embed_hash_buckets(
          ids=codepoints,
          embedding_size=self.config.hidden_size,
          num_hashes=self.config.num_hash_functions,
          num_buckets=self.config.num_hash_buckets,
          initializer_range=self.config.initializer_range)
      dropout_prob = self.config.hidden_dropout_prob
      if self._is_training:
        dropout_prob = 0.0
      return bert_modeling.embedding_postprocessor(
          input_tensor=embed_seq,
          use_token_type=True,
          token_type_ids=segment_ids,
          token_type_vocab_size=16,
          token_type_embedding_name="segment_embeddings",
          use_position_embeddings=True,
          position_embedding_name="char_position_embeddings",
          initializer_range=self.config.initializer_range,
          max_position_embeddings=self.config.max_positions,
          dropout_prob=dropout_prob)

  @tc.contract(
      tc.Require("char_embed_seq", shape=["batch", "char_seq", "char_dim"]),
      tc.Ensure(tc.RESULT, shape=["batch", "char_seq", "char_dim"]),
      tc.NamedDim("batch", "char_embed_seq", 0),
      tc.NamedDim("char_seq", "char_embed_seq", 1),
      tc.NamedDim("char_dim", "char_embed_seq", 2))
  def _encode_initial_chars(self, char_embed_seq: tf.Tensor,
                            char_attention_mask: tf.Tensor) -> tf.Tensor:
    """Encode characters using shallow/low dim transformer."""
    with tf.variable_scope("initial_char_encoder"):
      return local_attention.local_transformer_model(
          input_tensor=char_embed_seq,
          attention_mask=char_attention_mask,
          hidden_size=self.config.hidden_size,
          num_hidden_layers=1,
          num_attention_heads=self.config.num_attention_heads,
          intermediate_size=self.config.intermediate_size,
          intermediate_act_fn=bert_modeling.get_activation(
              self.config.hidden_act),
          hidden_dropout_prob=self.config.hidden_dropout_prob,
          attention_probs_dropout_prob=(
              self.config.attention_probs_dropout_prob),
          initializer_range=self.config.initializer_range,
          always_attend_to_first_position=False,
          first_position_attends_to_all=False,
          attend_from_chunk_width=self.config.local_transformer_stride,
          attend_from_chunk_stride=self.config.local_transformer_stride,
          attend_to_chunk_width=self.config.local_transformer_stride,
          attend_to_chunk_stride=self.config.local_transformer_stride)

  @tc.contract(
      tc.Require("char_encoding", shape=["batch", "char_seq", "char_dim"]),
      tc.Ensure(
          tc.RESULT,
          shape=["batch", "molecule_seq", "molecule_dim"]),
      tc.NamedDim("batch", "char_encoding", 0),
      tc.NamedDim("char_seq", "char_encoding", 1),
      tc.NamedDim("char_dim", "char_encoding", 2),
      tc.NamedDim("molecule_seq", value_of="expected_molecule_seq_length"),
      tc.NamedDim("molecule_dim", value_of="self.config.hidden_size"))
  def _chars_to_molecules(
      self,
      char_encoding: tf.Tensor,
      expected_molecule_seq_length: tf.Tensor) -> tf.Tensor:
    """Convert char seq to initial molecule seq."""

    del expected_molecule_seq_length  # Used by contract only.

    with tf.variable_scope("initial_char_encoder/chars_to_molecules"):
      downsampled = tf.layers.conv1d(
          inputs=char_encoding,
          filters=self.config.hidden_size,
          kernel_size=self.config.downsampling_rate,
          strides=self.config.downsampling_rate,
          padding="valid",
          activation=bert_modeling.get_activation(self.config.hidden_act),
          name="conv")

      # `cls_encoding`: [batch, 1, hidden_size]
      cls_encoding = char_encoding[:, 0:1, :]

      # Truncate the last molecule in order to reserve a position for [CLS].
      # Often, the last position is never used (unless we completely fill the
      # text buffer). This is important in order to maintain alignment on TPUs
      # (i.e. a multiple of 128).
      downsampled_truncated = downsampled[:, 0:-1, :]

      # We also keep [CLS] as a separate sequence position since we always
      # want to reserve a position (and the model capacity that goes along
      # with that) in the deep BERT stack.
      # `result`: [batch, molecule_seq, molecule_dim]
      result = tf.concat([cls_encoding, downsampled_truncated], axis=1)

      return bert_modeling.layer_norm(result)

  @tc.contract(
      tc.Require("molecules_in", shape=["batch", "seq", "dim"]),
      tc.Require("attention_mask", shape=["batch", "seq", "seq"]),
      tc.Ensure(tc.RESULT, tuple_index=0, shape=["batch", "seq", "dim"]),
      tc.NamedDim("batch", "molecules_in", 0),
      tc.NamedDim("seq", "molecules_in", 1),
      tc.NamedDim("dim", "molecules_in", 2))
  def _bert_stack(self, molecules_in: tf.Tensor,
                  attention_mask: tf.Tensor) -> Sequence[tf.Tensor]:
    """Encode the molecules using a deep transformer stack."""
    with tf.variable_scope("bert"):
      return bert_modeling.transformer_model(
          input_tensor=molecules_in,
          attention_mask=attention_mask,
          hidden_size=self.config.hidden_size,
          num_hidden_layers=self.config.num_hidden_layers,
          num_attention_heads=self.config.num_attention_heads,
          intermediate_size=self.config.intermediate_size,
          intermediate_act_fn=bert_modeling.get_activation(
              self.config.hidden_act),
          hidden_dropout_prob=self.config.hidden_dropout_prob,
          attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
          initializer_range=self.config.initializer_range,
          do_return_all_layers=True)

  @tc.contract(
      tc.Require(
          "molecules", shape=["batch", "molecule_seq", "molecule_dim"]),
      tc.Require("char_seq_length", dtype=tf.int32, rank=0),
      tc.Require("molecule_seq_length", dtype=tf.int32, rank=0),
      tc.Ensure(tc.RESULT, shape=["batch", "char_seq", "molecule_dim"]),
      tc.NamedDim("batch", "molecules", 0),
      tc.NamedDim("molecule_seq", "molecules", 1),
      tc.NamedDim("molecule_dim", "molecules", 2),
      tc.NamedDim("char_seq", value_of="char_seq_length"))
  def _repeat_molecules(self, molecules: tf.Tensor, char_seq_length: tf.Tensor,
                        molecule_seq_length: tf.Tensor) -> tf.Tensor:
    """Repeats molecules to make them the same length as the char sequence."""

    del molecule_seq_length  # Used for contract only.

    rate = self.config.downsampling_rate

    molecules_without_extra_cls = molecules[:, 1:, :]
    # `repeated`: [batch_size, almost_char_seq_len, molecule_hidden_size]
    repeated = tf.repeat(molecules_without_extra_cls, repeats=rate, axis=-2)

    # So far, we've repeated the elements sufficient for any `char_seq_length`
    # that's a multiple of `downsampling_rate`. Now we account for the last
    # n elements (n < `downsampling_rate`), i.e. the remainder of floor
    # division. We do this by repeating the last molecule a few extra times.
    last_molecule = molecules[:, -1:, :]
    remainder_length = tf.floormod(char_seq_length, rate)
    remainder_repeated = tf.repeat(
        last_molecule,
        # +1 molecule to compensate for truncation.
        repeats=remainder_length + rate,
        axis=-2)

    # `repeated`: [batch_size, char_seq_len, molecule_hidden_size]
    return tf.concat([repeated, remainder_repeated], axis=-2)

  @tc.contract(
      tc.Require(
          "molecules", shape=["batch", "molecule_seq", "molecule_dim"]),
      tc.Require("expected_char_seq_length", dtype=tf.int32, rank=0),
      tc.Require("molecule_seq_length", dtype=tf.int32, rank=0),
      tc.Ensure(tc.RESULT, shape=["batch", "char_seq", "char_dim"]),
      tc.NamedDim("batch", "molecules", 0),
      tc.NamedDim("molecule_seq", "molecules", 1),
      tc.NamedDim("molecule_dim", "molecules", 2),
      tc.NamedDim("char_seq", value_of="expected_char_seq_length"),
      tc.NamedDim("char_dim", value_of="expected_char_dim"))
  def _molecules_to_chars(self, molecules: tf.Tensor,
                          molecule_seq_length: tf.Tensor,
                          expected_char_seq_length: tf.Tensor,
                          expected_char_dim: int) -> tf.Tensor:
    """Converts molecule seq back to a char seq."""

    del expected_char_dim  # Used by contract only.

    with tf.variable_scope("molecules_to_chars"):
      repeated = self._repeat_molecules(
          molecules,
          char_seq_length=expected_char_seq_length,
          molecule_seq_length=molecule_seq_length)

      if self.config.hidden_size == self.config.hidden_size:
        # If the dimensionality matches, just directly add a residual (not the
        # typical case).
        return repeated

      # Use a *slice* of the original features in order to create a residual
      # connection despite having different dimensions. This is a fairly
      # unusual (novel?) way of performing residual connections since they
      # typically assume uniform dimensionality.
      orig_features_for_residual = (
          repeated[:, :, :self.config.hidden_size])

      # Project molecules back to `char_dim`.
      result = bert_modeling.dense_layer_2d(
          repeated, self.config.hidden_size,
          bert_modeling.create_initializer(self.config.initializer_range), None,
          "dense")
      if self._is_training:
        result = bert_modeling.dropout(result, self.config.hidden_dropout_prob)
      # Add a resnet connection from the final character stack back through
      # the molecule transformer stack for a *slice* of the features.
      return bert_modeling.layer_norm(
          _safe_add(result, orig_features_for_residual))

  @tc.contract(
      tc.Require(
          "final_char_input_seq", shape=["batch", "char_seq", "init_char_dim"]),
      tc.Require(
          "char_attention_mask", dtype=tf.float32,
          shape=["batch", "char_seq", "char_seq"]),
      tc.Require(
          "full_molecules", shape=["batch", "molecule_seq", "molecule_dim"]),
      tc.Require(
          "char_to_molecule_attention_mask", dtype=tf.float32,
          shape=["batch", "char_seq", "molecule_seq"]),
      tc.Require("molecule_seq_length", dtype=tf.int32, rank=0),
      tc.Ensure(tc.RESULT, shape=["batch", "final_char_seq", "final_char_dim"]),
      tc.NamedDim("batch", "final_char_input_seq", 0),
      tc.NamedDim("char_seq", "final_char_input_seq", 1),
      tc.NamedDim("final_char_seq", value_of="self._final_char_seq_length"),
      tc.NamedDim("init_char_dim", "final_char_input_seq", 2),
      tc.NamedDim("final_char_dim",
                  value_of="self.config.hidden_size"),
      tc.NamedDim("molecule_seq", "full_molecules", 1),
      tc.NamedDim("molecule_dim", "full_molecules", 2))
  def _encode_final_chars(
      self,
      final_char_input_seq: tf.Tensor,
      char_attention_mask: tf.Tensor,
      full_molecules: tf.Tensor,
      char_to_molecule_attention_mask: tf.Tensor,
      molecule_seq_length: tf.Tensor,
      final_seq_char_positions: Optional[tf.Tensor]) -> tf.Tensor:
    """Run a shallow/low-dim transformer to get a final character encoding."""

    _, char_seq_length, _ = bert_modeling.get_shape_list(final_char_input_seq)

    # `final_char_input_seq` is a projected version of the deep molecule BERT
    # stack with slice-wise resnet connections.
    with tf.variable_scope("final_char_encoder"):
      # `repeated_molecules`: [batch_size, char_seq_len, molecule_hidden_size]
      repeated_molecules = self._repeat_molecules(
          full_molecules,
          char_seq_length=char_seq_length,
          molecule_seq_length=molecule_seq_length)
      layers = [final_char_input_seq, repeated_molecules]
      # `concat`:
      #     [batch_size, char_seq_len, molecule_hidden_size+char_hidden_final]
      concat = tf.concat(layers, axis=-1)

      # `result`: [batch_size, char_seq_len, hidden_size]
      result = tf.layers.conv1d(
          inputs=concat,
          filters=self.config.hidden_size,
          kernel_size=self.config.upsampling_kernel_size,
          strides=1,
          padding="same",
          activation=bert_modeling.get_activation(self.config.hidden_act),
          name="conv")
      result = bert_modeling.layer_norm(result)
      if self._is_training:
        result = bert_modeling.dropout(result,
                                       self.config.hidden_dropout_prob)
      final_char_seq = result

      if final_seq_char_positions is not None:
        # Limit transformer query seq and attention mask to these character
        # positions to greatly reduce the compute cost. Typically, this is just
        # done for the MLM training task.

        # `query_seq`: [batch, final_char_seq, char_dim]
        query_seq = tf.gather(
            final_char_seq, final_seq_char_positions, batch_dims=1)
        # `char_to_molecule_attention_mask`:
        #   [batch, final_len, molecule_seq]
        char_to_molecule_attention_mask = tf.gather(
            char_to_molecule_attention_mask,
            final_seq_char_positions,
            batch_dims=1)
        char_attention_mask = tf.gather(
            char_attention_mask,
            final_seq_char_positions,
            batch_dims=1)
      else:
        query_seq = final_char_seq
        # `char_to_molecule_attention_mask` remains unmodified.

      return bert_modeling.transformer_model(
          input_tensor=query_seq,
          input_kv_tensor=final_char_seq,
          attention_mask=char_attention_mask,
          hidden_size=self.config.hidden_size,
          num_hidden_layers=1,
          num_attention_heads=self.config.num_attention_heads,
          intermediate_size=self.config.intermediate_size,
          intermediate_act_fn=bert_modeling.get_activation(
              self.config.hidden_act),
          hidden_dropout_prob=self.config.hidden_dropout_prob,
          attention_probs_dropout_prob=(
              self.config.attention_probs_dropout_prob),
          initializer_range=self.config.initializer_range)

  @tc.contract(
      tc.Require("seq_to_pool",
                 shape=["batch", tc.Unchecked("seq"), "hidden_size"]),
      tc.Ensure(tc.RESULT, shape=["batch", "hidden_size"]),
      tc.NamedDim("batch", "seq_to_pool", 0),
      tc.NamedDim("hidden_size", "seq_to_pool", 2))
  def _pool(self, seq_to_pool: tf.Tensor) -> tf.Tensor:
    """Grab the [CLS] molecule for use in classification tasks."""
    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.
    with tf.variable_scope("pooler"):
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained.
      # This snippet is taken from vanilla BERT.
      first_token_tensor = tf.squeeze(seq_to_pool[:, 0:1, :], axis=1)
      return tf.layers.dense(
          first_token_tensor,
          self.config.hidden_size,
          activation=tf.tanh,
          kernel_initializer=bert_modeling.create_initializer(
              self.config.initializer_range))

  @tc.contract(
      tc.Require("char_attention_mask", dtype=tf.float32,
                 shape=["batch", tc.Unchecked("seq"), tc.Unchecked("seq")]),
      tc.Ensure(tc.RESULT, dtype=tf.float32,
                shape=["batch", tc.Unchecked("seq"), tc.Unchecked("seq")]),
      tc.NamedDim("batch", "char_attention_mask", 0))
  def downsample_attention_mask(self,
                                char_attention_mask: tf.Tensor,
                                downsampling_rate: int,
                                dim: int = -1) -> tf.Tensor:
    """Downsample one dimension of an attention mask."""
    perm = None
    if dim != -1:
      ndims = 3
      perm = list(range(ndims))
      # Swap desired dimension with last dimension at beginning/end of
      # function.
      perm[dim], perm[-1] = perm[-1], perm[dim]

    if perm is not None:
      char_attention_mask = tf.transpose(char_attention_mask, perm)

    # `poolable_char_mask`: <float>[batch, char_seq, char_seq, 1]
    poolable_char_mask = tf.expand_dims(char_attention_mask, axis=-1)

    # `poolable_char_mask`: <float>[batch, from_seq, to_seq, 1]
    pooled_molecule_mask = tf.nn.max_pool2d(
        input=poolable_char_mask,
        ksize=[1, downsampling_rate],
        strides=[1, downsampling_rate],
        padding="VALID")

    # `molecule_attention_mask`: <float>[batch, from_seq, to_seq]
    molecule_attention_mask = tf.squeeze(pooled_molecule_mask, axis=-1)

    if perm is not None:
      molecule_attention_mask = tf.transpose(molecule_attention_mask, perm)
    return molecule_attention_mask

  def _hash_bucket_tensors(self, ids: tf.Tensor, num_hashes: int,
                           num_buckets: int) -> Sequence[tf.Tensor]:
    """Converts ids to hash bucket ids via multiple hashing.

    Args:
      ids: The codepoints or other IDs to be hashed.
      num_hashes: The number of hash functions to use.
      num_buckets: The number of hash buckets (i.e. embeddings in each table).

    Returns:
      A sequence of tensors, each of which is the hash bucket IDs from one hash
      function.
    """
    if num_hashes > len(_PRIMES):
      raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

    primes = _PRIMES[:num_hashes]

    result_tensors = []
    for prime in primes:
      hashed = ((ids + 1) * prime) % num_buckets
      result_tensors.append(hashed)
    return result_tensors

  @tc.contract(
      tc.Require("ids", dtype=tf.int32, shape=["batch", "seq"]),
      tc.Ensure(tc.RESULT, dtype=tf.float32, shape=["batch", "seq", "dim"]),
      tc.NamedDim("batch", "ids", 0),
      tc.NamedDim("seq", "ids", 1),
      tc.NamedDim("dim", value_of="embedding_size"))
  def _embed_hash_buckets(self, ids: tf.Tensor, embedding_size: int,
                          num_hashes: int, num_buckets: int,
                          initializer_range: int) -> tf.Tensor:
    """Converts IDs (e.g. codepoints) into embeddings via multiple hashing.

    Args:
      ids: The codepoints or other IDs to be hashed.
      embedding_size: The dimensionality of the returned embeddings.
      num_hashes: The number of hash functions to use.
      num_buckets: The number of hash buckets (i.e. embeddings in each table).
      initializer_range: Maximum absolute value for initial weights.

    Returns:
      The codepoint emeddings.
    """

    if embedding_size % num_hashes != 0:
      raise ValueError(f"Expected `embedding_size` ({embedding_size}) % "
                       f"`num_hashes` ({num_hashes}) == 0")

    shard_embedding_size = embedding_size // num_hashes

    hash_bucket_tensors = self._hash_bucket_tensors(
        ids, num_hashes=num_hashes, num_buckets=num_buckets)
    embedding_shards = []
    for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
      embedding_table = tf.get_variable(
          name=f"embeddings/HashBucketCodepointEmbedder_{i}",
          shape=[num_buckets, shard_embedding_size],
          initializer=bert_modeling.create_initializer(initializer_range))
      shard_embeddings = tf.nn.embedding_lookup(embedding_table,
                                                hash_bucket_ids)
      embedding_shards.append(shard_embeddings)
    return tf.concat(embedding_shards, axis=-1)

  @tc.contract(
      tc.Ensure(tc.RESULT, tuple_index=-1, dtype=tf.float32,
                shape=["batch", "downsampled_seq", "hidden_size"]),
      tc.NamedDim("batch", value_of="self._batch_size"),
      tc.NamedDim("downsampled_seq", value_of="self.molecule_seq_length"),
      tc.NamedDim("hidden_size", value_of="self.config.hidden_size"))
  def get_downsampled_layers(self) -> Sequence[tf.Tensor]:
    """Gets a sequence representation, one position per character."""
    assert len(self.downsampled_layers) == self.config.num_hidden_layers
    return self.downsampled_layers

  @tc.contract(
      tc.Ensure(tc.RESULT, dtype=tf.float32,
                shape=["batch", "char_seq", "hidden_size"]),
      tc.NamedDim("batch", value_of="self._batch_size"),
      tc.NamedDim("char_seq", value_of="self._final_char_seq_length"),
      tc.NamedDim("hidden_size",
                  value_of="self.config.hidden_size"))
  def get_sequence_output(self) -> tf.Tensor:
    """Gets a sequence representation, one position per character."""
    return self.final_char_encoding

  @tc.contract(
      tc.Ensure(tc.RESULT, dtype=tf.float32, shape=["batch", "hidden_size"]),
      tc.NamedDim("batch", value_of="self._batch_size"),
      tc.NamedDim("hidden_size", value_of="self.config.hidden_size"))
  def get_pooled_output(self) -> tf.Tensor:
    """Gets a single sequence representation for classification."""
    return self.pooled

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
"""Contains utils for masking passages."""

from typing import Dict, Optional

from language.mentionmemory.utils import default_values
import numpy as np
import tensorflow.compat.v2 as tf


def non_zero_1d(tensor: tf.Tensor):
  """Return positions of non-zero elements in the input 1D tensor."""
  return tf.squeeze(tf.cast(tf.where(tf.not_equal(tensor, 0)), tensor.dtype), 1)


def sparse_to_dense_1d(sparse_values: tf.Tensor, seq_length: int):
  """Convert sparse tensor ([0, 1, 4]) to dense tensor ([1, 1, 0, 0, 1])."""
  updates = tf.fill(tf.shape(sparse_values), value=1)
  updates = tf.cast(updates, sparse_values.dtype)
  return tf.scatter_nd(tf.expand_dims(sparse_values, 1), updates, [seq_length])


def random_choice_1d(num_values: tf.Tensor, num_to_sample: tf.Tensor):
  """Samples num_to_sample elements from [0; num_values) without replacement."""
  num_to_sample = tf.minimum(num_to_sample, num_values)
  _, sampled_spans = tf.math.top_k(
      tf.random.uniform([num_values]), k=num_to_sample)
  return sampled_spans


def get_dense_is_inside_for_dense_spans(
    dense_start_positions: tf.Tensor,
    dense_end_positions: tf.Tensor) -> tf.Tensor:
  """Dense mask whether position is inside span given dense starts / ends."""
  # `tf.cumsum(dense_start_positions)[i]` computes how many spans start before
  # or on the i-th position.
  # `tf.cumsum(dense_end_positions, exclusive=True`) computes how many spans
  #  ends strictly before i-th positions.
  # Their difference is equal to how many spans start before i-th position and
  # end after or on i-th position. This is precisely how many spans contain
  # i-th position.
  is_inside_span = (
      tf.cumsum(dense_start_positions) -
      tf.cumsum(dense_end_positions, exclusive=True))
  # Adjust for the case of overlapping spans
  is_inside_span = tf.minimum(is_inside_span, 1)
  return is_inside_span


def get_dense_is_inside_for_sparse_spans(sparse_start_positions: tf.Tensor,
                                         sparse_end_positions: tf.Tensor,
                                         seq_length: int) -> tf.Tensor:
  """Dense mask whether position is inside span given sparse starts / ends."""
  dense_start_positions = sparse_to_dense_1d(sparse_start_positions, seq_length)
  dense_end_positions = sparse_to_dense_1d(sparse_end_positions, seq_length)
  return get_dense_is_inside_for_dense_spans(dense_start_positions,
                                             dense_end_positions)


def dynamic_padding_1d(tensor: tf.Tensor,
                       length: int,
                       padding_token_id: int = 0) -> tf.Tensor:
  """Padds or truncates 1D tensor to a specified length."""
  length_to_pad = length - tf.shape(tensor)[0]
  length_to_pad_or_zero = tf.maximum(length_to_pad, 0)
  length_to_pad_or_zero = tf.expand_dims(length_to_pad_or_zero, 0)
  paddings = tf.concat([tf.constant([0]), length_to_pad_or_zero], axis=0)
  paddings = tf.expand_dims(paddings, 0)

  def pad():
    return tf.pad(
        tensor, paddings, 'CONSTANT', constant_values=padding_token_id)

  padded_tensor = tf.cond(
      length_to_pad > 0, true_fn=pad, false_fn=lambda: tensor[:length])
  padded_tensor.set_shape(length)
  return padded_tensor


def mask_tokens_by_spans(
    text_ids: tf.Tensor, start_positions: tf.Tensor, end_positions: tf.Tensor,
    mask_rate: float, num_positions_to_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
  """Mask `mask_rate` fraction of spans."""
  # Compute the number of spans to sample
  num_spans = tf.shape(start_positions)[0]
  num_spans_to_mask = tf.cast(num_spans, tf.float32) * mask_rate
  num_spans_to_mask = tf.cast(num_spans_to_mask, tf.int32)
  # `num_positions_to_mask` is an upperbound to how many spans we need to mask
  num_spans_to_mask = tf.minimum(num_spans_to_mask, num_positions_to_mask)

  # Sample spans and corresponding positions
  sampled_spans = random_choice_1d(num_spans, num_spans_to_mask)
  sampled_start_positions = tf.gather(start_positions, sampled_spans)
  sampled_end_positions = tf.gather(end_positions, sampled_spans)

  # Get text positions that needs to be masked
  seq_length = tf.shape(text_ids)[0]
  dense_positions = get_dense_is_inside_for_sparse_spans(
      sampled_start_positions, sampled_end_positions, seq_length)
  sparse_positions = non_zero_1d(dense_positions)

  # Adjust the number of positions if sampled too many of them
  num_masked_positions = tf.shape(sparse_positions)[0]
  sampled_masked_positions = random_choice_1d(num_masked_positions,
                                              num_positions_to_mask)
  sampled_sparse_positions = tf.gather(sparse_positions,
                                       sampled_masked_positions)
  return sampled_sparse_positions


def mask_mentions_and_tokens_tf(
    text_ids: tf.Tensor,
    text_mask: tf.Tensor,
    dense_span_starts: tf.Tensor,
    dense_span_ends: tf.Tensor,
    non_mention_mask_rate: float,
    mention_mask_rate: float,
    max_mlm_targets: int,
    mask_token_id: int,
    vocab_size: int,
    random_replacement_prob: float = 0.1,
    identity_replacement_prob: float = 0.1,
) -> Dict[str, tf.Tensor]:
  """Randomly masks whole mentions and random tokens up to a maximum.

  First, mentions are masked according to mention mask rate. If a mention is
  masked, all tokens in the mention are replaced by the mask token. If the
  passage has at least one mention and the mention rask rate is greater than
  zero, we mask at least one mention.

  After masking mentions, if there are fewer masked tokens than maximum mlm
  targets, we additionally mask non-mention words. TODO: If a token in a word
  is masked, all tokens in the word are masked. Some proportion of targets are
  not masked to ameliorate pretrain-finetune mismatch. If there are insufficient
  masked tokens, the target array is padded up to max targets.

  Args:
    text_ids: [seq_length] tensor with token ids.
    text_mask: [seq_length] tensor with 1s for tokens and 0 for padding.
    dense_span_starts: [seq_length] tensor with 1s for mention start positions
      and 0 otherwise.
    dense_span_ends: [seq_length] tensor with 1s for mention end positions and 0
      otherwise.
    non_mention_mask_rate: percentage of non mention tokens to be masked.
    mention_mask_rate: percentage of mentions to be masked.
    max_mlm_targets: total number of mlm targets.
    mask_token_id: token id for mask token.
    vocab_size: vocabulary size.
    random_replacement_prob: probability that to-be-masked token will be
      replaced with a random token instead of [MASK].
    identity_replacement_prob: probability that to-be-masked token will be
      replaced with itself instead of [MASK].

  Returns:
    Dictionary with masked text, mask positions, target ids, target weights.
  """
  # Mask mentions
  mention_start_positions = non_zero_1d(dense_span_starts)
  mention_end_positions = non_zero_1d(dense_span_ends)
  mention_masked_positions = mask_tokens_by_spans(text_ids,
                                                  mention_start_positions,
                                                  mention_end_positions,
                                                  mention_mask_rate,
                                                  max_mlm_targets)

  dense_is_mention = get_dense_is_inside_for_dense_spans(
      dense_span_starts, dense_span_ends)
  dense_is_not_mention = 1 - dense_is_mention
  dense_is_not_mention = dense_is_not_mention * text_mask

  # Mask non-mentions
  non_mention_start_positions = non_zero_1d(dense_is_not_mention)
  # TODO(urikz): Implement whole-word masking
  non_mention_end_positions = non_mention_start_positions
  non_mention_masked_positions = mask_tokens_by_spans(
      text_ids, non_mention_start_positions, non_mention_end_positions,
      non_mention_mask_rate,
      max_mlm_targets - tf.shape(mention_masked_positions)[0])

  # Merge masked positions for mention and non-mention tokens
  mlm_target_positions = tf.concat(
      [mention_masked_positions, non_mention_masked_positions], -1)
  n_mlm_target_positions = tf.shape(mlm_target_positions)

  # Get target IDs, weights and other features
  mlm_target_ids = tf.gather(text_ids, mlm_target_positions)
  mlm_target_weights = tf.ones(n_mlm_target_positions, dtype=tf.int64)
  mlm_target_is_mention = tf.ones(
      tf.shape(mention_masked_positions), dtype=tf.int64)
  seq_length = tf.shape(text_ids)[0]
  dense_is_masked = sparse_to_dense_1d(mlm_target_positions, seq_length)

  # Replace masked tokens with [MASK], random or original tokens.
  replacement_scores = tf.random.uniform(n_mlm_target_positions)
  replacement_tokens = tf.where(
      replacement_scores > random_replacement_prob + identity_replacement_prob,
      # replace tokens with [MASK]
      tf.cast(
          tf.fill(n_mlm_target_positions, value=mask_token_id), dtype=tf.int64),
      tf.where(
          replacement_scores > random_replacement_prob,
          # keep original
          mlm_target_ids,
          # replace with random token
          tf.random.uniform(
              n_mlm_target_positions, maxval=vocab_size, dtype=tf.int64)))
  replacement_positions = tf.expand_dims(mlm_target_positions, 1)
  # Indicies should be tf.int32 only.
  replacement_positions = tf.cast(replacement_positions, tf.int32)
  replacement_tokens = tf.scatter_nd(replacement_positions, replacement_tokens,
                                     tf.shape(text_ids))
  masked_text_ids = (
      text_ids * (1 - dense_is_masked) + replacement_tokens * dense_is_masked)

  return {
      'masked_text_ids':
          masked_text_ids,
      'mlm_target_positions':
          dynamic_padding_1d(mlm_target_positions, max_mlm_targets),
      'mlm_target_ids':
          dynamic_padding_1d(mlm_target_ids, max_mlm_targets),
      'mlm_target_weights':
          dynamic_padding_1d(mlm_target_weights, max_mlm_targets),
      'mlm_target_is_mention':
          dynamic_padding_1d(mlm_target_is_mention, max_mlm_targets),
      'dense_is_masked':
          dense_is_masked,
  }


def mask_mentions_and_tokens(
    text_ids: np.ndarray,
    text_mask: np.ndarray,
    mention_start_positions: np.ndarray,
    mention_end_positions: np.ndarray,
    mask_rate: float,
    mention_mask_rate: float,
    max_mlm_targets: int,
    mask_token_id: int,
) -> Dict[str, np.ndarray]:
  """Randomly masks whole mentions and random tokens up to a maximum.

  First, mentions are masked according to mention mask rate. If a mention is
  masked, all tokens in the mention are replaced by the mask token. If the
  passage has at least one mention and the mention rask rate is greater than
  zero, we mask at least one mention.

  After masking mentions, if there are fewer masked tokens than maximum mlm
  targets, we additionally mask non-mention words. TODO: If a token in a word
  is masked, all tokens in the word are masked. Some proportion of targets are
  not masked to ameliorate pretrain-finetune mismatch. If there are insufficient
  masked tokens, the target array is padded up to max targets.

  Args:
    text_ids: array of token ids.
    text_mask: array with 1s for tokens and 0 for padding.
    mention_start_positions: array of sparse mention start positions.
    mention_end_positions: array of sparse mention end positions.
    mask_rate: percentage of normal tokens to be masked.
    mention_mask_rate: percentage of mentions to be masked.
    max_mlm_targets: total number of mlm targets.
    mask_token_id: token id for mask token.

  Returns:
    Masked text, mask positions, target ids, target weights
  """

  n_tokens = text_mask.sum(dtype=np.int64)

  n_mentions = len(mention_start_positions)
  n_sample_mentions = int(mention_mask_rate * n_mentions)
  # Mask at least one mention
  if n_sample_mentions == 0 and n_mentions > 0 and mention_mask_rate > 0.0:
    n_sample_mentions = 1

  # Mask mentions
  if n_sample_mentions > 0:
    sample_mention_mask_indices = np.random.choice(
        n_mentions, n_sample_mentions, replace=False)
    sample_mention_mask_indices.sort()
    mention_token_mask_positions = np.concatenate([
        np.arange(mention_start_positions[idx], mention_end_positions[idx] + 1)
        for idx in sample_mention_mask_indices
    ])
  else:
    mention_token_mask_positions = np.zeros(0, dtype=np.int64)

  if n_mentions == 0:
    non_mention_positions = np.arange(n_tokens)
  else:
    non_mention_spans = [
        np.arange(mention_end_positions[idx] + 1,
                  mention_start_positions[idx + 1])
        for idx in range(n_mentions - 1)
    ]

    non_mention_spans = [np.arange(mention_start_positions[0])
                        ] + non_mention_spans + [
                            np.arange(mention_end_positions[-1] + 1, n_tokens)
                        ]
    non_mention_positions = np.concatenate(non_mention_spans)

  # Mask non-mention words
  n_remaining_masks = max_mlm_targets - len(mention_token_mask_positions)
  if n_remaining_masks < 0:
    mention_token_mask_positions = mention_token_mask_positions[:
                                                                max_mlm_targets]
    all_mask_positions = mention_token_mask_positions
  else:
    n_non_mention_masks = min(n_remaining_masks,
                              int(mask_rate * len(non_mention_positions)))
    non_mention_mask_positions = np.random.choice(
        non_mention_positions, n_non_mention_masks, replace=False)
    all_mask_positions = np.concatenate(
        (mention_token_mask_positions, non_mention_mask_positions))

  masked_text_ids = np.copy(text_ids)
  masked_text_ids[all_mask_positions] = mask_token_id

  mlm_pad_shape = (0, max_mlm_targets - len(all_mask_positions))
  mlm_target_positions = np.pad(
      all_mask_positions, mlm_pad_shape, mode='constant')
  mlm_target_ids = text_ids[all_mask_positions]
  mlm_target_ids = np.pad(mlm_target_ids, mlm_pad_shape, mode='constant')
  mlm_target_weights = np.ones(len(all_mask_positions), dtype=np.float32)
  mlm_target_weights = np.pad(
      mlm_target_weights, mlm_pad_shape, mode='constant')

  mlm_target_is_mention = np.zeros_like(mlm_target_ids)
  mlm_target_is_mention[:len(mention_token_mask_positions)] = 1

  dense_is_masked = np.zeros_like(text_ids)
  dense_is_masked[all_mask_positions] = 1

  return {
      'masked_text_ids': masked_text_ids,
      'mlm_target_positions': mlm_target_positions,
      'mlm_target_ids': mlm_target_ids,
      'mlm_target_weights': mlm_target_weights,
      'mlm_target_is_mention': mlm_target_is_mention,
      'dense_is_masked': dense_is_masked,
  }


def get_dense_span_ends_from_starts(dense_span_starts: tf.Tensor,
                                    dense_span_ends: tf.Tensor) -> tf.Tensor:
  """For every mention start positions finds the corresponding end position."""
  seq_len = tf.shape(dense_span_starts)[0]
  start_pos = tf.cast(tf.where(tf.equal(dense_span_starts, 1)), tf.int32)
  end_pos = tf.cast(
      tf.squeeze(tf.where(tf.equal(dense_span_ends, 1)), 1), tf.int32)
  dense_span_ends_from_starts = tf.zeros(seq_len, dtype=tf.int32)
  dense_span_ends_from_starts = tf.tensor_scatter_nd_add(
      dense_span_ends_from_starts, start_pos, end_pos)
  return dense_span_ends_from_starts


def _flatten(tensor):
  return tf.reshape(tensor, [-1])


def prepare_mention_target_features(
    mention_batch_positions: tf.Tensor,
    mention_start_positions: tf.Tensor,
    mention_end_positions: tf.Tensor,
    mention_mask: tf.Tensor,
    mention_target_weights: tf.Tensor,
    mention_target_indices: tf.Tensor,
) -> Dict[str, tf.Tensor]:
  """Produce mention target features based on batchwise mention features."""
  mention_target_weights = mention_target_weights * tf.gather(
      mention_mask, mention_target_indices)
  mention_target_batch_positions = tf.gather(mention_batch_positions,
                                             mention_target_indices)
  mention_target_start_positions = tf.gather(mention_start_positions,
                                             mention_target_indices)
  mention_target_end_positions = tf.gather(mention_end_positions,
                                           mention_target_indices)
  return {
      'mention_target_batch_positions': mention_target_batch_positions,
      'mention_target_start_positions': mention_target_start_positions,
      'mention_target_end_positions': mention_target_end_positions,
      'mention_target_weights': mention_target_weights,
  }


def process_batchwise_mention_targets(
    dense_span_starts: tf.Tensor,
    dense_span_ends: tf.Tensor,
    dense_mention_ids: tf.Tensor,
    dense_linked_mention_mask: tf.Tensor,
    dense_is_masked: tf.Tensor,
    max_mentions: int,
    max_mention_targets: int,
) -> Dict[str, tf.Tensor]:
  """Processes mention targets and subsamples/pads as necessary.

  This function does two things. First, it selects which mentions to mark as
  mentions for mention-aware text encoders (in case the number of mentions
  exceeds the max number of mentions). Second, it selects which linked
  mentions to use as targets for mention objectives. To reduce subsampling and
  padding, the function operates over all mentions in a batch, generating
  flattened arrays. The encoder reconstructs the original mention positions
  from an array which specifies each mention's position in the batch. Linked
  mentions are given priority for sampling.

  Args:
    dense_span_starts: dense mention start positions.
    dense_span_ends: dense mention end positions.
    dense_mention_ids: dense entity ids for linked mentions in passage.
    dense_linked_mention_mask: dense mask for linked mentions in passage.
    dense_is_masked: dense mask for masked positions in passage.
    max_mentions: max number of mentions to be considered in model.
    max_mention_targets: max number of mentions to be used for linking loss.

  Returns:
    Mention starts, mention ends, mention mask,
    mention target indices (into start/end positions),
    mention target ids, mention target weights, mention_target_batch_positions,
    mention_target_start_positions, mention_target_end_positions
  """

  seq_len = tf.shape(dense_span_starts)[1]

  # The linking mask has 1's for every part of the mention, we only
  # want it for starts...
  linking_mask_start_indexed = dense_span_starts * dense_linked_mention_mask

  # values in {0, 1, 2}:
  # 0: not a masking location.
  # 1: a masking location.
  # 2: a masking and linking location.
  prioritized_span_starts = dense_span_starts + linking_mask_start_indexed
  prioritized_span_starts = tf.cast(prioritized_span_starts, tf.float32)

  # Add random [0; 1) values for a uniform sampling in case
  # there are more mention than `max_mentions`
  prioritized_span_starts += tf.random.uniform(
      tf.shape(prioritized_span_starts))

  _, global_start_indices = tf.math.top_k(
      _flatten(prioritized_span_starts), k=max_mentions)

  dense_span_starts_flatten = _flatten(dense_span_starts)
  dense_span_ends_at_starts = get_dense_span_ends_from_starts(
      dense_span_starts_flatten, _flatten(dense_span_ends))
  global_end_indices = tf.gather(dense_span_ends_at_starts,
                                 global_start_indices)

  dtype = dense_span_starts.dtype
  mention_batch_positions = tf.math.floordiv(global_start_indices, seq_len)
  mention_batch_positions = tf.cast(mention_batch_positions, dtype=dtype)
  mention_start_positions = tf.math.floormod(global_start_indices, seq_len)
  mention_start_positions = tf.cast(mention_start_positions, dtype=dtype)
  mention_end_positions = tf.math.floormod(global_end_indices, seq_len)
  mention_end_positions = tf.cast(mention_end_positions, dtype=dtype)
  mention_mask = tf.gather(dense_span_starts_flatten, global_start_indices)
  mention_mask = tf.cast(mention_mask, dtype=dtype)
  mention_batch_positions *= mention_mask
  mention_start_positions *= mention_mask
  mention_end_positions *= mention_mask

  mention_target_weights = tf.gather(
      _flatten(linking_mask_start_indexed), global_start_indices)
  mention_target_weights = mention_target_weights[:max_mention_targets]
  mention_target_weights = tf.cast(mention_target_weights, dtype=dtype)
  mention_target_indices = tf.range(max_mention_targets, dtype=dtype)
  mention_target_indices = mention_target_indices * mention_target_weights
  mention_target_ids = tf.gather(
      _flatten(dense_mention_ids), global_start_indices)
  mention_target_ids = mention_target_ids[:max_mention_targets]
  mention_target_ids = tf.cast(mention_target_ids, dtype=dtype)
  mention_target_ids = mention_target_ids * mention_target_weights
  indices = tf.stack((mention_batch_positions, mention_start_positions), axis=1)
  mention_is_masked = tf.gather_nd(dense_is_masked, indices)
  mention_target_is_masked = mention_is_masked[:max_mention_targets]

  features = {
      'mention_batch_positions': mention_batch_positions,
      'mention_start_positions': mention_start_positions,
      'mention_end_positions': mention_end_positions,
      'mention_mask': mention_mask,
      'mention_is_masked': mention_is_masked,
      'mention_target_ids': mention_target_ids,
      'mention_target_indices': mention_target_indices,
      'mention_target_is_masked': mention_target_is_masked,
  }
  mention_target_features = prepare_mention_target_features(
      mention_batch_positions, mention_start_positions, mention_end_positions,
      mention_mask, mention_target_weights, mention_target_indices)
  features.update(mention_target_features)
  return features


def _batched_range(batch_size: int, length: int, axis: int,
                   dtype: tf.dtypes.DType) -> tf.Tensor:
  """Produces multiple tf.range stacked along the `axis` `batch_size` times."""
  if axis == 0:
    repeats = [batch_size, 1]
  elif axis == 1:
    repeats = [1, batch_size]
  else:
    raise ValueError('`_batched_range` only accepts axis argument being 0 or 1')
  return tf.tile(tf.expand_dims(tf.range(length, dtype=dtype), axis), repeats)


def _get_2d_index(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Generates 2D index given separate arrays for each coordinate."""
  # Index is always int32
  return tf.cast(tf.stack([x, y], axis=1), tf.int32)


def compute_positions_shift_with_entity_tokens(
    mention_mask: tf.Tensor,
    mention_batch_positions: tf.Tensor,
    mention_start_positions: tf.Tensor,
    mention_end_positions: tf.Tensor,
    batch_size: int,
    old_length: int,
) -> tf.Tensor:
  """Computes the new position for every position in the old sequence."""
  old_shape = (batch_size, old_length)

  def get_positions_shift(positions: tf.Tensor, exclusive: bool) -> tf.Tensor:
    index_2d = _get_2d_index(mention_batch_positions, positions)
    return tf.cumsum(
        tf.scatter_nd(index_2d, mention_mask, old_shape),
        axis=1,
        exclusive=exclusive)

  # We start with identity mapping from old positions to new positions.
  positions = _batched_range(batch_size, old_length, 0, mention_mask.dtype)
  # We need to insert entity tokens BEFORE the first token of every mention.
  # (note `exclusive=False`). Therefore, all the tokens after will need to be
  # shifted right by 1.
  positions += get_positions_shift(mention_start_positions, exclusive=False)
  # We need to insert entity tokens AFTER the last token of every mention.
  # (note `exclusive=True`). Therefore, all the tokens after will need to be
  # shifted right by 1.
  positions += get_positions_shift(mention_end_positions, exclusive=True)
  return positions


def compute_which_mentions_fit_with_entity_tokens(
    mention_mask: tf.Tensor,
    mention_batch_positions: tf.Tensor,
    mention_start_positions: tf.Tensor,
    mention_end_positions: tf.Tensor,
    batch_size: int,
    old_length: int,
    new_length: int,
) -> tf.Tensor:
  """Computes a mask for which mentions will fit after adding entity tokens."""

  positions = compute_positions_shift_with_entity_tokens(
      mention_mask, mention_batch_positions, mention_start_positions,
      mention_end_positions, batch_size, old_length)

  def get_new_positions(old_positions):
    index_2d = _get_2d_index(mention_batch_positions, old_positions)
    return tf.gather_nd(positions, index_2d)

  # `get_new_positions(mention_end_positions)` returns new positions for the
  # last token per mention. However, we want `mention_end_positions` to point
  # to the entity end token, so we add 1.
  new_mention_end_positions = get_new_positions(mention_end_positions) + 1

  new_mention_mask = tf.less(new_mention_end_positions, new_length)
  return tf.cast(new_mention_mask, mention_mask.dtype)


def add_entity_tokens(
    text_ids: tf.Tensor,
    text_mask: tf.Tensor,
    mention_mask: tf.Tensor,
    mention_batch_positions: tf.Tensor,
    mention_start_positions: tf.Tensor,
    mention_end_positions: tf.Tensor,
    new_length: int,
    mlm_target_positions: Optional[tf.Tensor] = None,
    mlm_target_weights: Optional[tf.Tensor] = None,
    entity_start_token_id: int = default_values.ENTITY_START_TOKEN,
    entity_end_token_id: int = default_values.ENTITY_END_TOKEN,
) -> Dict[str, tf.Tensor]:
  """Adds entity start / end tokens around mentions.

  Inserts `entity_start_token_id` and `entity_end_token_id` tokens around each
  mention and update mention_start_positions / mention_end_positions to point
  to these tokens.

  New text length will be `new_length` and texts will be truncated if nessesary.
  If a mention no longer fits into the new text, its mask (`mention_mask`) will
  be set to 0.

  The function can also update MLM position and weights (`mlm_target_positions`
  and `mlm_target_weights`) if these arguments are provided. Similarly to
  mentions, if MLM position no longer fits into the new text, its mask
  (`mlm_target_weights`) will be set to 0.

  Args:
    text_ids: [seq_length] tensor with token ids.
    text_mask: [seq_length] tensor with 1s for tokens and 0 for padding.
    mention_mask: [n_mentions] mask indicating whether a mention is a padding.
    mention_batch_positions: [n_mentions] sample ID of a mention in the batch.
    mention_start_positions: [n_mentions] position of a mention first token
      within a sample.
    mention_end_positions: [n_mentions] position of a mention last token within
      a sample.
    new_length: new length of text after entity tokens will be added.
    mlm_target_positions: [batch_size, max_mlm_targets] positions of tokens to
      be used for MLM task.
    mlm_target_weights: [batch_size, max_mlm_targets] mask indicating whether
      `mlm_target_positions` is a padding.
    entity_start_token_id: token to be used as entity start token.
    entity_end_token_id: token to be used as entity end token.

  Returns:
    New text_ids and text_mask, updated mentions positions including
    mention_start_positions, mention_end_positions and mention_mask.
    Returns updated mlm_target_positions and mlm_target_weights if they were
    provided as arguments.
  """
  batch_size = tf.shape(text_ids)[0]
  old_length = tf.shape(text_ids)[1]
  new_shape = (batch_size, new_length)

  mentions_fit_mask = compute_which_mentions_fit_with_entity_tokens(
      mention_mask,
      mention_batch_positions,
      mention_start_positions,
      mention_end_positions,
      batch_size,
      old_length,
      new_length,
  )
  # Ignore mentions that does not fit into new texts.
  new_mention_mask = mention_mask * mentions_fit_mask
  mention_start_positions = mention_start_positions * new_mention_mask
  mention_end_positions = mention_end_positions * new_mention_mask

  positions = compute_positions_shift_with_entity_tokens(
      new_mention_mask, mention_batch_positions, mention_start_positions,
      mention_end_positions, batch_size, old_length)

  def get_2d_index(positions: tf.Tensor) -> tf.Tensor:
    return _get_2d_index(mention_batch_positions, positions)

  def get_new_positions(old_positions: tf.Tensor) -> tf.Tensor:
    index_2d = get_2d_index(old_positions)
    return tf.gather_nd(positions, index_2d)

  new_mention_start_positions = get_new_positions(mention_start_positions) - 1
  new_mention_start_positions = new_mention_start_positions * new_mention_mask
  new_mention_end_positions = get_new_positions(mention_end_positions) + 1
  new_mention_end_positions = new_mention_end_positions * new_mention_mask

  if mlm_target_positions is not None:
    if mlm_target_weights is None:
      raise ValueError('`mlm_target_weights` must be specified if '
                       '`mlm_target_positions` is provided.')
    mlm_target_positions = tf.gather(
        positions, mlm_target_positions, batch_dims=1)
    mlm_target_positions_within_len = tf.less(mlm_target_positions, new_length)
    mlm_target_positions_within_len = tf.cast(mlm_target_positions_within_len,
                                              mlm_target_weights.dtype)
    mlm_target_weights = mlm_target_weights * mlm_target_positions_within_len
    # Zero-out positions for pad MLM targets
    mlm_target_positions = mlm_target_positions * mlm_target_weights

  # Cut texts that are longer than `new_length`
  text_within_new_length = tf.less(positions, new_length)
  text_ids = text_ids * tf.cast(text_within_new_length, text_ids.dtype)
  text_mask = text_mask * tf.cast(text_within_new_length, text_mask.dtype)
  positions = tf.minimum(positions, new_length - 1)

  # Prepare 2D index for tokens positions in the next text_ids and text_mask.
  # Note that we use flat 2D index and flat values
  # (e.g. `tf.reshape(text_ids, [-1])`) since `tf.scatter_nd` does not support
  # batch dimension.
  batch_positions = _batched_range(old_length, batch_size, 1, positions.dtype)
  batch_positions = tf.reshape(batch_positions, [-1])
  text_index_2d = _get_2d_index(batch_positions, tf.reshape(positions, [-1]))

  new_text_ids = tf.scatter_nd(text_index_2d, tf.reshape(text_ids, [-1]),
                               new_shape)
  new_text_mask = tf.scatter_nd(text_index_2d, tf.reshape(text_mask, [-1]),
                                new_shape)

  # Insert entity start / end tokens into the new text_ids and text_mask.
  new_mention_start_positions_2d = get_2d_index(new_mention_start_positions)
  new_mention_end_positions_2d = get_2d_index(new_mention_end_positions)

  new_text_ids = tf.tensor_scatter_nd_add(
      new_text_ids, new_mention_start_positions_2d,
      new_mention_mask * entity_start_token_id)
  new_text_ids = tf.tensor_scatter_nd_add(
      new_text_ids, new_mention_end_positions_2d,
      new_mention_mask * entity_end_token_id)

  new_mention_mask = tf.cast(new_mention_mask, dtype=text_mask.dtype)
  new_text_mask = tf.tensor_scatter_nd_add(new_text_mask,
                                           new_mention_start_positions_2d,
                                           new_mention_mask)
  new_text_mask = tf.tensor_scatter_nd_add(new_text_mask,
                                           new_mention_end_positions_2d,
                                           new_mention_mask)

  features = {
      'text_ids': new_text_ids,
      'text_mask': new_text_mask,
      'mention_start_positions': new_mention_start_positions,
      'mention_end_positions': new_mention_end_positions,
      'mention_mask': new_mention_mask,
  }

  if mlm_target_positions is not None:
    features['mlm_target_weights'] = mlm_target_weights
    features['mlm_target_positions'] = mlm_target_positions

  return features


def text_hash(text: np.ndarray) -> np.ndarray:
  """Given 1D integer array with token IDs produces integer hash."""
  return np.polyval(text, 100003)


def text_hash_tf(text: tf.Tensor, seq_length: int) -> tf.Tensor:
  """Given 1D integer array with token IDs produces integer hash."""
  return tf.squeeze(
      tf.math.polyval(
          tf.split(text, num_or_size_splits=seq_length),
          tf.constant(100003, dtype=text.dtype)), 0)


def modified_cantor_pairing(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """Given integer tensors a and b, produces tensor of hashes."""
  # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
  # The function produces a modification of the original function making it
  # work consistently under modular 2^32 and 2^64. Namely, we remove // 2 from
  # (a + b) * (a + b + 1) // 2 + b
  b = tf.cast(b, dtype=a.dtype)
  return (a + b) * (a + b + 1) + b

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
"""Contains losses for mention encodings (e.g., coref resolution and MTB)."""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from language.mentionmemory.utils import mention_utils
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array

_SMALL_NUMBER = 1e-5


def build_coref_positive_negative_mask(mention_batch_positions: Array,
                                       all_mention_batch_positions: Array,
                                       mention_target_ids: Array,
                                       all_mention_target_ids: Array):
  """Decides which samples are positive / negative for coref resolution loss.

  positive_mask[i, j] is True <=> mention j in `all_mention_target_ids`
  could be used as a positive sample for mention i in `mention_target_ids`
  for the coreference resolution loss. It requires the following conditions:
  (1) mention_target_ids[i] == all_mention_target_ids[j]
  (2) mention i and mention j are not from the same passage
  (3) mention_target_ids[i] != 0 <=> is not padding
  (4) all_mention_target_ids[j] != 0 <=> is not padding

  negative_mask[i, j] is True <=> mention j in `all_mention_target_ids`
  could be used as a negative sample for mention i in `mention_target_ids`
  for the coreference resolution loss. It requires the following conditions:
  (1) mention_target_ids[i] != all_mention_target_ids[j]
  (2) mention i and mention j are not from the same passage
  (3) mention_target_ids[i] != 0 <=> is not padding
  (4) all_mention_target_ids[j] != 0 <=> is not padding

  Args:
    mention_batch_positions: position of a mention in its batch.
    all_mention_batch_positions: position of a global mention in its batch.
    mention_target_ids: shape [n_mentions]. IDs of mentions on the current
      device.
    all_mention_target_ids: shape [n_mentions]. IDs of all mentions on all
      devices.

  Returns:
    Positive and negative masks.
  """
  same_passage_mask = mention_utils.all_compare(mention_batch_positions,
                                                all_mention_batch_positions)
  not_same_passage_mask = jnp.logical_not(same_passage_mask)
  same_entity_mask = mention_utils.all_compare(mention_target_ids,
                                               all_mention_target_ids)
  not_same_entity_mask = jnp.logical_not(same_entity_mask)
  # Mask rows where base mention is PAD. In other words, none of the mentions
  # are positive or negative examples for PAD mentions.
  not_pad_mask = jnp.expand_dims(mention_target_ids, 1) > 0
  # Mask columns where target mention is PAD. In other words, we don't consider
  # PAD mentions to be either positive nor negative example.
  not_pad_all_mask = jnp.expand_dims(all_mention_target_ids, 0) > 0

  positive_mask = jnp.logical_and(same_entity_mask, not_same_passage_mask)
  positive_mask = jnp.logical_and(positive_mask, not_pad_mask)
  positive_mask = jnp.logical_and(positive_mask, not_pad_all_mask)

  negative_mask = jnp.logical_and(not_same_entity_mask, not_same_passage_mask)
  negative_mask = jnp.logical_and(negative_mask, not_pad_mask)
  negative_mask = jnp.logical_and(negative_mask, not_pad_all_mask)
  return positive_mask, negative_mask


def build_mtb_positive_negative_mask(mention_batch_positions: Array,
                                     all_mention_batch_positions: Array,
                                     mention_target_ids: Array,
                                     all_mention_target_ids: Array,
                                     mtb_positive_mask: Array):
  """Decides which samples are positive / negative for MTB loss.

  On of the argument `mtb_positive_mask` contains information regarding how
  many unique common entities does different passages in the batch contain.
  mtb_positive_mask[i, j] is True <=> a passage in the local batch that contains
  mention i has enough (typically, at least 2) unique common entities with
  a passage in the global batch that corresponds to mention j.

  positive_mask[i, j] is True <=> mention j in `all_mention_target_ids`
  could be used as a positive sample for mention i in `mention_target_ids`
  for the MTB loss. It requires the following conditions:
  (1) mention_target_ids[i] == all_mention_target_ids[j]
  (2) mention i and mention j are not from the same passage
  (3) mention_target_ids[i] != 0 <=> is not padding
  (4) all_mention_target_ids[j] != 0 <=> is not padding
  (5) mtb_positive_mask[i, j] is True

  hard_negative_mask[i, j] is True <=> mention j in `all_mention_target_ids`
  could be used as a hard negative sample for mention i in `mention_target_ids`
  for the MTB loss. It requires the following conditions:
  (1) mention_target_ids[i] == all_mention_target_ids[j]
  (2) mention i and mention j are not from the same passage
  (3) mention_target_ids[i] != 0 <=> is not padding
  (4) all_mention_target_ids[j] != 0 <=> is not padding
  (5) mtb_positive_mask[i, j] is False

  negative_mask[i, j] is True <=> mention j in `all_mention_target_ids`
  could be used as a negative sample for mention i in `mention_target_ids`
  for the MTB loss. It requires the following conditions:
  (1) mention_target_ids[i] != all_mention_target_ids[j]
  (2) mention i and mention j are not from the same passage
  (3) mention_target_ids[i] != 0 <=> is not padding
  (4) all_mention_target_ids[j] != 0 <=> is not padding
  OR hard_negative_mask[i, j] is True

  Args:
    mention_batch_positions: position of a mention in its batch.
    all_mention_batch_positions: position of a global mention in its batch.
    mention_target_ids: shape [n_mentions]. IDs of mentions on the current
      device.
    all_mention_target_ids: shape [n_mentions]. IDs of all mentions on all
      devices.
    mtb_positive_mask: shape [n_mentions, n_devices * n_mentions]. Whether
      mentions are considered as positive in MTB sense -- their passage share at
      least 2 unique common entities.

  Returns:
    Positive_mask, hard_negative_mask and negative_mask.
  """
  same_passage = mention_utils.all_compare(mention_batch_positions,
                                           all_mention_batch_positions)
  not_same_passage_mask = jnp.logical_not(same_passage)
  same_entity_mask = mention_utils.all_compare(mention_target_ids,
                                               all_mention_target_ids)
  not_same_entity_mask = jnp.logical_not(same_entity_mask)
  not_pad = jnp.expand_dims(mention_target_ids, 1) > 0
  not_pad_all = jnp.expand_dims(all_mention_target_ids, 0) > 0

  positive_mask = jnp.logical_and(same_entity_mask, not_same_passage_mask)
  positive_mask = jnp.logical_and(positive_mask, not_pad)
  positive_mask = jnp.logical_and(positive_mask, not_pad_all)

  hard_negative_mask = jnp.logical_and(positive_mask,
                                       jnp.logical_not(mtb_positive_mask))
  positive_mask = jnp.logical_and(positive_mask, mtb_positive_mask)

  negative_mask = jnp.logical_and(not_same_entity_mask, not_same_passage_mask)
  negative_mask = jnp.logical_and(negative_mask, not_pad)
  negative_mask = jnp.logical_and(negative_mask, not_pad_all)
  negative_mask = jnp.logical_or(negative_mask, hard_negative_mask)
  return positive_mask, hard_negative_mask, negative_mask


def get_all_encodings_and_target_ids(
    mention_encodings: Array, mention_target_ids: Array) -> Tuple[Array, Array]:
  """Gather encodings and IDs from all devices."""
  hidden_dim = mention_encodings.shape[-1]
  device_id = mention_utils.get_device_id('batch')
  if device_id is not None:
    all_mention_encodings = jax.lax.all_gather(mention_encodings, 'batch')
    all_mention_encodings = all_mention_encodings.reshape([-1, hidden_dim])

    all_mention_target_ids = jax.lax.all_gather(mention_target_ids, 'batch')
    all_mention_target_ids = all_mention_target_ids.reshape([-1])
  else:
    all_mention_encodings = mention_encodings
    all_mention_target_ids = mention_target_ids

  return all_mention_encodings, all_mention_target_ids


def coreference_resolution_loss(
    mention_encodings: Array,
    mention_batch_positions: Array,
    mention_target_ids: Array,
    batch_size: int,
    mode: str,
    mention_target_is_masked: Optional[Array] = None,
    metrics_prefix: str = '') -> Tuple[float, Dict[str, Dict[str, float]]]:
  """Compute coreference resolution loss over global batch (across all devices).

  Args:
    mention_encodings: [n_mentions, hidden_dim] mention encodings to be used for
      computing the loss.
    mention_batch_positions: [n_mentions] position of a mention in its batch.
    mention_target_ids: [n_mentions] IDs of mentions.
    batch_size: batch size per device.
    mode: how to compute the scores -- using dot product ('dot'), dot product
      divided by the sqrt root of the hidden dim ('dot_sqrt') or cosine
      similarity ('cos').
    mention_target_is_masked: [n_mentions] whether a mention has been masked. If
      it is provided the function computes additional metrics (accuracy) for
      masked and non masked mentions separately.
    metrics_prefix: optional prefix for all metric names.

  Returns:
    Loss and a dictionary of metrics dictionaries.
  """
  hidden_dim = mention_encodings.shape[-1]
  all_mention_encodings, all_mention_target_ids = get_all_encodings_and_target_ids(
      mention_encodings, mention_target_ids)

  (local_mention_batch_positions, global_mention_batch_positions
  ) = mention_utils.get_globally_consistent_batch_positions(
      mention_batch_positions, batch_size)

  positive_mask, negative_mask = build_coref_positive_negative_mask(
      local_mention_batch_positions, global_mention_batch_positions,
      mention_target_ids, all_mention_target_ids)

  # We compute a similarity between encodings of mentions from the
  # current device VS encodings of mentions gathered from all devices.
  unnormalized_scores = jnp.einsum('ld,gd->lg', mention_encodings,
                                   all_mention_encodings)

  # Set score to 0 if one of the mentions is PAD. This doesn't affect the loss
  # since we pass weights and positives / negatives masks explicitly.
  # However, it helps to compute various statistics like average scores
  # while ignoring scores for PAD mentions.
  mention_is_not_pad = jnp.expand_dims(mention_target_ids > 0, 1)
  mention_is_not_pad = mention_is_not_pad.astype(unnormalized_scores.dtype)
  all_mention_is_not_pad = jnp.expand_dims(all_mention_target_ids > 0, 0)
  all_mention_is_not_pad = all_mention_is_not_pad.astype(
      unnormalized_scores.dtype)
  unnormalized_scores = unnormalized_scores * mention_is_not_pad
  unnormalized_scores = unnormalized_scores * all_mention_is_not_pad

  # Norms of encodings of mentions from the current device.
  mention_encodings_norm = jnp.linalg.norm(mention_encodings, axis=-1)

  scores = unnormalized_scores
  if mode == 'dot':
    pass
  elif mode == 'dot_sqrt':
    scores /= jnp.sqrt(hidden_dim)
  elif mode == 'cos':
    # The cosine similarity is computed as dot product divided by norms of
    # both vectors.
    # Norms of encodings of mentions from all device.
    all_mention_encodings_norm = jnp.linalg.norm(all_mention_encodings, axis=-1)
    scores /= (_SMALL_NUMBER + jnp.expand_dims(mention_encodings_norm, 1))
    scores /= (_SMALL_NUMBER + jnp.expand_dims(all_mention_encodings_norm, 0))
  else:
    raise ValueError('Unknown coreference resolution mode: ' + mode)

  (loss, metrics, (acc_per_sample, weights_per_sample)
  ) = metric_utils.compute_cross_entropy_loss_with_positives_and_negatives_masks(
      scores, positive_mask, negative_mask, mention_target_ids != 0)

  num_positives = positive_mask.astype(jnp.float32).sum(1)
  num_positives = jnp.dot(num_positives, weights_per_sample)

  num_negatives = negative_mask.astype(jnp.float32).sum(1)
  num_negatives = jnp.dot(num_negatives, weights_per_sample)

  avg_score = scores.sum(axis=-1) / (
      all_mention_is_not_pad.sum() + _SMALL_NUMBER)
  avg_unnorm_score = unnormalized_scores.sum(axis=-1) / (
      all_mention_is_not_pad.sum() + _SMALL_NUMBER)

  metrics.update({
      'n_positive': num_positives,
      'n_negative': num_negatives,
      'avg_norm': jnp.dot(mention_encodings_norm, weights_per_sample),
      'avg_score': jnp.dot(avg_score, weights_per_sample),
      'avg_unnorm_score': jnp.dot(avg_unnorm_score, weights_per_sample),
  })

  final_metrics = {metrics_prefix + 'coref_resolution': metrics}
  if mention_target_is_masked is not None:
    final_metrics[metrics_prefix + 'coref_resolution_masked'] = {
        'acc':
            jnp.dot(acc_per_sample,
                    weights_per_sample * mention_target_is_masked),
        'denominator':
            jnp.dot(weights_per_sample, mention_target_is_masked),
    }
    final_metrics[metrics_prefix + 'coref_resolution_non_masked'] = {
        'acc':
            jnp.dot(acc_per_sample,
                    weights_per_sample * (1 - mention_target_is_masked)),
        'denominator':
            jnp.dot(weights_per_sample, (1 - mention_target_is_masked)),
    }

  return loss, final_metrics


def mtb_loss(
    mention_encodings: Array,
    mention_batch_positions: Array,
    mention_target_ids: Array,
    batch_size: int,
    mode: str,
    mention_target_is_masked: Optional[Array] = None,
    metrics_prefix: str = '') -> Tuple[float, Dict[str, Dict[str, float]]]:
  """Compute MTB loss over global batch (over all devices).

  Args:
    mention_encodings: [n_mentions, hidden_dim] mention encodings to be used for
      computing the loss.
    mention_batch_positions: [n_mentions] position of a mention in its batch.
    mention_target_ids: [n_mentions] IDs of mentions.
    batch_size: batch size per device.
    mode: how to compute the scores -- using dot product ('dot') or cosine
      similarity ('cos').
    mention_target_is_masked: [n_mentions] whether a mention has been masked. If
      it is provided the function computes additional metrics (accuracy) for
      masked and non masked mentions separately.
    metrics_prefix: optional prefix for all metric names.

  Returns:
    Loss and a dictionary of metrics dictionaries.
  """
  hidden_dim = mention_encodings.shape[-1]
  all_mention_encodings, all_mention_target_ids = get_all_encodings_and_target_ids(
      mention_encodings, mention_target_ids)

  (local_mention_batch_positions, global_mention_batch_positions
  ) = mention_utils.get_globally_consistent_batch_positions(
      mention_batch_positions, batch_size)

  _, num_common_items = mention_utils.get_num_common_unique_items(
      mention_batch_positions, mention_target_ids, batch_size)

  (positive_mask, hard_negative_mask,
   negative_mask) = build_mtb_positive_negative_mask(
       local_mention_batch_positions, global_mention_batch_positions,
       mention_target_ids, all_mention_target_ids, num_common_items >= 2)

  # We compute a similarity between encodings of mentions from the
  # current device VS encodings of mentions gathered from all devices.
  unnormalized_scores = jnp.einsum('ld,gd->lg', mention_encodings,
                                   all_mention_encodings)

  # Set score to 0 if one of the mentions is PAD. This doesn't affect the loss
  # since we pass weights and positives / negatives masks explicittly.
  # However, it helps to compute various statistics like average scores
  # while ignoring scores for PAD mentions.
  mention_is_not_pad = jnp.expand_dims(mention_target_ids > 0, 1)
  mention_is_not_pad = mention_is_not_pad.astype(unnormalized_scores.dtype)
  all_mention_is_not_pad = jnp.expand_dims(all_mention_target_ids > 0, 0)
  all_mention_is_not_pad = all_mention_is_not_pad.astype(
      unnormalized_scores.dtype)
  unnormalized_scores = unnormalized_scores * mention_is_not_pad
  unnormalized_scores = unnormalized_scores * all_mention_is_not_pad

  # Norms of encodings of mentions from the current device.
  mention_encodings_norm = jnp.linalg.norm(mention_encodings, axis=-1)

  scores = unnormalized_scores
  if mode == 'dot':
    pass
  elif mode == 'dot_sqrt':
    scores /= jnp.sqrt(hidden_dim)
  elif mode == 'cos':
    # The cosine similarity is computed as dot product divided by norms of
    # both vectors.
    # Norms of encodings of mentions from all device.
    all_mention_encodings_norm = jnp.linalg.norm(all_mention_encodings, axis=-1)
    scores /= (_SMALL_NUMBER + jnp.expand_dims(mention_encodings_norm, 1))
    scores /= (_SMALL_NUMBER + jnp.expand_dims(all_mention_encodings_norm, 0))
  else:
    raise ValueError('Unknown MTB mode: ' + mode)

  num_hard_negatives = hard_negative_mask.astype(jnp.int32).sum(axis=-1)
  weights = jnp.logical_and(mention_target_ids != 0, num_hard_negatives > 0)

  (loss, metrics, (acc_per_sample, weights_per_sample)
  ) = metric_utils.compute_cross_entropy_loss_with_positives_and_negatives_masks(
      scores, positive_mask, negative_mask, weights)

  num_positives = positive_mask.astype(jnp.float32).sum(1)
  num_positives = jnp.dot(num_positives, weights_per_sample)

  num_negatives = negative_mask.astype(jnp.float32).sum(1)
  num_negatives = jnp.dot(num_negatives, weights_per_sample)

  num_hard_negatives = num_hard_negatives.astype(jnp.float32)
  num_hard_negatives = jnp.dot(num_hard_negatives, weights_per_sample)

  avg_score = scores.sum(axis=-1) / (
      all_mention_is_not_pad.sum() + _SMALL_NUMBER)
  avg_unnorm_score = unnormalized_scores.sum(axis=-1) / (
      all_mention_is_not_pad.sum() + _SMALL_NUMBER)

  metrics.update({
      'n_positive': num_positives,
      'n_negative': num_negatives,
      'n_hard_negative': num_hard_negatives,
      'avg_norm': jnp.dot(mention_encodings_norm, weights_per_sample),
      'avg_score': jnp.dot(avg_score, weights_per_sample),
      'avg_unnorm_score': jnp.dot(avg_unnorm_score, weights_per_sample),
  })

  final_metrics = {metrics_prefix + 'mtb': metrics}
  if mention_target_is_masked is not None:
    final_metrics[metrics_prefix + 'mtb_masked'] = {
        'acc':
            jnp.dot(acc_per_sample,
                    weights_per_sample * mention_target_is_masked),
        'denominator':
            jnp.dot(weights_per_sample, mention_target_is_masked),
    }
    final_metrics[metrics_prefix + 'mtb_non_masked'] = {
        'acc':
            jnp.dot(acc_per_sample,
                    weights_per_sample * (1 - mention_target_is_masked)),
        'denominator':
            jnp.dot(weights_per_sample, (1 - mention_target_is_masked)),
    }
  return loss, final_metrics


def entity_linking_loss(mention_encodings: Array, entity_embeddings: Array,
                        mention_target_ids: Array,
                        mention_target_weights: Array, mode: str) -> Array:
  """Compute entity linking loss.

  Args:
    mention_encodings: [n_mentions, hidden_size] mention encodings to be used
      for computing the loss.
    entity_embeddings: [n_entities, hidden_size] entity embeddings table.
    mention_target_ids: [n_mentions] IDs of mentions.
    mention_target_weights: [n_mentions] per-mention weight for computing loss
      and metrics.
    mode: how to compute the scores -- using dot product ('dot'), dot product
      divided by the sqrt root of the hidden dim ('dot_sqrt') or cosine
      similarity ('cos').

  Returns:
    Loss, a dictionary with metrics values, per sample infomation
    (a tuple of accuracy per mention and weight per mention).
  """
  scores = jnp.einsum('qd,ed->qe', mention_encodings, entity_embeddings)
  scores = scores.astype(jnp.float32)

  mention_encodings_norm = jnp.linalg.norm(mention_encodings, axis=-1)
  entity_embeddings_norm = jnp.linalg.norm(entity_embeddings, axis=-1)

  # The cosine similarity is computed as dot product divided by norms of
  # both vectors.
  cos_scores = scores
  cos_scores /= (_SMALL_NUMBER + jnp.expand_dims(mention_encodings_norm, 1))
  cos_scores /= (_SMALL_NUMBER + jnp.expand_dims(entity_embeddings_norm, 0))

  if mode == 'dot':
    pass
  elif mode == 'dot_sqrt':
    hidden_dim = mention_encodings.shape[1]
    scores /= jnp.sqrt(hidden_dim)
  elif mode == 'cos':
    scores = cos_scores
  else:
    raise ValueError('Unknown entity linking mode: ' + mode)

  mention_target_weights = mention_target_weights.astype(jnp.float32)

  loss, _ = metric_utils.compute_weighted_cross_entropy(
      scores, mention_target_ids, mention_target_weights, inputs_are_prob=False)

  acc_per_mention = jnp.equal(jnp.argmax(scores, axis=-1), mention_target_ids)

  acc_per_mention = acc_per_mention * mention_target_weights

  n_mentions = mention_target_ids.shape[0]
  cos_per_mention = cos_scores[jnp.arange(n_mentions), mention_target_ids]
  cos_per_mention = cos_per_mention * mention_target_weights

  metrics = {
      'loss': loss,
      'acc': acc_per_mention.sum(),
      'cos_sim': cos_per_mention.sum(),
      'denominator': mention_target_weights.sum()
  }
  return loss, metrics, (acc_per_mention, mention_target_weights)  # pytype: disable=bad-return-type  # jax-ndarray


def get_batch_and_retrievals_entity_overlap(
    mention_target_batch_positions: Array,
    mention_target_ids: Array,
    mention_target_weights: Array,
    memory_text_entities: Array,
    batch_size: int,
):
  """Compute the overlap between entities in the batch and in retrievals.

  Args:
    mention_target_batch_positions: [n_target_mentions] position of a mention in
      its batch.
    mention_target_ids: [n_target_mentions] IDs of mentions.
    mention_target_weights: [n_target_mentions] per-mention weight for computing
      loss and metrics.
    memory_text_entities: [n_retrievals, n_memory_text_entities] IDs of mentions
      in the passage where memory is coming from. Note, entities in the same
      retrieval are assumed to be unique.
    batch_size: batch size.

  Returns:
    Array of shape [batch_size, n_retrievals] with the number of
      overlapping unique entity IDs in the batch and in the retrieval results.
  """
  n_target_mentions = mention_target_batch_positions.shape[0]
  n_retrievals = memory_text_entities.shape[0]
  n_memory_text_entities = memory_text_entities.shape[1]

  # Step 1: de-duplicate entities in the batch.
  # [n_target_mentions]
  mention_target_ids = mention_target_ids * mention_target_weights
  # [n_target_mentions]
  batch_ids = mention_utils.mask_duplicate_ids(mention_target_batch_positions,
                                               mention_target_ids)

  # Step 2: compare all entities in the batch against all entities in the
  # retrieval result.
  memory_text_entities = memory_text_entities.reshape(
      [n_retrievals * n_memory_text_entities])
  # [n_target_mentions, n_mentions * k_top * n_memory_text_entities]
  mention_id_in_retrieved_passages = mention_utils.all_compare_without_pad(
      batch_ids, memory_text_entities)
  mention_id_in_retrieved_passages = mention_id_in_retrieved_passages.astype(
      jnp.int32)

  # Step 3: sum up the comparison results by retrieval ID
  # [n_target_mentions, n_retrievals]
  mention_id_in_retrieved_passages = mention_id_in_retrieved_passages.reshape(
      [n_target_mentions, n_retrievals, n_memory_text_entities]).sum(-1)

  # Step 4: sum up the comparison results by batch position
  # [batch_size, n_retrievals]
  num_common_ids_between_samples = mention_utils.sum_by_batch_position(
      mention_target_batch_positions, mention_id_in_retrieved_passages,
      batch_size)

  return num_common_ids_between_samples


def same_entity_set_retrieval_loss(
    mention_target_batch_positions: Array,
    mention_target_ids: Array,
    mention_target_weights: Array,
    mention_batch_positions: Array,
    mention_mask: Array,
    memory_text_entities: Array,
    memory_attention_weights: Array,
    memory_mask: Array,
    batch_size: int,
    same_entity_set_target_threshold: int,
):
  """Computes same-entity-set-retrieval loss.

  We want to maximize attention scores received by memories which passages have
  large enough entity overlap (at least `same_entity_set_target_threshold`
  unique entities in common) with query mention's passage.

  User specifies which entity IDs exist in the current batch via the arguments:
  `mention_target_batch_positions`, `mention_target_ids` and
  `mention_target_weights`. Note that these correspond to linked mentions. While
  we retrieve memories for all mentions, we don't know entity IDs for non-linked
  mentions and they are not needed for computing entity overlap.

  Formally, let E(j) be the set of entity IDs in the j-th sample in the batch.
  k-th retrieval for i-th mention is "correct" if and only if
  IntersectionSize(E(mention_batch_positions[i]), memory_text_entities[i, j]) is
  at least `same_entity_set_target_threshold`.

  The loss is first computed for every mention separately and is negative log of
  sum of memory_attention_weights[i, j] where j is a "correct" retrieval for the
  i-th mention. The final loss is average of losses for those mentions which
  have at least one "correct" retrieval.

  Args:
    mention_target_batch_positions: [n_target_mentions] position of a linked
      (target) mention in its batch.
    mention_target_ids: [n_target_mentions] IDs of mentions.
    mention_target_weights: [n_target_mentions] per-mention weight for linked
      (target) mentions indicating whether a mention is padding or not.
    mention_batch_positions: [n_mentions] position of a mention in its batch.
    mention_mask: [n_mentions] whether a mention is padding or not.
    memory_text_entities: [n_mentions, k_top, n_memory_text_entities] IDs of
      mentions in the passage where memory is coming from.
    memory_attention_weights: [n_mentions, k_top] attention weights for the
      retrieval results.
    memory_mask: [n_mentions, k_top] which retrievals to use and which are to
      ignore in the loss computations. Typical usage is to ignore "disallowed"
      or same-passage retrievals.
    batch_size: batch size.
    same_entity_set_target_threshold: how many common entities needs to be
      between memory's passage and mention's passage in order to treat retrieval
      result as positive. If it's equal to 2 this loss becomes
      `same-mtb-retrieval` loss.

  Returns:
    Tuple of scalar loss, avg correct probability and normalizing factor.
  """
  n_mentions = memory_text_entities.shape[0]
  k_top = memory_text_entities.shape[1]

  # [batch_size, n_mentions * k_top]
  num_common_ids_between_samples = get_batch_and_retrievals_entity_overlap(
      mention_target_batch_positions=mention_target_batch_positions,
      mention_target_ids=mention_target_ids,
      mention_target_weights=mention_target_weights,
      memory_text_entities=memory_text_entities.reshape(
          [n_mentions * k_top, -1]),
      batch_size=batch_size,
  )

  # [batch_size, n_mentions, k_top]
  num_common_ids_between_samples = num_common_ids_between_samples.reshape(
      [batch_size, n_mentions, k_top])

  # [batch_size, n_mentions]
  position2item = (
      jnp.expand_dims(jnp.arange(batch_size), 1) == mention_batch_positions)
  position2item = position2item.astype(jnp.int32)

  # [n_mentions,  k_top]
  num_common_ids_between_mentions = jnp.einsum('bm,bmk->mk', position2item,
                                               num_common_ids_between_samples)

  # compute which retrievals have enough common elements with the
  # passage in the batch (at least `same_entity_set_target_threshold`) and
  # therefore, should be marked as "correct" retrievals.
  # [n_mentions, k_top]
  enough_common_elements = (
      num_common_ids_between_mentions >= same_entity_set_target_threshold)
  enough_common_elements = enough_common_elements.astype(jnp.int32)
  correct_retrievals_mask = enough_common_elements * memory_mask
  incorrect_retrievals_mask = (1 - enough_common_elements) * memory_mask

  # [n_mentions]
  correct_retrieval_exists = correct_retrievals_mask.sum(-1) > 0
  incorrect_retrieval_exists = incorrect_retrievals_mask.sum(-1) > 0
  loss_mask = jnp.logical_and(correct_retrieval_exists,
                              incorrect_retrieval_exists)
  loss_mask = loss_mask.astype(mention_mask.dtype) * mention_mask
  loss_mask = loss_mask.astype(jnp.float32)

  # [n_mentions, k_top]
  correct_retrievals_mask = correct_retrievals_mask.astype(jnp.float32)

  # compute loss and metrics
  # [n_mentions, k_top]
  correct_probs = jnp.einsum('mk,mk->m', correct_retrievals_mask,
                             memory_attention_weights)
  avg_probs = correct_probs * loss_mask
  loss = -jnp.log(correct_probs + _SMALL_NUMBER)
  loss = loss * loss_mask

  return loss.sum(), avg_probs.sum(), loss_mask.sum(),

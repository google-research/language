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
"""Contains mention auto-encoder task."""

from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import mauto_encoder
from language.mentionmemory.modules import mention_losses
from language.mentionmemory.modules import mlm_layer
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.tasks import mention_memory_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils import mention_preprocess_utils
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class MautoModel(nn.Module):
  """Mention auto-encoder pre-training model.

  Attributes:
    encoder_config: Mention Memory encoder hyperparameters.
  """
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = mauto_encoder.MautoEncoder(**self.encoder_config)
    self.mlm_layer = mlm_layer.MLMLayer(
        vocab_size=self.encoder.vocab_size,
        hidden_size=self.encoder.hidden_size,
        dtype=self.encoder.dtype,
        layer_norm_epsilon=self.encoder.layer_norm_epsilon,
        embedding_init=self.encoder.kernel_init,
        bias_init=self.encoder.bias_init,
    )

  def __call__(
      self, batch: Dict[str, Array],
      deterministic: bool) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    encoded_input, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    loss_helpers['mlm_logits'] = self.mlm_layer(
        encoded_input=encoded_input,
        mlm_target_positions=batch['mlm_target_positions'],
        shared_embedding=loss_helpers['word_embeddings'])

    return loss_helpers, logging_helpers


@task_registry.register_task('mauto')
class MautoTask(mention_encoder_task.MentionEncoderTask):
  """Task for pre-training diagnostic mention autoencoder."""

  model_class = MautoModel
  encoder_name = 'mauto'

  @classmethod
  def make_loss_fn(
      cls, config: ml_collections.ConfigDict
  ) -> Callable[..., Tuple[float, MetricGroups, Dict[str, Any]]]:
    """Creates task loss function."""

    mlm_weight = config.mlm_weight
    coref_res_weight = config.get('coref_res_weight', 0)
    coref_res_mode = config.get('coref_res_mode', 'dot')

    def loss_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[str, Any],
        model_vars: Dict[str, Any],
        batch: Dict[str, Any],
        deterministic: bool,
        dropout_rng: Optional[Dict[str, Array]] = None,
    ) -> Tuple[float, MetricGroups, Dict[str, Any]]:
      """Model-specific loss function. See BaseTask."""

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, _ = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      mlm_logits = loss_helpers['mlm_logits']
      mlm_target_is_mention = batch['mlm_target_is_mention']
      mlm_target_is_not_mention = 1 - batch['mlm_target_is_mention']
      mention_target_is_masked = batch['mention_target_is_masked']

      mlm_loss, mlm_denom = metric_utils.compute_weighted_cross_entropy(
          mlm_logits, batch['mlm_target_ids'], batch['mlm_target_weights'])
      correct_mask = jnp.equal(
          jnp.argmax(mlm_logits, axis=-1),
          batch['mlm_target_ids']) * batch['mlm_target_weights']
      mlm_acc = correct_mask.sum()
      mlm_mention_acc = (correct_mask * mlm_target_is_mention).sum()
      mlm_mention_denom = (batch['mlm_target_weights'] *
                           mlm_target_is_mention).sum()
      mlm_non_mention_acc = (correct_mask * mlm_target_is_not_mention).sum()
      mlm_non_mention_denom = (batch['mlm_target_weights'] *
                               mlm_target_is_not_mention).sum()

      loss = mlm_weight * mlm_loss / mlm_denom

      metrics = {
          'mlm': {
              'loss': mlm_loss,
              'acc': mlm_acc,
              'denominator': mlm_denom,
          },
          'mlm_mention': {
              'acc': mlm_mention_acc,
              'denominator': mlm_mention_denom,
          },
          'mlm_non_mention': {
              'acc': mlm_non_mention_acc,
              'denominator': mlm_non_mention_denom,
          },
      }

      if coref_res_weight > 0:
        batch_size = batch['text_ids'].shape[0]
        mention_target_ids = batch['mention_target_ids']
        mention_target_ids = mention_target_ids * batch['mention_target_weights']

        (coref_res_loss,
         coref_res_metrics) = mention_losses.coreference_resolution_loss(
             loss_helpers['target_mention_encodings'],
             batch['mention_target_batch_positions'], mention_target_ids,
             batch_size, coref_res_mode, mention_target_is_masked)
        coref_res_denom = coref_res_metrics['coref_resolution']['denominator']
        loss += coref_res_weight * coref_res_loss / coref_res_denom
        metrics.update(coref_res_metrics)

      metrics['agg'] = {
          'loss': loss,
          'denominator': 1.0,
      }

      return loss, metrics, {}  # pytype: disable=bad-return-type  # jax-ndarray

    return loss_fn

  @staticmethod
  def make_preprocess_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    """Produces function to preprocess samples. See BaseTask."""
    max_length = config.model_config.encoder_config.max_length

    mention_preprocessing_fn = mention_encoder_task.MentionEncoderTask.make_preprocess_fn(config)  # pylint: disable=line-too-long

    def preprocess_fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Performs preprocessing for individual sample."""
      new_example = mention_preprocessing_fn(example)

      # Compute hash of text for text identifiers
      new_example['text_identifiers'] = mention_preprocess_utils.text_hash_tf(
          example['text_ids'], max_length)
      return new_example

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    """Produces function to preprocess batches.

    For a selected subset of mentions in the batch, we retrieve the
    corresponding mention from the mention memory and include it in the batch.
    These retrieved mentions are then incorporated into the Transformer model
    like retrieved mentions in the Mention Memory encoder would be.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    mm_collater_fn = mention_memory_task.MentionMemoryTask.make_collater_fn(config)  # pylint: disable=line-too-long
    if config.model_config.encoder_config.get('no_retrieval', False):
      return mm_collater_fn
    max_retrieval_indices = config.max_retrieval_indices

    memory_table = data_utils.load_sharded_array(
        pattern=config.memory_pattern, stride=config.memory_reduction, offset=0)
    memory_hash = data_utils.load_sharded_array(
        pattern=config.memory_hash_pattern,
        stride=config.memory_reduction,
        offset=0)

    logging.info('Sorting hash array')
    hash_sorted_idx = np.argsort(memory_hash)
    memory_hash_sorted = memory_hash[hash_sorted_idx]

    # Add maximal integer value, so that if hash is greater than largest hash in
    # memory, we just take the first vector. We set the weight of this to zero
    # later so it doesn't affect the results.
    memory_hash_sorted = np.concatenate(
        (memory_hash_sorted, [np.iinfo(np.int32).max])).astype(np.int32)

    hash_sorted_idx = np.concatenate((hash_sorted_idx, [0])).astype(np.int32)

    memory_table = tf.constant(memory_table)
    memory_hash_sorted = tf.constant(memory_hash_sorted)
    hash_sorted_idx = tf.constant(hash_sorted_idx)

    memory_entity_pattern = config.get('memory_entity_pattern', None)
    if memory_entity_pattern:
      memory_entity_ids = data_utils.load_sharded_array(
          pattern=config.memory_entity_pattern,
          stride=config.memory_reduction,
          offset=0)

    def collater_fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      batch = mm_collater_fn(batch)

      retrieve_masked = config.get('retrieve_masked', False)

      # Subselect mentions for which to retrieve corresponding memory.
      # We want to sample mentions which are linked, not masked, and not padded.
      scores = tf.random.uniform(tf.shape(
          batch['mention_target_is_masked'])) + 2 * tf.cast(
              batch['mention_target_weights'], tf.float32)

      if not retrieve_masked:
        scores -= tf.cast(batch['mention_target_is_masked'], tf.float32)

      _, mention_target_retrieval_indices = tf.math.top_k(
          scores, k=max_retrieval_indices)

      mention_retrieval_indices = tf.gather(batch['mention_target_indices'],
                                            mention_target_retrieval_indices)
      retrieval_mention_mask = tf.gather(batch['mention_target_weights'],
                                         mention_target_retrieval_indices)
      # set weight to 0 for masked retrievals if we do not want to include these
      if not retrieve_masked:
        retrieval_mention_mask *= tf.gather(
            1 - tf.cast(batch['mention_target_is_masked'], tf.int32),
            mention_target_retrieval_indices)

      retrieval_mention_start_positions = tf.gather(
          batch['mention_start_positions'], mention_retrieval_indices)
      retrieval_text_identifiers = tf.gather(batch['text_identifiers'],
                                             mention_retrieval_indices)
      retrieval_mention_hash = mention_preprocess_utils.modified_cantor_pairing(
          tf.cast(retrieval_mention_start_positions, tf.int64),
          retrieval_text_identifiers)
      retrieval_mention_hash = tf.cast(retrieval_mention_hash, tf.int32)

      retrieval_mention_sort_ids = tf.searchsorted(memory_hash_sorted,
                                                   retrieval_mention_hash)

      # Searchsorted does not check whether value is present in array, just
      # finds insertion point. Here we check and set to default retrieval if not
      # present.
      hash_not_present_mask = tf.not_equal(
          retrieval_mention_hash,
          tf.gather(memory_hash_sorted, retrieval_mention_sort_ids))
      hash_not_present = tf.where(hash_not_present_mask)
      update_values = tf.fill((tf.shape(hash_not_present)[0],),
                              tf.shape(hash_sorted_idx)[0] - 1)
      retrieval_mention_sort_ids = tf.tensor_scatter_nd_update(
          retrieval_mention_sort_ids, hash_not_present, update_values)

      # Set mask to 0 if no mention is found
      batch['retrieval_mention_mask'] = retrieval_mention_mask * (
          1 - tf.cast(hash_not_present_mask, tf.int32))

      retrieval_mention_ids = tf.gather(hash_sorted_idx,
                                        retrieval_mention_sort_ids)
      retrieval_mention_values = tf.gather(memory_table, retrieval_mention_ids)
      # Match passage entity_ids with memory entity ids as sanity check.
      if memory_entity_pattern:
        retrieval_memory_entity_ids = tf.gather(memory_entity_ids,
                                                retrieval_mention_ids)
        retrieval_passage_entity_ids = tf.gather(
            tf.cast(batch['mention_target_ids'], tf.int32),
            mention_target_retrieval_indices)
        entity_does_not_match = tf.not_equal(retrieval_memory_entity_ids,
                                             retrieval_passage_entity_ids)

        batch['entity_does_not_match'] = tf.logical_and(
            entity_does_not_match,
            tf.cast(batch['retrieval_mention_mask'], tf.bool))

      batch['retrieval_mention_values'] = retrieval_mention_values
      batch['retrieval_mention_scores'] = tf.ones_like(
          batch['retrieval_mention_mask'])
      batch['retrieval_mention_batch_positions'] = tf.gather(
          batch['mention_batch_positions'], mention_retrieval_indices)
      batch['retrieval_mention_start_positions'] = retrieval_mention_start_positions  # pylint: disable=line-too-long
      batch['retrieval_mention_end_positions'] = tf.gather(
          batch['mention_end_positions'], mention_retrieval_indices)
      batch['mention_retrieval_indices'] = mention_retrieval_indices

      return batch

    return collater_fn

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Produces model-specific dummy input batch. See BaseTask."""

    dummy_input = mention_memory_task.MentionMemoryTask.dummy_input(config)
    encoder_config = config.model_config.encoder_config
    float_type = encoder_config.dtype
    int_type = jnp.int32
    mention_retrieval_shape = (config.max_retrieval_indices)
    retrieval_mention_values = np.ones(
        (config.max_retrieval_indices, encoder_config.retrieval_dim))
    mauto_dummy_dict = {
        'retrieval_mention_values':
            jnp.asarray(retrieval_mention_values, float_type),
        'retrieval_mention_scores':
            jnp.ones(mention_retrieval_shape, float_type),
        'retrieval_mention_mask':
            jnp.ones(mention_retrieval_shape, int_type),
        'retrieval_mention_batch_positions':
            jnp.ones(mention_retrieval_shape, int_type),
        'retrieval_mention_start_positions':
            jnp.ones(mention_retrieval_shape, int_type),
        'retrieval_mention_end_positions':
            jnp.ones(mention_retrieval_shape, int_type),
    }
    dummy_input.update(mauto_dummy_dict)

    return dummy_input

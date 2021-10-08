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
"""Contains memory generation task implementation and utilities."""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import mention_encoder_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import data_utils
from language.mentionmemory.utils.custom_types import Array, Dtype  # pylint: disable=g-multiple-import
import language.mentionmemory.utils.mention_preprocess_utils as mention_preprocess_utils
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf

_SMALL = 1e-10


class MemoryGenerationModel(nn.Module):
  """Memory generation model.

  Attributes:
    encoder_name: encoder name in encoder registry.
    encoder_config: Mention Memory encoder hyperparameters.
  """
  dtype: Dtype
  encoder_name: str
  encoder_config: ml_collections.FrozenConfigDict
  memory_keys_feature: Optional[str] = None
  memory_values_feature: str = 'target_mention_encodings'

  def setup(self):
    self.encoder = encoder_registry.get_registered_encoder(
        self.encoder_name)(**self.encoder_config)

  def __call__(
      self, batch: Dict[str, Array],
      deterministic: bool) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    _, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)

    loss_helpers['memory_generation'] = {}
    loss_helpers['memory_generation']['values'] = loss_helpers[
        self.memory_values_feature]
    if self.memory_keys_feature is not None:
      loss_helpers['memory_generation']['keys'] = loss_helpers[
          self.memory_keys_feature]

    return loss_helpers, logging_helpers


@task_registry.register_task('memory_generation')
class MemoryGenerationTask(mention_encoder_task.MentionEncoderTask):
  """Task that generates memory from the corpus using an encoder."""

  model_class = MemoryGenerationModel

  @classmethod
  def make_prediction_fn(
      cls,
      config: ml_collections.ConfigDict) -> Callable[..., Dict[str, Array]]:
    """Creates task prediction function for inference."""

    def predict_fn(
        model_config: ml_collections.FrozenConfigDict,
        model_params: Dict[str, Any],
        model_vars: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Array]:
      """Model-specific prediction function.

      Args:
        model_config: contains model config hyperparameters.
        model_params: contains model parameters.
        model_vars: contains model variables (not optimized).
        batch: model input.

      Returns:
        Dict[str, Array]. predictions.
      """

      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)

      loss_helpers, _ = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=True, rngs=None)

      return loss_helpers['memory_generation']

    return predict_fn

  @staticmethod
  def make_preprocess_fn(
      config: ml_collections.ConfigDict
  ) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    """Produces function to preprocess samples.

    See BaseTask.

    Here we add a text identifier hash to the standard MentionEncoderTask
    preprocessing pipeline.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses samples to be usable for the model
      (mod casting from tf to jnp dtype).
    """
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

    See BaseTask.

    Batches text identifiers after standard mention preprocessing. Also masks
    out mentions that are too close to a passage boundary, and for which we may
    not have enough context to generate a meaningful encoding.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """
    mention_collater_fn = mention_encoder_task.MentionEncoderTask.make_collater_fn(config)  # pylint: disable=line-too-long
    min_distance_from_passage_boundary = config.min_distance_from_passage_boundary
    bsz = config.per_device_batch_size
    max_mentions_per_sample = config.max_mentions_per_sample

    def collater_fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      new_batch = mention_collater_fn(batch)
      # Only generate text identifiers and mention hashes for
      # the target (linked) mentions.
      new_batch['target_text_identifiers'] = tf.gather(
          new_batch['text_identifiers'],
          new_batch['mention_target_batch_positions'])
      new_batch[
          'target_mention_hashes'] = mention_preprocess_utils.modified_cantor_pairing(
              new_batch['mention_target_start_positions'],
              new_batch['target_text_identifiers'])

      seq_len = tf.shape(batch['text_ids'])[1]
      starts_far_from_passage_boundary = tf.greater_equal(
          new_batch['mention_target_start_positions'],
          min_distance_from_passage_boundary)
      ends_far_from_passage_boundary = tf.less(
          new_batch['mention_target_end_positions'],
          tf.cast(seq_len, new_batch['mention_target_end_positions'].dtype) -
          min_distance_from_passage_boundary)
      far_from_passage_boundary = tf.logical_and(
          starts_far_from_passage_boundary, ends_far_from_passage_boundary)
      far_from_passage_boundary = tf.cast(
          far_from_passage_boundary,
          dtype=new_batch['mention_target_weights'].dtype)
      new_batch['mention_target_weights'] = (
          new_batch['mention_target_weights'] * far_from_passage_boundary)

      # Collect unique mention IDs per sample in the batch
      unique_mention_ids = []
      # Mask-out not linked entities.
      dense_mention_ids = batch['dense_mention_ids'] * batch[
          'dense_mention_mask']
      for i in range(bsz):
        unique_mention_ids_per_i = tf.unique(dense_mention_ids[i]).y
        unique_mention_ids_per_i = tf.cast(unique_mention_ids_per_i, tf.int32)
        unique_mention_ids_per_i = mention_preprocess_utils.dynamic_padding_1d(
            unique_mention_ids_per_i, max_mentions_per_sample)
        unique_mention_ids.append(unique_mention_ids_per_i)
      new_batch['unique_mention_ids'] = tf.stack(unique_mention_ids)
      return new_batch

    return collater_fn

  @staticmethod
  def dummy_input(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Produces model-specific dummy input batch. See BaseTask."""

    dummy_input = mention_encoder_task.MentionEncoderTask.dummy_input(config)
    mention_position_shape = (
        config.max_mentions * config.per_device_batch_size)
    int_type = jnp.int32
    dummy_input['target_text_identifiers'] = jnp.ones(mention_position_shape,
                                                      int_type)
    dummy_input['target_mention_hashes'] = jnp.ones(mention_position_shape,
                                                    int_type)
    dummy_input['unique_mention_ids'] = jnp.ones(
        (config.per_device_batch_size, config.max_mentions_per_sample),
        int_type)
    return dummy_input

  @staticmethod
  def load_weights(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Load model weights."""
    encoder_name = config.model_config.encoder_name
    encoder_class = encoder_registry.get_registered_encoder(encoder_name)
    encoder_variables = encoder_class.load_weights(config)
    model_variables = {}
    for group_key in encoder_variables:
      model_variables[group_key] = {'encoder': encoder_variables[group_key]}

    return model_variables


class MemorySaver:
  """Class that collect memories into numpy arrays and saves to files."""

  def __init__(self, num_total_memories: int, memory_dim: int, max_length: int,
               max_mentions_per_sample: int, memory_key_dim: Optional[int]):
    self.num_total_memories = num_total_memories
    self.max_mentions_per_sample = max_mentions_per_sample
    self.memory_embeddings = np.zeros((self.num_total_memories, memory_dim),
                                      np.float32)
    if memory_key_dim is not None:
      self.memory_key_embeddings = np.zeros(
          (self.num_total_memories, memory_key_dim), np.float32)
    else:
      self.memory_key_embeddings = None
    self.memory_labels = np.zeros((self.num_total_memories), np.int32)
    self.memory_text_hashes = np.zeros((self.num_total_memories), np.int32)
    self.memory_mention_hashes = np.zeros((self.num_total_memories), np.int32)
    self.memory_permutation = np.random.permutation(self.num_total_memories)
    self.text_ids = np.zeros((self.num_total_memories, max_length), np.int32)
    self.start_end_positions = np.zeros((self.num_total_memories, 2), np.int32)
    self.text_entities = np.zeros(
        (self.num_total_memories, self.max_mentions_per_sample), np.int32)
    self.memory_index = 0

  def get_num_memories(self):
    return self.memory_index

  def add_memories(self, batch: Dict[str, Array], predictions: Dict[str,
                                                                    Array]):
    """Save generated memories in-memory storage."""
    mention_mask = batch['mention_target_weights'] > 0
    memory_index_end = min(self.num_total_memories,
                           self.memory_index + mention_mask.sum())
    memory_index_len = memory_index_end - self.memory_index
    indices = self.memory_permutation[self.memory_index:memory_index_end]

    # We might not save mention encodings for all target mentions.
    # First, some of them might are pad or are too close to a passage boundary
    # (in these cases we assume that `mention_target_weights` = 0).
    # Second, there might be more mentions then we actually need
    # since we limit the number of the total memories by `num_total_memories`.
    # Therefore, we create `mention_index` to select a subset of mentions,
    # which encodings we are planning to save.
    mention_index_0, mention_index_1 = jnp.nonzero(mention_mask)
    mention_index_0 = mention_index_0[:memory_index_len]
    mention_index_1 = mention_index_1[:memory_index_len]
    mention_index = (mention_index_0, mention_index_1)

    self.memory_embeddings[indices] = predictions['values'][mention_index]
    if self.memory_key_embeddings is not None:
      self.memory_key_embeddings[indices] = predictions['keys'][mention_index]
    self.memory_labels[indices] = batch['mention_target_ids'][mention_index]
    self.memory_text_hashes[indices] = batch['target_text_identifiers'][
        mention_index]
    self.memory_mention_hashes[indices] = batch['target_mention_hashes'][
        mention_index]

    # Convert to global batch positions
    n_devices, batch_size, _ = batch['text_ids'].shape
    mention_target_batch_positions = batch['mention_target_batch_positions']
    mention_target_batch_positions = (
        mention_target_batch_positions +
        np.expand_dims(np.arange(n_devices), 1) * batch_size)
    mention_target_batch_positions = mention_target_batch_positions[
        mention_index]

    self.text_entities[indices] = batch['unique_mention_ids'].reshape(
        n_devices * batch_size, -1)[mention_target_batch_positions]
    self.text_ids[indices] = batch['text_ids'].reshape(
        n_devices * batch_size, -1)[mention_target_batch_positions]
    self.start_end_positions[
        indices, 0] = batch['mention_target_start_positions'][mention_index]
    self.start_end_positions[
        indices, 1] = batch['mention_target_end_positions'][mention_index]
    self.memory_index = memory_index_end

  def save(self, output_dir: str, num_shards: int, stride: int, offset: int,
           shard_size_divisible: int):
    """Save generated memories into files."""

    def save_sharded_array(array: np.ndarray, filename: str):
      data_utils.save_sharded_array(array, os.path.join(output_dir,
                                                        filename), num_shards,
                                    stride, offset, shard_size_divisible)

    save_sharded_array(self.memory_embeddings, 'encodings')
    save_sharded_array(self.memory_labels, 'labels')
    save_sharded_array(self.memory_text_hashes, 'hashes')
    save_sharded_array(self.memory_mention_hashes, 'mention_hashes')
    save_sharded_array(self.text_ids, 'texts')
    save_sharded_array(self.start_end_positions, 'positions')
    save_sharded_array(self.text_entities, 'text_entities')
    if self.memory_key_embeddings is not None:
      save_sharded_array(self.memory_key_embeddings, 'key_encodings')

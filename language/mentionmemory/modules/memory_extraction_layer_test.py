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
"""Tests for memory extraction layer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import memory_extraction_layer
from language.mentionmemory.utils import test_utils
import numpy as np


class MemoryExtractionLayerTest(absltest.TestCase):
  """Entity linking layer tests."""

  memory_key_dim = 32
  memory_value_dim = 64
  hidden_size = 128
  bsz = 4
  seq_len = 20
  n_mentions = 5
  dtype = jnp.float32
  n_devices = 4

  def test_linking_layer(self):
    """Testing linking layer."""

    test_utils.force_multi_devices(self.n_devices)
    devices = jax.local_devices()

    encoded_input = jnp.ones(
        shape=(self.n_devices, self.bsz, self.seq_len, self.hidden_size),
        dtype=self.dtype)
    encoded_input = jax.device_put_sharded(list(encoded_input), devices)
    mention_batch_positions = np.random.randint(
        self.bsz, size=(self.n_devices, self.n_mentions))
    mention_batch_positions = jax.device_put_sharded(
        list(mention_batch_positions), devices)
    mention_start_positions = np.random.randint(
        self.seq_len - 1, size=(self.n_devices, self.n_mentions))
    mention_end_positions = mention_start_positions + 1
    mention_start_positions = jax.device_put_sharded(
        list(mention_start_positions), devices)
    mention_end_positions = jax.device_put_sharded(
        list(mention_end_positions), devices)
    mention_mask = jnp.ones(shape=(self.n_devices, self.n_mentions))
    mention_mask = jax.device_put_sharded(list(mention_mask), devices)
    mention_entity_ids = jnp.arange(self.n_devices * self.n_mentions).reshape(
        self.n_devices, self.n_mentions)
    mention_entity_ids = jax.device_put_sharded(
        list(mention_entity_ids), devices)

    model = memory_extraction_layer.MemoryExtractionLayer(
        memory_key_dim=self.memory_key_dim,
        memory_value_dim=self.memory_value_dim,
        dtype=self.dtype,
    )
    pinit_with_output = jax.pmap(model.init_with_output, axis_name='batch')

    rng = jax.random.PRNGKey(0)
    split_rng = jax.random.split(rng, self.n_devices)
    result_dict, _ = pinit_with_output(
        split_rng,
        encoding=encoded_input,
        mention_batch_positions=mention_batch_positions,
        mention_start_positions=mention_start_positions,
        mention_end_positions=mention_end_positions,
        mention_mask=mention_mask,
        mention_entity_ids=mention_entity_ids,
    )

    # Check shapes are as expected
    self.assertSequenceEqual(
        result_dict['memory_keys'].shape,
        (self.n_devices, self.n_devices * self.n_mentions, self.memory_key_dim))
    self.assertSequenceEqual(result_dict['memory_values'].shape,
                             (self.n_devices, self.n_devices * self.n_mentions,
                              self.memory_value_dim))
    self.assertSequenceEqual(result_dict['memory_mask'].shape,
                             (self.n_devices, self.n_devices * self.n_mentions))
    self.assertSequenceEqual(result_dict['memory_entity_ids'].shape,
                             (self.n_devices, self.n_devices * self.n_mentions))

    # Memory mask and entity ids should just have been all gathered
    self.assertTrue(
        jnp.all(result_dict['memory_mask'][0].reshape(
            self.n_devices, self.n_mentions) == mention_mask))
    self.assertTrue(
        jnp.all(result_dict['memory_entity_ids'][0].reshape(
            self.n_devices, self.n_mentions) == mention_entity_ids))


if __name__ == '__main__':
  absltest.main()

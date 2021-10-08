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
"""Tests for retrieval update layers."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import retrieval_update_layers
import numpy as np

_HIDDEN_SIZE = 32
_RETRIEVAL_DIM = 16
_DTYPE = jnp.float32
_LAYER_NORM_EPS = 1e-12
_NAMED_PARAMETERS = (
    {
        'retrieval_update_type': 'additive',
        'retrieval_update_config': {
            'input_dim': _HIDDEN_SIZE,
            'dtype': _DTYPE,
            'layer_norm_epsilon': _LAYER_NORM_EPS,
        },
    },
    {
        'retrieval_update_type': 'concat_mlp',
        'retrieval_update_config': {
            'input_dim': _HIDDEN_SIZE,
            'retrieval_dim': _RETRIEVAL_DIM,
            'hidden_dim': _HIDDEN_SIZE * 4,
            'n_additional_concat_layers': 1,
            'n_pooled_layers': 1,
            'dropout_rate': 0.1,
            'dtype': _DTYPE,
            'layer_norm_epsilon': _LAYER_NORM_EPS,
        },
    },
    {
        'retrieval_update_type': 'dummy',
        'retrieval_update_config': {
            'input_dim': _HIDDEN_SIZE,
            'dtype': _DTYPE,
            'layer_norm_epsilon': _LAYER_NORM_EPS,
        }
    },
)


class RetrievalUpdateLayerTest(parameterized.TestCase):

  bsz = 4
  seq_len = 20
  n_mentions = 8
  k_retrieval = 2

  def setUp(self):
    super().setUp()

    self.encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, _HIDDEN_SIZE), dtype=_DTYPE)
    self.retrieval_values = jnp.ones(
        shape=(self.n_mentions, self.k_retrieval, _RETRIEVAL_DIM), dtype=_DTYPE)
    self.retrieval_scores = jnp.ones(
        shape=(self.n_mentions, self.k_retrieval), dtype=_DTYPE)
    # input transformed by layer norm, neededed for comparison
    self.normed_input = jnp.zeros(
        shape=(self.bsz, self.seq_len, _HIDDEN_SIZE), dtype=_DTYPE)
    self.rng = jax.random.PRNGKey(0)
    self.mention_batch_positions = np.random.randint(
        self.bsz, size=(self.n_mentions))
    self.mention_start_positions = np.random.randint(
        self.seq_len - 1, size=(self.n_mentions))
    self.mention_end_positions = self.mention_start_positions + 1

  @parameterized.parameters(*_NAMED_PARAMETERS)
  def test_attention_layer(self, retrieval_update_type,
                           retrieval_update_config):
    """Testing entity attention layer."""

    update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        retrieval_update_type](**retrieval_update_config)

    mention_mask = jnp.ones(shape=(self.n_mentions))

    encoded_output, _ = update_layer.init_with_output(
        self.rng,
        encoded_input=self.encoded_input,
        retrieval_values=self.retrieval_values,
        retrieval_scores=self.retrieval_scores,
        mention_batch_positions=self.mention_batch_positions,
        mention_start_positions=self.mention_start_positions,
        mention_end_positions=self.mention_end_positions,
        mention_mask=mention_mask,
        deterministic=True,
    )

    # Check input was changed
    if retrieval_update_type == 'dummy':
      self.assertTrue(jnp.allclose(encoded_output, self.encoded_input))
    else:
      self.assertFalse(jnp.allclose(encoded_output, self.normed_input))

    # Check input was not changed where it should not be
    all_indices = set(
        itertools.product(jnp.arange(self.bsz), jnp.arange(self.seq_len)))
    start_indices = set(
        zip(self.mention_batch_positions, self.mention_start_positions))
    non_start_indices = all_indices.difference(start_indices)
    non_start_indices_1, non_start_indices_2 = zip(*non_start_indices)
    non_start_indices_1 = jnp.asarray(non_start_indices_1)
    non_start_indices_2 = jnp.asarray(non_start_indices_2)

    non_start_outputs = encoded_output[non_start_indices_1, non_start_indices_2]

    if retrieval_update_type == 'dummy':
      non_start_inputs = self.encoded_input[non_start_indices_1,
                                            non_start_indices_2]
    else:
      non_start_inputs = self.normed_input[non_start_indices_1,
                                           non_start_indices_2]

    self.assertTrue(jnp.allclose(non_start_outputs, non_start_inputs))

    self.assertSequenceEqual(encoded_output.shape,
                             (self.bsz, self.seq_len, _HIDDEN_SIZE))

  @parameterized.parameters(*_NAMED_PARAMETERS)
  def test_masking(self, retrieval_update_type, retrieval_update_config):
    """Check masked positions not contributing to input."""

    update_layer = retrieval_update_layers.RETRIEVAL_UPDATE_REGISTRY[
        retrieval_update_type](**retrieval_update_config)

    mention_mask = jnp.zeros(shape=(self.n_mentions))
    encoded_output, _ = update_layer.init_with_output(
        self.rng,
        encoded_input=self.encoded_input,
        retrieval_values=self.retrieval_values,
        retrieval_scores=self.retrieval_scores,
        mention_batch_positions=self.mention_batch_positions,
        mention_start_positions=self.mention_start_positions,
        mention_end_positions=self.mention_end_positions,
        mention_mask=mention_mask,
        deterministic=True,
    )

    if retrieval_update_type == 'dummy':
      self.assertTrue(jnp.allclose(encoded_output, self.encoded_input))
    else:
      self.assertTrue(jnp.allclose(encoded_output, self.normed_input))


if __name__ == '__main__':
  absltest.main()

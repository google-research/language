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
"""Tests for entity linking layer."""

import itertools

from absl.testing import absltest
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import entity_attention_layer
import numpy as np
import scipy.spatial


class EntityAttentionLayerTest(absltest.TestCase):
  """Entity attention layer tests."""
  entity_vocab_size = 1000
  hidden_size = 32
  entity_dim = 16
  dtype = jnp.float32

  bsz = 4
  seq_len = 20
  n_mentions = 10

  def setUp(self):
    super().setUp()
    self.model = entity_attention_layer.EntityAttentionLayer(
        entity_dim=self.entity_dim,
        hidden_size=self.hidden_size,
        dtype=self.dtype,
    )

    entity_embeddings = np.random.rand(self.entity_vocab_size, self.entity_dim)
    self.entity_embeddings = jnp.asarray(entity_embeddings)

    self.encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.hidden_size), dtype=self.dtype)
    # input transformed by layer norm, neededed for comparison
    self.normed_input = jnp.zeros(
        shape=(self.bsz, self.seq_len, self.hidden_size), dtype=self.dtype)
    self.rng = jax.random.PRNGKey(0)
    self.mention_batch_positions = np.random.randint(
        self.bsz, size=(self.n_mentions))
    self.mention_start_positions = np.random.randint(
        self.seq_len - 1, size=(self.n_mentions))
    self.mention_end_positions = self.mention_start_positions + 1

  def test_attention_layer(self):
    """Testing entity attention layer."""

    mention_mask = jnp.ones(shape=(self.n_mentions))

    output, params = self.model.init_with_output(
        self.rng,
        encoded_input=self.encoded_input,
        mention_batch_positions=self.mention_batch_positions,
        mention_start_positions=self.mention_start_positions,
        mention_end_positions=self.mention_end_positions,
        mention_mask=mention_mask,
        entity_embeddings=self.entity_embeddings,
    )
    params = params['params']

    # Check input was changed
    self.assertFalse(jnp.allclose(output['encoded_output'], self.normed_input))

    # Check input was not changed where it should not be
    all_indices = set(
        itertools.product(jnp.arange(self.bsz), jnp.arange(self.seq_len)))
    start_indices = set(
        zip(self.mention_batch_positions, self.mention_start_positions))
    non_start_indices = all_indices.difference(start_indices)
    non_start_indices_1, non_start_indices_2 = zip(*non_start_indices)
    non_start_indices_1 = jnp.asarray(non_start_indices_1)
    non_start_indices_2 = jnp.asarray(non_start_indices_2)

    non_start_outputs = output['encoded_output'][non_start_indices_1,
                                                 non_start_indices_2]
    non_start_inputs = self.normed_input[non_start_indices_1,
                                         non_start_indices_2]

    self.assertTrue(jnp.allclose(non_start_outputs, non_start_inputs))

    self.assertSequenceEqual(output['encoded_output'].shape,
                             (self.bsz, self.seq_len, self.hidden_size))

    for i in range(self.n_mentions):
      mention_start_encodings = self.encoded_input[
          self.mention_batch_positions[i], self.mention_start_positions[i]]
      mention_end_encodings = self.encoded_input[
          self.mention_batch_positions[i], self.mention_end_positions[i]]
      mention_encodings = jnp.concatenate(
          [mention_start_encodings, mention_end_encodings], axis=-1)
      mention_encodings = jnp.matmul(
          mention_encodings, params['mention_query_projector']['kernel'])
      mention_encodings = mention_encodings + params['mention_query_projector'][
          'bias']
      self.assertSequenceAlmostEqual(
          mention_encodings, output['mention_encodings'][i], places=4)

    self.assertSequenceEqual(output['cosine_similarity'].shape,
                             (self.n_mentions, self.entity_vocab_size))

    for i in range(self.n_mentions):
      for j in range(self.entity_vocab_size):
        self.assertAlmostEqual(
            output['cosine_similarity'][i, j],
            1 - scipy.spatial.distance.cosine(output['mention_encodings'][i],
                                              self.entity_embeddings[j]),
            places=2)

    self.assertSequenceEqual(output['attention_weights'].shape,
                             (self.n_mentions, self.entity_vocab_size))

  def test_masking(self):
    """Check masked positions not contributing to input."""

    mention_mask = jnp.zeros(shape=(self.n_mentions))
    output, _ = self.model.init_with_output(
        self.rng,
        encoded_input=self.encoded_input,
        mention_batch_positions=self.mention_batch_positions,
        mention_start_positions=self.mention_start_positions,
        mention_end_positions=self.mention_end_positions,
        mention_mask=mention_mask,
        entity_embeddings=self.entity_embeddings,
    )

    self.assertTrue(jnp.allclose(output['encoded_output'], self.normed_input))


if __name__ == '__main__':
  absltest.main()

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
"""Tests for embedding."""

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import embedding
import numpy as np


class EmbedTest(absltest.TestCase):
  """Test embedding layer."""

  num_embeddings = 24
  embedding_dim = 12
  bsz = 2
  seq_len = 4

  def test_embed(self):
    """Test embedding layer."""

    model = embedding.Embed(
        num_embeddings=self.num_embeddings,
        embedding_dim=self.embedding_dim,
        dtype=jnp.bfloat16,
    )

    embedding_input = np.random.randint(
        self.num_embeddings, size=(self.bsz, self.seq_len))
    embedding_input = jnp.asarray(embedding_input)

    rng = jax.random.PRNGKey(0)
    output, _ = model.init_with_output(
        rng,
        embedding_input=embedding_input,
    )

    # Check shape is correct
    self.assertSequenceEqual(output.shape,
                             (self.bsz, self.seq_len, self.embedding_dim))


class DictEmbedTest(absltest.TestCase):
  """DictEmbed test."""

  input_dim = 8
  num_embeddings = 10

  bsz = 2
  seq_len = 4

  def test_dict_embed(self):
    """Testing DictEmbed."""

    embedding_dict = {
        'A':
            embedding.Embed(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.input_dim,
            ),
        'B':
            embedding.Embed(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.input_dim,
            )
    }

    model = embedding.DictEmbed(embedding_dict)

    input_dict = {
        'A': jnp.ones(shape=(self.bsz, self.seq_len), dtype=jnp.int32),
        'B': jnp.ones(shape=(self.bsz, self.seq_len), dtype=jnp.int32),
    }

    rng = jax.random.PRNGKey(0)
    output, variables = model.init_with_output(
        rng,
        inputs=input_dict,
    )

    # Check shape is correct
    self.assertSequenceEqual(output.shape,
                             (self.bsz, self.seq_len, self.input_dim))

    # Check embedder adds up results correctly
    input_dict_a = {
        'A': jnp.ones(shape=(self.bsz, self.seq_len), dtype=jnp.int32),
    }
    output_a = model.apply(variables, input_dict_a)
    input_dict_b = {
        'B': jnp.ones(shape=(self.bsz, self.seq_len), dtype=jnp.int32),
    }
    output_b = model.apply(variables, input_dict_b)

    self.assertTrue(jnp.allclose(output, output_a + output_b))


if __name__ == '__main__':
  absltest.main()

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
"""Tests for Transformer."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from language.mentionmemory.modules import transformer


class TransformerTest(absltest.TestCase):
  """Transformer layer test."""

  num_layers = 4
  model_dim = 16
  intermediate_dim = 64
  num_heads = 4
  dtype = jnp.float32
  dropout_rate = 0.1

  bsz = 4
  seq_len = 20

  def test_transformer_layer_shape(self):
    """Testing transformer layer shape."""

    encoding = jnp.ones(
        shape=(self.bsz, self.seq_len, self.model_dim), dtype=self.dtype)

    attention_mask = jnp.ones(shape=(self.bsz, self.seq_len), dtype=self.dtype)

    model = transformer.TransformerLayer(
        model_dim=self.model_dim,
        intermediate_dim=self.intermediate_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
    )

    rng = jax.random.PRNGKey(0)
    output, _ = model.init_with_output(
        rng,
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=True,
    )

    self.assertSequenceEqual(output.shape, encoding.shape)

  def test_transformer_block_shape(self):
    """Testing transformer block shape."""

    encoding = jnp.ones(
        shape=(self.bsz, self.seq_len, self.model_dim), dtype=self.dtype)

    attention_mask = jnp.ones(shape=(self.bsz, self.seq_len), dtype=self.dtype)

    model = transformer.TransformerBlock(
        num_layers=self.num_layers,
        model_dim=self.model_dim,
        intermediate_dim=self.intermediate_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
    )

    rng = jax.random.PRNGKey(0)
    output, _ = model.init_with_output(
        rng,
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=True,
    )

    self.assertSequenceEqual(output.shape, encoding.shape)


if __name__ == '__main__':
  absltest.main()

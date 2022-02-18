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
"""Tests for layers."""

import itertools

from absl.testing import parameterized
import jax
import jax.numpy as jnp

from language.gscan.xattn_model.model import layers
import tensorflow as tf


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      list(itertools.product((1, 8), (8, 16), (False, True))))
  def test_transformer_embeddings(self, batch_size, hidden_size, deterministic):
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    vocab_size = 10
    type_vocab_size = 2
    max_seq_len = 30
    transformer_embeddings = layers.TransformerEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        deterministic=deterministic)
    inputs = jax.random.randint(
        jax.random.PRNGKey(0),
        (batch_size, max_seq_len), minval=0, maxval=vocab_size)
    outputs, _ = transformer_embeddings.init_with_output(rngs, inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_len, hidden_size))

  @parameterized.parameters(
      list(itertools.product((1, 8), (8, 16), ('max', 'first', 'mean'))))
  def test_transformer_pooler(self, batch_size, hidden_size, pooling_operator):
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    max_seq_len = 30
    embed_dim = 16
    transformer_pooler = layers.TransformerPooler(
        hidden_size=hidden_size, pooling_operator=pooling_operator)
    inputs = jnp.ones((batch_size, max_seq_len, embed_dim), jnp.float32)
    outputs, _ = transformer_pooler.init_with_output(rngs, inputs)
    self.assertEqual(outputs.shape, (batch_size, hidden_size))

  @parameterized.parameters(
      list(itertools.product((1, 8), (8, 16), (1, 4), (False, True))))
  def test_transformer_layer(self, batch_size, hidden_size, num_heads,
                             deterministic):
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    max_seq_len = 30
    transformer_layer = layers.TransformerLayer(
        num_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        deterministic=deterministic)
    inputs = jnp.ones((batch_size, max_seq_len, hidden_size), jnp.float32)
    outputs, _ = transformer_layer.init_with_output(rngs, inputs, inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_len, hidden_size))

  @parameterized.parameters(
      list(itertools.product((1, 8), (8, 16), (1, 4), (False, True))))
  def test_transformer_cross_layer(self, batch_size, hidden_size, num_heads,
                                   deterministic):
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    max_seq_len = 30
    max_region_num = 10
    transformer_layer = layers.TransformerCrossLayer(
        bi_num_heads=num_heads,
        bi_hidden_size=hidden_size,
        hidden_size1=hidden_size,
        hidden_size2=hidden_size,
        intermediate_size1=hidden_size,
        intermediate_size2=hidden_size,
        deterministic=deterministic)
    inputs = jnp.ones((batch_size, max_seq_len, hidden_size), jnp.float32)
    inputs2 = jnp.ones((batch_size, max_region_num, hidden_size), jnp.float32)
    outputs, _ = transformer_layer.init_with_output(rngs, inputs, inputs2)
    self.assertEqual(outputs[0].shape, (batch_size, max_seq_len, hidden_size))
    self.assertEqual(outputs[1].shape,
                     (batch_size, max_region_num, hidden_size))

  @parameterized.parameters(
      list(itertools.product((8, 16), (1, 4), (False, True), (False, True))))
  def test_transformer_encoder_decoder_layer(self, hidden_size, num_heads,
                                             deterministic, decode):
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    max_seq_len = 30
    max_region_num = 10
    batch_size = 8
    transformer_layer = layers.TransformerEncoderDecoderLayer(
        num_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        deterministic=deterministic,
        decode=decode)
    inputs = jnp.ones((batch_size, max_seq_len, hidden_size), jnp.float32)
    inputs2 = jnp.ones((batch_size, max_region_num, hidden_size), jnp.float32)
    outputs, _ = transformer_layer.init_with_output(rngs, inputs, inputs2)
    self.assertEqual(outputs.shape, (batch_size, max_seq_len, hidden_size))


if __name__ == '__main__':
  tf.test.main()

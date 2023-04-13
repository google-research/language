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
"""Tests for mlm layer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from language.mentionmemory.modules import mlm_layer
import numpy as np


class MLMLayerTest(absltest.TestCase):
  """MLM layer tests."""

  vocab_size = 1000
  hidden_size = 32
  dtype = jnp.float32
  layer_norm_epsilon = 1e-12

  seq_len = 20
  bsz = 4
  n_mentions = 3

  def test_mlm_layer(self):
    """Testing mlm layer."""

    encoded_input = jnp.ones(
        shape=(self.bsz, self.seq_len, self.hidden_size), dtype=self.dtype)
    mlm_target_positions = np.random.randint(
        self.seq_len, size=(self.bsz, self.n_mentions))

    model = mlm_layer.MLMLayer(  # pytype: disable=wrong-arg-types  # jax-types
        vocab_size=self.vocab_size,
        hidden_size=self.hidden_size,
        dtype=self.dtype,
        embedding_init=jax.nn.initializers.lecun_normal(),
        bias_init=jax.nn.initializers.zeros,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

    embeddings = np.random.rand(self.vocab_size, self.hidden_size)
    embeddings = jnp.asarray(embeddings, dtype=self.dtype)

    rng = jax.random.PRNGKey(0)
    output, _ = model.init_with_output(
        rng,
        encoded_input=encoded_input,
        mlm_target_positions=mlm_target_positions,
        shared_embedding=embeddings,
    )

    self.assertSequenceEqual(output.shape,
                             (self.bsz, self.n_mentions, self.vocab_size))


if __name__ == '__main__':
  absltest.main()

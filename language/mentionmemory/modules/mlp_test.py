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
"""Tests for mlp."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from language.mentionmemory.modules import mlp


class MLPBlockTest(absltest.TestCase):
  """MLP block test."""

  input_dim = 16
  hidden_dim = 64
  dtype = jnp.float32
  dropout_rate = 0.1
  layer_norm_epsilon = 1e-12

  bsz = 4
  seq_len = 20

  def test_mlp_block(self):
    """Testing mlp block."""

    x = jnp.ones(
        shape=(self.bsz, self.seq_len, self.input_dim), dtype=self.dtype)

    model = mlp.MLPBlock(
        input_dim=self.input_dim,
        hidden_dim=self.hidden_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

    rng = jax.random.PRNGKey(0)
    output, _ = model.init_with_output(
        rng,
        x=x,
        deterministic=True,
    )

    self.assertSequenceEqual(output.shape, x.shape)


if __name__ == '__main__':
  absltest.main()

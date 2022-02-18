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
"""Tests for transformer."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp

from language.gscan.xattn_model import test_utils
from language.gscan.xattn_model.model import models

import tensorflow as tf


class ModelsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_model(self, cross_attn):
    config = test_utils.get_model_test_config().to_dict()
    config.update(cross_attn=cross_attn)
    config = models.TransformerConfig(**config)
    batch_size = 8
    max_seq_len = 4
    grid_size = 4
    img_dim = 15
    vocab_size = config.target_vocab_size

    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
    inputs = {
        'index':
            jnp.zeros((batch_size,), dtype=jnp.int32),
        'token':
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
        'txt_mask':
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
        'target_token':
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
        'target_txt_mask':
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
        'image':
            jnp.zeros((batch_size, grid_size, grid_size, img_dim),
                      dtype=jnp.float32),
    }
    model = models.Model(config)
    outputs, _ = model.init_with_output(rngs, inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_len, vocab_size))


if __name__ == '__main__':
  tf.test.main()

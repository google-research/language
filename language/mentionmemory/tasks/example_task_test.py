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
"""Tests for ExampleTask."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from language.mentionmemory.tasks import example_task
import ml_collections


class ExampleTaskTest(absltest.TestCase):

  def test_output_expected(self):

    model_config_dict = {
        'dtype': 'float32',
        'features': [32, 32],
        'use_bias': False,
    }

    config_dict = {
        'model_config': model_config_dict,
        'per_device_batch_size': 32,
        'seed': 0,
    }
    config = ml_collections.ConfigDict(config_dict)
    model_config = ml_collections.FrozenConfigDict(model_config_dict)

    batch = {
        'x':
            jnp.zeros(
                (config.per_device_batch_size, config.model_config.features[0]),
                dtype=config.model_config.dtype,
            )
    }

    init_rng = jax.random.PRNGKey(config.seed)

    # Create dummy input
    dummy_input = example_task.ExampleTask.dummy_input(config)

    model = example_task.ExampleTask.build_model(model_config)
    initial_variables = jax.jit(model.init)(init_rng, dummy_input, False)
    loss_fn = example_task.ExampleTask.make_loss_fn(config)

    loss, _, _ = loss_fn(
        model_config=model_config,
        model_params=initial_variables['params'],
        model_vars={},
        batch=batch,
        deterministic=False,
    )

    self.assertEqual(loss, 0.0)


if __name__ == '__main__':
  absltest.main()

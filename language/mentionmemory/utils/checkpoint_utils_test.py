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
"""Tests for checkpoint_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
from language.mentionmemory.tasks import example_task
from language.mentionmemory.utils import checkpoint_utils
import ml_collections


class WeightTest(absltest.TestCase):
  """Test whether saving and loading weights works as expected."""

  weight_path = '/tmp/test/weights.test'

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

  def test_saving_loading_weights_returns_original(self):
    """Test whether saving weights and then loading them returns original."""

    # Create initial variables
    init_rng = jax.random.PRNGKey(self.config.seed)

    dummy_input = example_task.ExampleTask.dummy_input(self.config)

    model = example_task.ExampleTask.build_model(self.model_config)
    initial_variables = jax.jit(model.init)(init_rng, dummy_input, True)

    # Save weights
    checkpoint_utils.save_weights(self.weight_path, initial_variables['params'])

    # Load weights
    loaded_params = checkpoint_utils.load_weights(self.weight_path)

    arrayeq = lambda x, y: jnp.all(x == y)
    self.assertTrue(
        jax.tree_map(
            arrayeq,
            loaded_params,
            flax.core.unfreeze(initial_variables['params']),
        )
    )


class MergeNestedDictTest(parameterized.TestCase):
  """Test whether merging of nested dictionaries works as expected."""

  @parameterized.parameters(
      (dict(A=1), dict(B=2), dict(A=1, B=2)),
      (dict(A=1), dict(A=2), dict(A=2)),
      (dict(A=1, B=dict(C=3, D=dict(E=1))), dict(
          A=2, B=dict(
              F=5, D=dict(E=13))), dict(A=2, B=dict(C=3, F=5, D=dict(E=13)))),
      (dict(A=1, B=dict(C=3, D=dict(E=1))), dict(
          A=2, B=dict(
              F=5, D=dict(E=13))), dict(A=2, B=dict(C=3, F=5, D=dict(E=13)))),
  )
  def test_merge_nested_dicts(self, dict_a, dict_b, expected_result):
    checkpoint_utils.merge_nested_dicts(dict_a, dict_b)
    actual_result = checkpoint_utils.flatten_nested_dict(dict_a)
    expected_result = checkpoint_utils.flatten_nested_dict(expected_result)
    self.assertDictEqual(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()

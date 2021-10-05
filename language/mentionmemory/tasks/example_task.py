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
"""Simple task for testing purposes."""


import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.tasks import base_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils.custom_types import Array, Dtype, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections
import tensorflow.compat.v2 as tf


class ExampleModel(nn.Module):
  """Model corresponding to example task."""
  dtype: Dtype
  features: Sequence[int]
  use_bias: bool = True

  def setup(self):
    self.layers = [
        nn.Dense(feat, dtype=self.dtype, use_bias=self.use_bias)
        for feat in self.features
    ]

  def __call__(self, batch, deterministic):
    x = batch['x']
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)

    return x


@task_registry.register_task('example_task')
class ExampleTask(base_task.BaseTask):
  """Example task for illustration and testing purposes."""

  model_class = ExampleModel

  @classmethod
  def make_loss_fn(
      cls, config
  ):
    """Creates task loss function."""

    def loss_fn(
        model_config,
        model_params,
        model_vars,  # pylint: disable=unused-argument
        batch,
        deterministic,
        dropout_rng = None,  # pylint: disable=unused-argument
    ):
      """Task-specific loss function. See BaseTask."""
      encoding = cls.build_model(model_config).apply(
          {'params': model_params},
          batch,
          deterministic=deterministic,
      )

      loss = jnp.sum(encoding)
      metrics = {
          'agg': {
              'loss': loss,
              'denominator': batch['x'].shape[0],
          }
      }

      return loss, metrics, {}

    return loss_fn

  @staticmethod
  def make_preprocess_fn(
      config
  ):
    """Produces function to preprocess samples. See BaseTask."""

    def preprocess_fn(sample):
      return sample

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config
  ):
    """Produces function to preprocess batches. See BaseTask."""

    def collater_fn(batch):
      return batch

    return collater_fn

  @staticmethod
  def get_name_to_features(config):
    """Return feature dict for decoding purposes. See BaseTask."""

    name_to_features = {
        'x':
            tf.io.FixedLenFeature([config.model_config.features[0]],
                                  tf.float32),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config):
    """Produces model-specific dummy input batch. See BaseTask."""

    return {
        'x':
            jnp.ones(
                (config.per_device_batch_size, config.model_config.features[0]),
                dtype=config.model_config.dtype,
            )
    }

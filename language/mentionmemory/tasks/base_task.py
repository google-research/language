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
"""Defines abstract task class used as base for other tasks in package."""



import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.utils.custom_types import Array, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class BaseTask():
  """Abstract task class.

  Task classes contain task-specific methods for training a model, such as those
  responsible for data preprocessing, task objective, and model building.
  """

  model_class = nn.Module

  @classmethod
  def build_model(cls,
                  model_config):
    """Builds model by instantiating flax module associated with task."""
    return cls.model_class(**model_config)

  @classmethod
  def make_loss_fn(
      cls,  # pylint: disable=unused-argument
      config  # pylint: disable=unused-argument
  ):
    """Creates task loss function.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Loss function.
    """

    def loss_fn(
        model_config,
        model_params,
        model_vars,
        batch,
        deterministic,
        dropout_rng = None,
    ):
      """Model-specific loss function.

      Tasks are responsible for providing a loss function that
      produces a scalar loss and any relevant metrics.

      Args:
        model_config: contains model config hyperparameters.
        model_params: contains model parameters.
        model_vars: contains model variables (not optimized).
        batch: model input.
        deterministic: whether dropout etc should be applied.
        dropout_rng: seed for dropout randomness.

      Returns:
        Loss, metrics and auxiliary output.
      """
      raise NotImplementedError

    return loss_fn

  @staticmethod
  def get_name_to_features(
      config  # pylint: disable=unused-argument
  ):
    """Return feature dict for decoding purposes.

    Models are responsible for preprocessing and therefore should
    already know what raw features are coming from the pipeline.

    Args:
      config: experiment config.

    Returns:
      Dict mapping feature names to feature types.
    """
    raise NotImplementedError

  @staticmethod
  def make_preprocess_fn(
      config  # pylint: disable=unused-argument
  ):
    """Produces function to preprocess samples.

    Tasks should be aware of what type of input they require and are
    responsible for creating these inputs during data preprocessing.

    This default implementation provides an identity function, which returns
    input as-is without applying any preprocessing.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses samples to be usable for the model
      (mod casting from tf to jnp dtype).
    """

    def preprocess_fn(sample):
      return sample

    return preprocess_fn

  @staticmethod
  def make_collater_fn(
      config  # pylint: disable=unused-argument
  ):
    """Produces function to preprocess batches.

    The difference compared to `preprocess_fn` is that `collater_fn_` does not
    receive individual examples, but instead full batches as its input.

    Tasks should be aware of what type of input they require and are
    responsible for creating these inputs during data preprocessing.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that preprocesses batches to be usable for the model
      (mod casting from tf to jnp dtype).
    """

    def collater_fn(batch):
      raise NotImplementedError

    return collater_fn

  @staticmethod
  def dummy_input(config):
    """Produces model-specific dummy input batch.

    Tasks are responsible for producing a dummy input in the form of
    a py-tree with the same shape as the model input to be used in model
    initialization.

    Args:
      config: experiment config.

    Returns:
      Dict model input.
    """
    raise NotImplementedError

  @classmethod
  def load_weights(cls, config):
    """Load model weights from file.

    Args:
      config: experiment config.

    Returns:
      Dictionary of model weights.
    """
    raise NotImplementedError

  @classmethod
  def make_output_postprocess_fn(
      cls,
      config  # pylint: disable=unused-argument
  ):
    """Produces function to postprocess task samples (input and output).

    The method is occasionally called during training or evaluation to save
    model inputs and outputs for manual inspection. Given an input batch and
    model's auxiliary output, the method needs to produce JSON-serializable
    dictionary with all the relevant features.

    This default implementation provides a function that takes all of the
    features in the `batch` and in `auxiliary_output`, adjust array types if
    necessary (jnp.bfloat16 is not JSON-serializable) and converts all JAX and
    numpy arrays to list (arrays are not JSON-serializable).

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that postprocesses model's input and output for serialization.
    """

    def postprocess_fn(batch,
                       auxiliary_output):
      """Function that prepares model's input and output for serialization."""

      features = {}
      for dict_of_features in [batch, auxiliary_output]:
        for key, value in dict_of_features.items():
          convert_to_list = (
              isinstance(value, Array) or isinstance(value, np.ndarray))
          if convert_to_list and value.dtype == jnp.bfloat16:
            # bfloat16 is not JSON serializable => convert to float32.
            features[key] = value.astype(jnp.float32).tolist()
          elif convert_to_list:
            # Cannot serialize numpy / jax arrays to json => convert to list.
            features[key] = value.tolist()
          else:
            features[key] = value

      return features

    return postprocess_fn

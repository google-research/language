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
"""Defines abstract encoder class used as base for other encoders in package."""



import flax.linen as nn
import jax
from language.mentionmemory.utils import checkpoint_utils
from language.mentionmemory.utils.custom_types import Array
import ml_collections


class BaseEncoder(nn.Module):
  """Abstract encoder class.

  Encoders take a tokenized text sequence (possibly annotated with mentions) as
  input, and output a representation of the text sequence along with auxiliary
  values that may be useful for auxiliary objectives or logging (for example,
  mention representations).
  """

  def forward(
      self,
      batch,
      deterministic,
  ):
    """The forward pass of the encoder.

    Models that use an encoder should call this method to encode a passage.

    Args:
      batch: input to the encoder.
      deterministic: whether to apply dropout.

    Returns:
      A tuple of:
        [bsz, seq_len, model_dim] passage encoding.
        Dictionary of auxiliary values.
        Dictionary of logging values.
    """
    raise NotImplementedError

  @staticmethod
  def load_weights(config):
    """Load model weights from file."""
    params = checkpoint_utils.load_weights(config.load_weights)
    params = jax.device_put_replicated(params, jax.local_devices())
    return {'params': params}

  @classmethod
  def make_output_postprocess_fn(
      cls,
      config  # pylint: disable=unused-argument
  ):
    """Produces function to postprocess task samples (input and output).

    The method is called occasionally during training or evaluation to save
    model inputs and outputs for manual inspection. Given a input batch and
    model's auxiliary output the method needs to produce JSON-serializable
    dictionary with all the relevant features.

    This default implementation provides a function that returns an empty
    dictionary.

    Args:
      config: contains experiment hyperparameters.

    Returns:
      Function that postprocesses model's input and output for serialization.
    """

    def postprocess_fn(
        batch,  # pylint: disable=unused-argument
        auxiliary_output  # pylint: disable=unused-argument
    ):
      """Function that prepares model's input and output for serialization."""
      return {}

    return postprocess_fn

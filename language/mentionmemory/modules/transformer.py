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
"""Modules for Transformer encoder and layers."""



import flax.linen as nn

from language.mentionmemory.modules import attention
from language.mentionmemory.modules import mlp
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import


class TransformerLayer(nn.Module):
  """Transformer layer.

  Attributes:
    model_dim: dimensionality of model
    intermediate_dim: dimensionality of MLP block intermediate representation
    num_heads: number of attention heads
    dropout_rate: dropout rate
    dtype: datatype of computation
    layer_norm_epsilon: numerical stability parameter of layer norm.
    kernel_init: weight initializer function
    bias_init: bias initializer function
  """

  model_dim: int
  intermediate_dim: int
  num_heads: int
  dropout_rate: float
  dtype: Dtype
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

  def setup(self):
    self.attention_block = attention.AttentionBlock(
        model_dim=self.model_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

    self.mlp_block = mlp.MLPBlock(
        input_dim=self.model_dim,
        hidden_dim=self.intermediate_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        layer_norm_epsilon=self.layer_norm_epsilon,
    )

  def __call__(
      self,
      encoding,
      attention_mask,
      deterministic,
  ):
    """Transformer layer forward.

    Args:
      encoding: [bsz, seq_len, model_dim] model state.
      attention_mask: [bsz, seq_len].
      deterministic: if true, do not apply dropout.

    Returns:
      updated encoding
    """

    encoding = self.attention_block(
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=deterministic)

    encoding = self.mlp_block(x=encoding, deterministic=deterministic)

    return encoding


class LayerSequence(nn.Module):
  """Contains multiple encoder layers of same type.

  Layers should take an encoding as input and produce an encoding as the output.

  Attributes:
    num_layers: number of layers.
    layer_factory: method which produces layer when called.
  """

  num_layers: int
  layer_factory: Callable[Ellipsis, nn.Module]

  def setup(self):
    self.layers = [self.layer_factory() for _ in range(self.num_layers)]

  def __call__(self, encoding, *args, **kwargs):
    for layer in self.layers:
      encoding = layer(encoding, *args, **kwargs)
    return encoding


class TransformerBlock(nn.Module):
  """Block of Transformer layers.

  Attributes:
    num_layers: number of Transformer layers.
    model_dim: dimensionality of model.
    intermediate_dim: dimensionality of MLP block intermediate representation.
    num_heads: number of attention heads.
    dropout_rate: dropout rate.
    dtype: datatype of computation.
    kernel_init: weight initializer function.
    bias_init: bias initializer function.
    layer_norm_epsilon: numerical stability parameter of layer norm.
  """

  num_layers: int
  model_dim: int
  intermediate_dim: int
  num_heads: int
  dropout_rate: float
  dtype: Dtype
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

  def setup(self):

    def layer_factory():
      return TransformerLayer(
          model_dim=self.model_dim,
          intermediate_dim=self.intermediate_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          layer_norm_epsilon=self.layer_norm_epsilon,
      )

    self.layer_sequence = LayerSequence(
        num_layers=self.num_layers,
        layer_factory=layer_factory,
    )

  def __call__(
      self,
      encoding,
      attention_mask,
      deterministic,
  ):
    """Transformer layer forward.

    Args:
      encoding: [bsz, seq_len, model_dim] model state.
      attention_mask: [bsz, seq_len].
      deterministic: if true, do not apply dropout.

    Returns:
      Updated encoding.
    """
    return self.layer_sequence(
        encoding=encoding,
        attention_mask=attention_mask,
        deterministic=deterministic,
    )

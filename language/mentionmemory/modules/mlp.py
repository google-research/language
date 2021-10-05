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
"""Contains MLP layer block."""

import flax.linen as nn

from language.mentionmemory.utils import default_values
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import


class MLPBlock(nn.Module):
  """MLP block as found in Transformer architecture.

  Attributes:
    input_dim: dimensionality of input representation.
    hidden_dim: dimensionality of intermediate representation.
    dropout_rate: rate of dropout.
    dtype: precision of computation.
    layer_norm_epsilon: epsilon of layer norm.
  """

  input_dim: int
  hidden_dim: int
  dropout_rate: float
  dtype: Dtype
  layer_norm_epsilon: float = default_values.layer_norm_epsilon
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init

  def setup(self):
    self.mlp = nn.Dense(
        features=self.hidden_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)
    self.dense = nn.Dense(
        features=self.input_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)
    self.dropout = nn.Dropout(self.dropout_rate)
    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

  def __call__(self, x, deterministic):
    """Applies MLP block update.

    Args:
      x: [..., input_dim].
      deterministic: don't apply dropout if true.

    Returns:
      Updated array x of same shape.
    """
    update = nn.gelu(self.mlp(x))
    update = self.dense(update)
    update = self.dropout(update, deterministic=deterministic)
    x = self.layer_norm(x + update)

    return x

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
"""Contains attention layer block."""

import flax.linen as nn

from language.mentionmemory.utils import default_values
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import


class AttentionBlock(nn.Module):
  """Attention block as found in Transformer architecture.

  Attributes:
    num_heads: number of attention heads.
    model_dim: dimensionality of input representation.
    dropout_rate: rate of dropout.
    dtype: precision of computation.
    layer_norm_epsilon: numerical stability parameter of layer norm.
    kernel_init: kernel initializer function.
    bias_init: bias initializer function.
  """

  num_heads: int
  model_dim: int
  dropout_rate: float
  dtype: Dtype
  layer_norm_epsilon: float = default_values.layer_norm_epsilon
  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init

  def setup(self):
    self.attention_layer = nn.SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.model_dim,
        dropout_rate=self.dropout_rate,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
    )
    self.dropout = nn.Dropout(self.dropout_rate)
    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

  def __call__(self, encoding, attention_mask,
               deterministic):
    """Self attention layer forward.

    Args:
      encoding: [bsz, seq_len, model_dim] model state.
      attention_mask: [bsz, seq_len].
      deterministic: if true, do not apply dropout.

    Returns:
      Updated encoding.
    """

    attention_mask = nn.make_attention_mask(attention_mask, attention_mask)
    update = self.attention_layer(
        inputs_q=encoding, mask=attention_mask, deterministic=deterministic)
    update = self.dropout(update, deterministic=deterministic)
    encoding = self.layer_norm(encoding + update)

    return encoding

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
"""Contains mlm layer."""

import flax.linen as nn

from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import


class MLMLayer(nn.Module):
  """Performs masked language model scoring.

    Attributes:
      vocab_size: size of token vocabulary.
      hidden_size: dimensionality of input token representations.
      dtype: precision of computation.
  """

  vocab_size: int
  hidden_size: int
  dtype: Dtype
  embedding_init: InitType
  bias_init: InitType
  layer_norm_epsilon: float

  def setup(self):
    self.dense = nn.Dense(
        features=self.hidden_size,
        dtype=self.dtype,
    )
    self.layer_norm = nn.LayerNorm(self.layer_norm_epsilon)
    self.embedding_dense = nn.Dense(
        features=self.vocab_size,
        use_bias=False,
        dtype=self.dtype,
        kernel_init=self.embedding_init,
    )
    self.bias = self.param('bias', self.bias_init, (self.vocab_size,))

  def __call__(
      self,
      encoded_input,
      mlm_target_positions,
      shared_embedding,
  ):
    """Perform masked language modeling scoring.

    Args:
      encoded_input: [bsz, n_tokens, hidden_size].
      mlm_target_positions: [bsz, max_mlm_targets] positions of mlm targets in
        passage.
      shared_embedding: [vocab_size, hidden_size] word embedding array, shared
        with initial embedding.

    Returns:
      Array of masked language modeling logits.
    """

    target_encodings = jut.matmul_slice(encoded_input, mlm_target_positions)
    target_encodings = self.dense(target_encodings)
    target_encodings = nn.gelu(target_encodings)
    target_encodings = self.layer_norm(target_encodings)

    mlm_logits = self.embedding_dense.apply(
        {'params': {
            'kernel': shared_embedding.T
        }}, target_encodings)
    mlm_logits = mlm_logits + self.bias

    return mlm_logits

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
"""Contains embedding layers."""



import flax.linen as nn
import jax
import jax.numpy as jnp

from language.mentionmemory.utils import default_values
from language.mentionmemory.utils.custom_types import Array, Dtype, PRNGKey, Shape  # pylint: disable=g-multiple-import


class Embed(nn.Module):
  """Embedding layer.

  Attributes:
    num_embeddings: number of embeddings.
    embedding_dim: dimensionality of embeddings.
    dtype: precision of the layer output.
    embedding_init: embedding initializer.
  """
  num_embeddings: int
  embedding_dim: int
  dtype: Dtype = jnp.float32
  embedding_init: Callable[[PRNGKey, Shape, Dtype],
                           Array] = default_values.kernel_init

  def setup(self):
    self.embedding = self.param('embedding', self.embedding_init,
                                (self.num_embeddings, self.embedding_dim),
                                jnp.float32)

  def __call__(self, embedding_input):
    """Embeds the inputs along the last dimension.

    Args:
      embedding_input: [..., embedding_dim] integer array of embedding indices.

    Returns:
      Embedded inputs, with embedding dimension appended and cast to dtype.
    """
    return jnp.asarray(self.embedding[embedding_input], dtype=self.dtype)


class DictEmbed(nn.Module):
  """Adds up embeddings from dictionary of embedding tables.

  Attributes:
    embedders: dictionary of embedding tables.
  """
  embedders: Dict[str, Embed]

  def __call__(self, inputs):
    embeddings = {}
    for key, embedding_input in inputs.items():
      embeddings[key] = self.embedders[key](embedding_input)
    return jax.tree_util.tree_reduce(
        lambda value, embedding: value + embedding, tree=embeddings)

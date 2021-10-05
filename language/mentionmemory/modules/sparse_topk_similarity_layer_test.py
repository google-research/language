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
"""Tests for sparse topk similarity layer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from language.mentionmemory.modules import sparse_topk_similarity_layer
import numpy as np


class SparseTopKSimilarityLayerTest(parameterized.TestCase):
  """Sparse top-k similarity layer test."""

  dtype = jnp.float32
  vector_dim = 32
  query_size = 16
  table_size = 256

  @parameterized.parameters(
      (1, 1, 1, 1, 1),
      (1, 2, 2, 1, 1),
      (1, 1, 2, 1, 1),
      (1, 1, 1, 2, 1),
      (2, 2, 1, 1, 2),
  )
  def test_similarity_layer(self, k_top, clusters, rows, splits, n_search):
    """Testing similarity layer."""

    model = sparse_topk_similarity_layer.SparseTopKSimilarityLayer(
        k_top=k_top,
        splits=splits,
        n_search=n_search,
    )

    queries = np.random.rand(self.query_size, self.vector_dim)
    table = np.random.rand(clusters, rows, self.table_size // (rows * clusters),
                           self.vector_dim)
    prototypes = np.random.rand(clusters, self.vector_dim)

    queries = jnp.asarray(queries, dtype=self.dtype)
    queries = queries / jnp.linalg.norm(queries, axis=-1).reshape(-1, 1)
    table = jnp.asarray(table, dtype=self.dtype)
    prototypes = jnp.asarray(prototypes, dtype=self.dtype)

    topk_values, topk_scores, topk_ids = model.apply(
        {},
        queries=queries,
        table=table,
        prototypes=prototypes,
    )

    # Check shapes as expected
    self.assertSequenceEqual(topk_values.shape,
                             (self.query_size, k_top, self.vector_dim))
    self.assertSequenceEqual(topk_scores.shape, (self.query_size, k_top))
    self.assertSequenceEqual(topk_ids.shape, (self.query_size, k_top))


if __name__ == '__main__':
  absltest.main()

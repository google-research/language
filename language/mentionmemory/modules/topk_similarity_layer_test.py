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
"""Tests for topk_similarity_layer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from language.mentionmemory.modules import topk_similarity_layer
import numpy as np


class TopKSimilarityLayerTest(parameterized.TestCase):
  """Top k similarity layer test."""

  dtype = jnp.float32
  vector_dim = 32
  query_size = 16
  table_size = 128
  splits = 2
  k_top = 2

  @parameterized.parameters(
      (1, 1, 1),
      (1, 2, 1),
      (2, 2, 1),
      (1, 1, 2),
      (2, 2, 2),
  )
  def test_similarity_layer(self, k_top, rows, splits):
    """Testing similarity layer."""

    model = topk_similarity_layer.TopKSimilarityLayer(
        k_top=k_top, splits=splits)

    queries = np.random.rand(self.query_size, self.vector_dim)
    table = np.random.rand(rows, self.table_size // rows, self.vector_dim)
    test_high_scoring_vector = np.random.rand(self.vector_dim)

    table[0] = test_high_scoring_vector
    queries[0] = test_high_scoring_vector

    queries = jnp.asarray(queries, dtype=self.dtype)
    queries = queries / jnp.linalg.norm(queries, axis=-1).reshape(-1, 1)
    table = jnp.asarray(
        table, dtype=self.dtype) / jnp.linalg.norm(
            table, axis=-1).reshape(rows, -1, 1)
    test_high_scoring_vector = jnp.asarray(test_high_scoring_vector, self.dtype)
    test_high_scoring_vector = test_high_scoring_vector / jnp.linalg.norm(
        test_high_scoring_vector)

    topk_values, topk_scores, topk_ids = model.apply(
        {},
        queries=queries,
        table=table,
    )

    # Check shapes as expected
    self.assertSequenceEqual(topk_values.shape,
                             (self.query_size, k_top, self.vector_dim))
    self.assertSequenceEqual(topk_scores.shape, (self.query_size, k_top))
    self.assertSequenceEqual(topk_ids.shape, (self.query_size, k_top))

    # Check correctly retrieves identical vector from table
    self.assertTrue(jnp.allclose(topk_values[0, 0], test_high_scoring_vector))
    self.assertEqual(topk_ids[0, 0], 0)
    self.assertAlmostEqual(
        topk_scores[0, 0],
        jnp.dot(test_high_scoring_vector, test_high_scoring_vector),
        places=4)


if __name__ == '__main__':
  absltest.main()

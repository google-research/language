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
"""Contains topk similarity layer."""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array


class TopKSimilarityLayer(nn.Module):
  """Performs approximate top-k similarity search over vector table.

  This layer is an implementation of the ScamTPUTopK operation. Given queries,
  the layer retrieves the top-k (approximately) most similar vectors from a
  vector table. Similarity scores are computed exactly, but the top-k operation
  is approximate as exact top-k is very expensive to compute on TPU.

  The vector table is expected to be in the form of [rows, values per row, dim].
  Search is performed by first finding the closest vector in each row, and then
  retrieving the k closest vectors from those by performing standard top-k.
  This procedure yields different results from exact search if a single row
  contains multiple vectors that should be in the top-k.

  Attributes:
    k_top: number of vectors to retrieve.
    splits: governs a tradeoff between speed and memory usage and has no effect
      on actual search results. A higher number of splits is slower but uses
      less memory due to smaller intermediate score tensor.
  """

  k_top: int
  splits: int

  def __call__(
      self,
      queries: Array,
      table: Array,
  ) -> Tuple[Array, Array, Array]:
    """Perform approximate top-k similarity search over vector table.

    Args:
      queries: [n_queries, vector_dim].
      table: [rows, values per row, vector_dim] vector table. The number of rows
        in the table governs the recall vs speed of the topk similarity search.
        Search is performed by taking max over each row, and then top-k between
        rows. Distributing the same values over more rows leads to higher recall
        but slower search.

    Returns:
      Top-k vectors, scores and ids.
    """
    n_queries = queries.shape[0]
    queries_per_split = n_queries // self.splits
    scores_per_row = table.shape[1]
    table_size = scores_per_row * table.shape[0]
    vector_dim = queries.shape[1]
    assert table.shape[-1] == vector_dim

    # Split queries to reduce size of intermediate score tensor and save memory.
    queries = queries.reshape(self.splits, queries_per_split, vector_dim)

    def split_top_k(split_queries):
      split_scores = jnp.einsum('qd,rvd->qrv', split_queries, table)

      # Find highest scoring vector for each row.
      top_id_by_row = jnp.argmax(split_scores, axis=-1)
      top_score_by_row = jnp.max(split_scores, axis=-1)

      # Take k highest scores among all rows.
      top_row_idx = jnp.argsort(
          top_score_by_row, axis=-1)[:, :-self.k_top - 1:-1]

      # Sub-select best indices for k best rows.
      ids_by_topk_row = jut.matmul_slice(top_id_by_row, top_row_idx)

      # Gather highest scoring vectors for k best rows.
      split_topk_values = table[top_row_idx, ids_by_topk_row]

      # Convert row indices to indices into flattened table.
      top_table_id_by_row = top_id_by_row + jnp.arange(0, table_size,
                                                       scores_per_row)
      # Get best ids into flattened table.
      split_topk_ids = jut.matmul_slice(top_table_id_by_row, top_row_idx)

      split_topk_scores = jut.matmul_slice(top_score_by_row, top_row_idx)

      return split_topk_values, split_topk_scores, split_topk_ids

    # Perform similarity over each chunk of queries sequentially
    # (not in parallel), so that only one score tensor is in memory at a time.
    topk_values, topk_scores, topk_ids = jax.lax.map(split_top_k, queries)

    topk_values = topk_values.reshape(n_queries, self.k_top, -1)
    topk_scores = topk_scores.reshape(n_queries, self.k_top)
    topk_ids = topk_ids.reshape(n_queries, self.k_top)

    return topk_values, topk_scores, topk_ids

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
"""Contains sparse topk similarity layer."""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array


class SparseTopKSimilarityLayer(nn.Module):
  """Performs sparse approximate top-k similarity search over vector table.

  Given queries, the layer retrieves the top-k (approximately) most similar
  vectors from a vector table through two-level similarity search.

  The vector table is expected to be in the form of [n_clusters,
  rows_per_cluster, values per row, dim]. The layer also expects an array of
  cluster prototypes. Search is performed by first finding the clusters with
  highest similarity between query and prototype, and then performing
  brute-force scoring with approximate top-k (as in the standard
  topk_similarity_layer) within the selected clusters.

  Attributes:
    k_top: Number of vectors to retrieve.
    n_search: the number of clusters to perform brute-force search over. Higher
      n_search is slower but leads to better recall.
    splits: Governs a tradeoff between speed and memory usage and has no effect
      on actual search results. A higher number of splits is slower but uses
      less memory due to smaller intermediate score tensor.
  """

  k_top: int
  n_search: int
  splits: int

  def __call__(
      self,
      queries: Array,
      table: Array,
      prototypes: Array,
  ) -> Tuple[Array, Array, Array]:
    """Perform approximate top-k similarity search over vector table.

    Args:
      queries: [n_queries, vector_dim].
      table: [n_clusters, rows, values per row, vector_dim] vector table.
      prototypes: [n_clusters, vector_dim] representative vectors for clusters.

    Returns:
      Top-k vectors, scores and ids.
    """
    n_queries = queries.shape[0]
    queries_per_split = n_queries // self.splits

    rows_per_cluster = table.shape[1]
    values_per_row = table.shape[2]
    values_per_cluster = rows_per_cluster * values_per_row

    table_size = values_per_row * table.shape[0]
    vector_dim = queries.shape[1]
    assert table.shape[-1] == vector_dim

    # Split queries to reduce size of selected clusters and save memory.
    queries = queries.reshape(self.splits, queries_per_split, vector_dim)

    def split_top_k(split_queries: Array) -> Tuple[Array, Array, Array]:
      # Find most similar clusters
      prototype_scores = jnp.einsum('qd,pd->qp', split_queries, prototypes)
      top_indices = jax.lax.top_k(prototype_scores, self.n_search)[1]
      # Perform approximate top-k similarity search over most similar clusters.
      selected_data = table[top_indices]
      split_scores = jnp.einsum('qd,qcrvd->qcrv', split_queries, selected_data)

      # Find highest scoring vector for each row.
      top_id_by_row = jnp.argmax(split_scores, axis=-1)
      top_score_by_row = jnp.max(split_scores, axis=-1)

      top_id_by_row = top_id_by_row.reshape(queries_per_split,
                                            self.n_search * rows_per_cluster)
      top_score_by_row = top_score_by_row.reshape(
          queries_per_split, self.n_search * rows_per_cluster)

      # Take k highest scores among all rows.
      top_row_idx = jnp.argsort(
          top_score_by_row, axis=-1)[:, :-self.k_top - 1:-1]

      # Sub-select best indices for k best rows.
      ids_by_topk_row = jut.matmul_slice(top_id_by_row, top_row_idx)

      # Gather highest scoring vectors for k best rows.
      query_index = jnp.arange(queries_per_split).reshape(-1, 1).tile(
          [1, self.k_top])
      top_cluster_idx, top_cluster_row_idx = jnp.divmod(top_row_idx,
                                                        rows_per_cluster)
      split_topk_values = selected_data[query_index, top_cluster_idx,
                                        top_cluster_row_idx, ids_by_topk_row]

      row_offset = jnp.mod(
          jnp.arange(0, self.n_search * values_per_cluster, values_per_row),
          values_per_cluster)
      cluster_offset = jnp.arange(0, table_size, values_per_cluster)

      # Convert row indices to indices into flattened table.
      top_table_id_by_row = top_id_by_row + row_offset.reshape(
          1, -1) + cluster_offset[top_indices].repeat(
              rows_per_cluster, axis=-1)
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

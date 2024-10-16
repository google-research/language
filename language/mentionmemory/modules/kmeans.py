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
"""K-means clustering algorithm."""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from language.mentionmemory.utils.custom_types import Array


def l2_distance(a: Array, b: Array):
  """Compute l2 distance between arrays."""
  aa = jnp.einsum('ad,ad->a', a, a)
  bb = jnp.einsum('bd,bd->b', b, b)
  ab = jnp.einsum('ad,bd->ab', a, b)
  squared_dist = jnp.expand_dims(
      aa, axis=1) + jnp.expand_dims(
          bb, axis=0) - 2 * ab
  dist = jnp.sqrt(squared_dist)
  return dist


def compute_assignments(
    centroids: Array,
    observations: Array,
    n_splits: int,
) -> Tuple[Array, Array]:
  """Assigns observations to cluster centroids.

  Computes l2 distance between each pair of observation and centroids, and
  assigns observation to closest cluster. Because the array of pairwise
  distances is very large, the cluster assignment is performed chunk by chunk.

  Args:
    centroids: [n_clusters, dim] cluster centroids.
    observations: [n_observations, dim] data points.
    n_splits: split observations into this many chunks.

  Returns:
    assignments: [n_observations] closest cluster for each observation.
    min_dist: [n_observations] distance to closest cluster by observation.
  """
  reshaped_observations = observations.reshape(n_splits, -1,
                                               centroids.shape[-1])

  def compute_split_assignments(split_points):
    dist = l2_distance(split_points, centroids)
    split_assignments = jnp.argmin(dist, axis=-1)
    split_min_dist = jnp.min(dist, axis=-1)
    return split_assignments, split_min_dist

  assignments, min_dist = jax.lax.map(compute_split_assignments,
                                      reshaped_observations)
  assignments = assignments.reshape(-1)
  min_dist = min_dist.reshape(-1)

  return assignments, min_dist


def kmeans_step(
    val: Tuple[Array, Array, float, Optional[float], int],
    n_splits: int,
    parallel_computation: bool = False,
) -> Tuple[Array, Array, float, float, int]:
  """Perform single K-means step.

  Standard K-means step. Assigns observations to nearest cluster, then updates
  cluster centroids as mean of assigned observations. Inputs are packed into
  'val' to facilitate while loop condition for K-means.

  Args:
    val: tuple of [n_clusters, dim] cluster centroids. [n_observations, dim]
      observations. prev_dist, mean distance between observations and closest
      cluster centroid in previous K-means step. prev_2_dist, distance from two
      steps prior. step, idx of current step.
    n_splits: number of splits for compute assignments
    parallel_computation: if true, assumes is run inside pmap with
      'observations' axis.

  Returns:
    new_centroids: [n_clusters, dim] new cluster centroids.
    observations: [n_observations, dim].
    mean_dist: mean distance between observations and closest cluster centroid.
    prev_dist: mean distance for previous K-means step.
    step: idx of next step.
  """
  centroids, observations, prev_dist, _, step = val
  assignments, min_dist = compute_assignments(centroids, observations, n_splits)

  mean_dist = jnp.mean(min_dist)

  # Compute new cluster centroids as average of observations in cluster
  cluster_sums = jnp.zeros(centroids.shape).at[assignments].add(observations)
  counts = jnp.bincount(assignments, length=centroids.shape[0])

  if parallel_computation:
    cluster_sums = jax.lax.psum(cluster_sums, axis_name='observations')
    counts = jax.lax.psum(counts, axis_name='observations')
    mean_dist = jax.lax.pmean(mean_dist, axis_name='observations')

  new_centroids = cluster_sums / counts[:, None].clip(min=1.0)

  jax.debug.print('step={}', step)
  jax.debug.print('dist_diff={}', mean_dist - prev_dist)
  step += 1

  return new_centroids, observations, mean_dist, prev_dist, step  # type: ignore


def kmeans(
    observations: Array,
    centroids: Array,
    n_splits: int,
    threshold: float = 1e-5,
    max_iterations: Optional[int] = None,
    parallel_computation: bool = False) -> Tuple[Array, Array, float, int]:
  """Perform K-means clustering.

  Args:
    observations: [n_observations, dim] data points.
    centroids: [n_clusters, dim] initial cluster centroids.
    n_splits: split observations into this many chunks during scoring.
    threshold: stop computation if updates lead to improvement less than this.
    max_iterations: stop computation after this many iterations.
    parallel_computation: if true, assumes is run inside pmap.

  Returns:
    centroids: [n_clusters, dim] new cluster centroids.
    assignments: [n_observations] final assignment of observations to clusters.
    dist_diff: improvement in last k-means step.
    step: idx of next step.
  """
  # Run one step of K-means to get the initial value for the while loop.
  partial_step = functools.partial(
      kmeans_step, n_splits=n_splits, parallel_computation=parallel_computation)
  initial_val = partial_step((centroids, observations, jnp.inf, None, 0))

  def condition_fun(val) -> bool:
    below_max_iters = (max_iterations is None or val[4] < max_iterations)
    above_threshold = val[3] - val[2] > threshold
    should_continue = below_max_iters & above_threshold
    return should_continue

  centroids, _, mean_dist, prev_dist, step = jax.lax.while_loop(
      condition_fun, partial_step, initial_val)

  assignments, _ = compute_assignments(centroids, observations, n_splits)

  return centroids, assignments, prev_dist - mean_dist, step

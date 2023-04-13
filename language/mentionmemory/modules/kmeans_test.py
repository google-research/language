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
"""Tests for K-means clustering."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.modules import kmeans
from language.mentionmemory.utils import test_utils
import numpy as np


class L2Test(test_utils.TestCase):

  dim = 16
  size_a = 8
  size_b = 32

  def test_l2_equal_to_numpy(self):
    a = np.random.rand(self.size_a, self.dim)
    b = np.random.rand(self.size_b, self.dim)

    computed_distance = kmeans.l2_distance(a, b)
    reference_distance = np.linalg.norm(
        np.expand_dims(a, 1) - np.expand_dims(b, 0), axis=-1)

    self.assertArrayAlmostEqual(
        np.asarray(computed_distance), reference_distance, 5
    )


class KMeansTest(test_utils.TestCase):
  """K-means clustering tests."""

  n_clusters = 8
  n_observations = 128
  dim = 64
  n_splits = 2
  n_devices = 4

  def setUp(self):
    super().setUp()
    test_utils.force_multi_devices(self.n_devices)

  def test_compute_assignments(self):
    centroids = np.random.rand(self.n_clusters, self.dim)
    observations = np.random.rand(self.n_observations, self.dim)

    assignments, min_dist = kmeans.compute_assignments(centroids, observations,
                                                       1)

    reference_assignments = np.zeros_like(assignments)
    reference_min_dist = np.zeros_like(min_dist)
    for idx in range(self.n_observations):
      observation = observations[idx]
      centroid_distances = [
          np.linalg.norm(observation - centroid) for centroid in centroids
      ]
      assignment = np.argmin(centroid_distances)
      distance = centroid_distances[assignment]
      reference_assignments[idx] = assignment
      reference_min_dist[idx] = distance

    self.assertArrayEqual(np.asarray(assignments), reference_assignments)
    self.assertArrayAlmostEqual(
        np.asarray(min_dist), reference_min_dist, places=5
    )

  def test_kmeans_step(self):
    centroids = np.random.rand(self.n_clusters, self.dim)
    observations = np.random.rand(self.n_observations, self.dim)

    def partial_kmeans_step(centroids, observations):
      return kmeans.kmeans_step((centroids, observations, 0, None, 0),
                                self.n_splits,
                                parallel_computation=False)

    new_centroids, observations_copy, _, prev_min_dist, step = partial_kmeans_step(
        centroids, observations)

    self.assertEqual(step, 1)
    self.assertEqual(prev_min_dist, 0)
    self.assertEqual(new_centroids.shape, centroids.shape)
    self.assertEqual(observations_copy.shape, observations.shape)

    # Perform same tests with parallel version
    sharded_observations = observations.reshape(self.n_devices, -1, self.dim)

    def partial_pkmeans_step(centroids, observations):
      return kmeans.kmeans_step((centroids, observations, 0, None, 0),
                                self.n_splits,
                                parallel_computation=True)

    pkmeans_step = jax.pmap(
        partial_pkmeans_step,
        axis_name='observations',
        in_axes=(None, 0),
        out_axes=(None, 0, None, None, None),
        devices=jax.devices())

    new_centroids, observations_copy, _, prev_min_dist, step = pkmeans_step(
        centroids, sharded_observations)
    self.assertEqual(step, 1)
    self.assertEqual(prev_min_dist, 0)
    self.assertEqual(new_centroids.shape, centroids.shape)
    self.assertEqual(observations_copy.shape, sharded_observations.shape)

  @parameterized.parameters(
      (1e-3, None),
      (1e-5, None),
      (1e-7, None),
      (1e-7, 2),
      (1e-7, 5),
      (1e-7, 10),
  )
  def test_kmeans(self, threshold, max_iterations):
    centroids = np.random.rand(self.n_clusters, self.dim)
    observations = np.random.rand(self.n_observations, self.dim)

    partial_kmeans = functools.partial(
        kmeans.kmeans,
        n_splits=self.n_splits,
        threshold=threshold,
        max_iterations=max_iterations,
        parallel_computation=False)

    new_centroids, assignments, dist_diff, step = partial_kmeans(
        observations, centroids)

    self.assertEqual(new_centroids.shape, centroids.shape)
    self.assertEqual(assignments.shape, (self.n_observations,))

    if max_iterations is not None:
      self.assertLessEqual(step, max_iterations)
    else:
      self.assertLessEqual(dist_diff, threshold)

    # Perform same tests with parallel version
    sharded_observations = observations.reshape(self.n_devices, -1, self.dim)
    partial_pkmeans = functools.partial(
        kmeans.kmeans,
        n_splits=self.n_splits,
        threshold=threshold,
        max_iterations=max_iterations)

    pkmeans = jax.pmap(
        partial_pkmeans,
        axis_name='observations',
        in_axes=(0, None),
        out_axes=(None, 0, None, None),
    )

    new_centroids, assignments, dist_diff, step = pkmeans(
        sharded_observations, centroids)

    self.assertEqual(new_centroids.shape, centroids.shape)
    self.assertEqual(assignments.shape,
                     (self.n_devices, self.n_observations // self.n_devices))

    if max_iterations is not None:
      self.assertLessEqual(step, max_iterations)
    else:
      self.assertLessEqual(dist_diff, threshold)


if __name__ == '__main__':
  absltest.main()

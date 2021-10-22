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
"""Tests for jax utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from language.mentionmemory.utils import jax_utils as jut
import numpy as np
import scipy.spatial

_MAX_INT_VALUE = 100000000


class SliceTest(parameterized.TestCase):
  """Test whether slices produces similar values."""

  @parameterized.parameters(
      (1, 1, 1, 1),
      (3, 1, 1, 1),
      (1, 3, 1, 1),
      (1, 1, 1, 3),
      (7, 20, 5, 11),
  )
  def test_slice_values_float(self, bsz, seq_len, index_len, dim):
    # no batch dim
    array = np.random.rand(seq_len, dim)
    indices = np.random.randint(seq_len, size=(index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = array[indices]
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))

    # 2d array
    array = np.random.rand(bsz, seq_len)
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = jut.vmap_slice(array, indices)
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))

    # 3d array
    array = np.random.rand(bsz, seq_len, dim)
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = jut.vmap_slice(array, indices)
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))

  @parameterized.parameters(
      (1, 1, 1, 1),
      (3, 1, 1, 1),
      (1, 3, 1, 1),
      (1, 1, 1, 3),
      (7, 20, 5, 11),
  )
  def test_slice_values_int(self, bsz, seq_len, index_len, dim):
    # no batch dim
    array = np.random.randint(_MAX_INT_VALUE, size=(seq_len, dim))
    indices = np.random.randint(seq_len, size=(index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = array[indices]
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))

    # 2d array
    array = np.random.randint(_MAX_INT_VALUE, size=(bsz, seq_len))
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = jut.vmap_slice(array, indices)
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))

    # 3d array
    array = np.random.randint(_MAX_INT_VALUE, size=(bsz, seq_len, dim))
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    matmul_slice = jut.matmul_slice(array, indices)
    vmap_slice = jut.vmap_slice(array, indices)
    self.assertTrue(jnp.allclose(matmul_slice, vmap_slice))


class IndexSelectTest(parameterized.TestCase):
  """Test whether `matmul_2d_index_select` implementation is correct."""

  @parameterized.parameters(
      (4, 3, None, 1),
      (7, 2, None, 5),
      (2, 3, None, 10),
      (3, 2, 9, 2),
      (2, 3, 7, 5),
  )
  def test_matmul_2d_index_select(self, dim1, dim2, dim3, n_index):
    shape = [dim1, dim2]
    if dim3 is not None:
      shape.append(dim3)
    array = np.random.randint(_MAX_INT_VALUE, size=shape)
    indices_1 = np.random.randint(dim1, size=(n_index))
    indices_2 = np.random.randint(dim2, size=(n_index))
    actual = jut.matmul_2d_index_select(array, (indices_1, indices_2))
    self.assertTrue(jnp.array_equal(actual, array[indices_1, indices_2]))


class IndexAddTest(parameterized.TestCase):
  """Test whether index_add produces similar values."""

  @parameterized.parameters(
      (1, 1, 1, 1),
      (3, 1, 1, 1),
      (1, 3, 1, 1),
      (1, 1, 1, 3),
      (7, 20, 5, 11),
      (2, 3, 7, 5),
      (7, 5, 3, 2),
      (11, 13, 5, 5),
  )
  def test_add_values_float(self, bsz, seq_len, index_len, dim):
    array = np.random.rand(bsz, seq_len, dim)
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    values = np.random.rand(bsz, index_len, dim)
    matmul_add = jut.matmul_index_add(array, indices, values)
    vmap_add = jut.vmap_index_add(array, indices, values)
    self.assertTrue(jnp.allclose(matmul_add, vmap_add))

  @parameterized.parameters(
      (1, 1, 1, 1),
      (3, 1, 1, 1),
      (1, 3, 1, 1),
      (1, 1, 1, 3),
      (7, 20, 5, 11),
      (2, 3, 7, 5),
      (7, 5, 3, 2),
      (11, 13, 5, 5),
  )
  def test_add_values_int(self, bsz, seq_len, index_len, dim):
    array = np.random.randint(_MAX_INT_VALUE, size=(bsz, seq_len, dim))
    indices = np.random.randint(seq_len, size=(bsz, index_len))
    values = np.random.randint(_MAX_INT_VALUE, size=(bsz, index_len, dim))
    matmul_add = jut.matmul_index_add(array, indices, values)
    vmap_add = jut.vmap_index_add(array, indices, values)
    self.assertTrue(jnp.allclose(matmul_add, vmap_add))


class Index2DAddTest(parameterized.TestCase):
  """Test whether index_add produces similar values."""

  @parameterized.parameters(
      (4, 3, None, 1),
      (7, 2, None, 5),
      (2, 3, None, 10),
      (3, 2, 3, 2),
      (2, 3, 7, 5),
      (7, 5, 3, 2),
      (11, 13, 5, 5),
  )
  def test_matmul_2d_index_add(self, dim1, dim2, dim3, n_index):
    shape_array = [dim1, dim2]
    shape_values = [n_index]
    if dim3 is not None:
      shape_array.append(dim3)
      shape_values.append(dim3)
    array = np.random.randint(_MAX_INT_VALUE, size=shape_array)
    indices_1 = np.random.randint(dim1, size=(n_index))
    indices_2 = np.random.randint(dim2, size=(n_index))
    values = np.random.randint(_MAX_INT_VALUE, size=shape_values)
    expected = array.copy()
    # NOTE: this naive numpy implementation does not work
    # if there are index contain duplicates
    # expected[indices_1, indices_2] += values
    for i in range(n_index):
      expected[indices_1[i], indices_2[i]] += values[i]
    actual = jut.matmul_2d_index_add(array, (indices_1, indices_2), values)
    self.assertTrue(jnp.array_equal(actual, expected))


class CosineSimilarityTest(parameterized.TestCase):
  """Test whether index_add produces similar values."""

  @parameterized.parameters(
      (1, 1),
      (1, 2),
      (10, 10),
      (10, 20),
  )
  def test_matmul_2d_index_add(self, batch_size, hidden_dim):
    a = np.random.random((batch_size, hidden_dim))
    b = np.random.random((batch_size, hidden_dim))
    actual_cos_sim = jut.cosine_similarity(a, b)
    for i in range(batch_size):
      expected_cos_sim = 1 - scipy.spatial.distance.cosine(a[i], b[i])
      self.assertAlmostEqual(actual_cos_sim[i], expected_cos_sim, places=4)


if __name__ == '__main__':
  absltest.main()

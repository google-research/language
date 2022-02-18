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
"""Common utilities for data preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp


def filter_sparse_rows(sp_mat, max_cols):
  """Filter rows of a CSR sparse matrix to retain upto max_cols."""
  all_remove_idx = []
  num_removed = np.zeros((sp_mat.indptr.shape[0] - 1,), dtype=np.int64)
  for ii in range(sp_mat.shape[0]):
    row_st = sp_mat.indptr[ii]
    row_en = sp_mat.indptr[ii + 1]
    my_scores = sp_mat.data[row_st:row_en]
    if len(my_scores) > max_cols:
      remove = np.argpartition(-my_scores, max_cols)[max_cols:]
      all_remove_idx.extend(remove + row_st)
    num_removed[ii] = len(all_remove_idx)
  new_data = np.delete(sp_mat.data, all_remove_idx)
  new_indices = np.delete(sp_mat.indices, all_remove_idx)
  new_indptr = np.copy(sp_mat.indptr)
  new_indptr[1:] -= num_removed
  return sp.csr_matrix((new_data, new_indices, new_indptr), shape=sp_mat.shape)

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
"""Utilities for performing dense or sparse search."""

import collections
import math
import random

import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
import tensorflow.compat.v1 as tf
from tqdm import tqdm


def get_ngrams(seq, n):
  """Yields n-grams upto order n."""
  for nn in range(1, n + 1):
    for j in range(nn, len(seq) + 1):
      yield u" ".join(seq[j - nn:j])


def mm3hash(token, num_buckets):
  """Returns a murmur hash for given string."""
  return murmurhash3_32(token, positive=True) % num_buckets


def get_hashed_counts(seq, ngrams, hash_size):
  """Returns hash bucket counts for ngrams in sequence."""
  return collections.Counter(
      [mm3hash(ng, hash_size) for ng in get_ngrams(seq, ngrams)])


def build_count_matrix(input_ids,
                       input_mask,
                       ngrams=2,
                       hash_size=int(math.pow(2, 24))):
  """Build a sparse matrix of size ngrams to doc ids with counts."""
  rows, cols, data = [], [], []
  for ii in tqdm(range(len(input_ids))):
    my_seq = [
        str(tok)
        for jj, tok in enumerate(input_ids[ii])
        if input_mask[ii][jj] == 1
    ]
    my_counts = get_hashed_counts(my_seq, ngrams, hash_size)
    rows.extend(my_counts.keys())
    cols.extend([ii] * len(my_counts))
    data.extend(my_counts.values())
  return sp.csr_matrix((data, (rows, cols)), shape=(hash_size, len(input_ids)))


def counts_to_idfs(count_matrix, cutoff=0.):
  """Compute IDFs for the vocab given a V x N count matrix.

  Args:
    count_matrix: Sparse matrix from vocab to documents.
    cutoff: Fraction of lowest IDF words to remove (i.e. set IDF=0).

  Returns:
    idfs: Sparse diagonal matrix of IDF values.
  """
  freqs = np.array((count_matrix > 0).astype(int).sum(1)).squeeze()
  idfs = np.log((count_matrix.shape[1] - freqs + 0.5) / (freqs + 0.5))
  idfs[idfs < 0] = 0
  sorted_idx = np.argsort(idfs)
  num_to_remove = int(sorted_idx.shape[0] * cutoff)
  tf.logging.info("Removing %d lowest IDF words.", num_to_remove)
  idx_to_keep = sorted_idx[num_to_remove:]
  idfs = sp.csr_matrix((idfs[idx_to_keep], (idx_to_keep, idx_to_keep)),
                       shape=[idfs.shape[0], idfs.shape[0]])
  return idfs


def counts_to_tfidf(count_matrix, idfs):
  """Convert counts to Tf-Idf scores given diagonal idf matrix."""
  tfs = count_matrix.log1p()
  tfidfs = idfs.dot(tfs)
  return tfidfs


def write_to_checkpoint(var_name, np_db, dtype, checkpoint_path):
  """Write np array to checkpoint."""
  with tf.Graph().as_default():
    init_value = tf.py_func(lambda: np_db, [], dtype, stateful=False)
    init_value.set_shape(np_db.shape)
    tf_db = tf.get_variable(var_name, initializer=init_value)
    saver = tf.train.Saver([tf_db])
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      saver.save(session, checkpoint_path)


def write_ragged_to_checkpoint(var_name, sp_mat, checkpoint_path):
  """Write scipy CSR matrix to checkpoint for loading to ragged tensor."""
  data = sp_mat.data
  indices = sp_mat.indices
  rowsplits = sp_mat.indptr
  with tf.Graph().as_default():
    init_data = tf.py_func(
        lambda: data.astype(np.float32), [], tf.float32, stateful=False)
    init_data.set_shape(data.shape)
    init_indices = tf.py_func(
        lambda: indices.astype(np.int64), [], tf.int64, stateful=False)
    init_indices.set_shape(indices.shape)
    init_rowsplits = tf.py_func(
        lambda: rowsplits.astype(np.int64), [], tf.int64, stateful=False)
    init_rowsplits.set_shape(rowsplits.shape)
    tf_data = tf.get_variable(var_name + "_data", initializer=init_data)
    tf_indices = tf.get_variable(
        var_name + "_indices", initializer=init_indices)
    tf_rowsplits = tf.get_variable(
        var_name + "_rowsplits", initializer=init_rowsplits)
    saver = tf.train.Saver([tf_data, tf_indices, tf_rowsplits])
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      saver.save(session, checkpoint_path)
  with tf.gfile.Open(checkpoint_path + ".info", "w") as f:
    f.write(str(sp_mat.shape[0]) + " " + str(sp_mat.getnnz()))


def write_sparse_to_checkpoint(var_name, sp_mat, checkpoint_path):
  """Write scipy sparse CSR matrix to checkpoint."""
  sp_mat = sp_mat.tocoo()
  # Sort the indices lexicographically.
  sort_i = np.lexsort((sp_mat.col, sp_mat.row))
  indices = np.mat([sp_mat.row[sort_i], sp_mat.col[sort_i]]).transpose()
  data = sp_mat.data[sort_i]
  with tf.Graph().as_default():
    init_data = tf.py_func(
        lambda: data.astype(np.float32), [], tf.float32, stateful=False)
    init_data.set_shape(data.shape)
    init_indices = tf.py_func(
        lambda: indices.astype(np.int64), [], tf.int64, stateful=False)
    init_indices.set_shape(indices.shape)
    init_shape = tf.py_func(
        lambda: np.array(sp_mat.shape, dtype=np.int64), [],
        tf.int64,
        stateful=False)
    init_shape.set_shape([len(sp_mat.shape)])
    tf_data = tf.get_variable(var_name + "_data", initializer=init_data)
    tf_indices = tf.get_variable(
        var_name + "_indices", initializer=init_indices)
    tf_shape = tf.get_variable(var_name + "_shape", initializer=init_shape)
    saver = tf.train.Saver([tf_data, tf_indices, tf_shape])
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      saver.save(session, checkpoint_path)
  with tf.gfile.Open(checkpoint_path + ".info", "w") as f:
    f.write(str(sp_mat.getnnz()))


def load_ragged_matrix(var_name, checkpoint_path):
  """Load sparse matrix from checkpoint."""
  with tf.gfile.Open(checkpoint_path + ".info") as f:
    num_row, num_nnz = [int(xx) for xx in f.read().split()]
  tf_data = tf.get_local_variable(
      var_name + "_data", shape=[num_nnz], dtype=tf.float32, use_resource=True)
  tf_indices = tf.get_local_variable(
      var_name + "_indices", shape=[num_nnz], dtype=tf.int64, use_resource=True)
  tf_rowsplits = tf.get_local_variable(
      var_name + "_rowsplits",
      shape=[num_row + 1],
      dtype=tf.int64,
      use_resource=True)
  init_from_checkpoint(
      checkpoint_path, target_variables=[tf_data, tf_indices, tf_rowsplits])
  return tf_data, tf_indices, tf_rowsplits


def load_sparse_matrix(var_name, checkpoint_path):
  """Load sparse matrix from checkpoint."""
  with tf.gfile.Open(checkpoint_path + ".info") as f:
    num_nnz = int(f.read())
  tf_data = tf.get_local_variable(
      var_name + "_data", shape=[num_nnz], dtype=tf.float32, use_resource=True)
  tf_indices = tf.get_local_variable(
      var_name + "_indices",
      shape=[num_nnz, 2],
      dtype=tf.int64,
      use_resource=True)
  tf_shape = tf.get_local_variable(
      var_name + "_shape", shape=[2], dtype=tf.int64, use_resource=True)
  init_from_checkpoint(
      checkpoint_path, target_variables=[tf_data, tf_indices, tf_shape])
  tf_sp = tf.SparseTensor(tf_indices, tf_data, tf_shape)
  return tf_sp


def log_variables(name, var_names):
  tf.logging.info("%s (%d total): %s", name, len(var_names),
                  random.sample(var_names, min(len(var_names), 5)))


def init_from_checkpoint(checkpoint_path,
                         checkpoint_prefix=None,
                         variable_prefix=None,
                         target_variables=None):
  """Initializes all of the variables using `init_checkpoint."""
  tf.logging.info("Loading variables from %s", checkpoint_path)
  checkpoint_variables = {
      name: name for name, _ in tf.train.list_variables(checkpoint_path)
  }
  if target_variables is None:
    target_variables = tf.trainable_variables()
  target_variables = {var.name.split(":")[0]: var for var in target_variables}

  if checkpoint_prefix is not None:
    checkpoint_variables = {
        checkpoint_prefix + "/" + name: varname
        for name, varname in checkpoint_variables.items()
    }
  if variable_prefix is not None:
    target_variables = {
        variable_prefix + "/" + name: var
        for name, var in target_variables.items()
    }

  checkpoint_var_names = set(checkpoint_variables.keys())
  target_var_names = set(target_variables.keys())
  intersected_var_names = target_var_names & checkpoint_var_names
  assignment_map = {
      checkpoint_variables[name]: target_variables[name]
      for name in intersected_var_names
  }
  tf.train.init_from_checkpoint(checkpoint_path, assignment_map)

  log_variables("Loaded variables", intersected_var_names)
  log_variables("Uninitialized variables",
                target_var_names - checkpoint_var_names)
  log_variables("Unused variables", checkpoint_var_names - target_var_names)


def load_database(var_name, shape, checkpoint_path, dtype=tf.float32):
  """Load variable from checkpoint."""
  if shape is None:
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    shape = var_to_shape_map[var_name]
  tf_db = tf.get_local_variable(
      var_name, shape=shape, dtype=dtype, use_resource=True)
  init_from_checkpoint(checkpoint_path, target_variables=[tf_db])
  return tf_db


def create_mips_searcher(var_name,
                         checkpoint_path,
                         num_neighbors,
                         local_var_name="mips_init_barrier"):
  """Create searcher for returning top-k closest elements."""
  tf_db = load_database(var_name, None, checkpoint_path)

  with tf.control_dependencies([tf_db.initializer]):
    mips_init_barrier = tf.constant(True)

  # Make sure DB is initialized.
  tf.get_local_variable(local_var_name, initializer=mips_init_barrier)

  def _search(query):
    with tf.device("/cpu:0"):
      distance = tf.matmul(query, tf_db, transpose_b=True)
      topk_dist, topk_idx = tf.nn.top_k(distance, num_neighbors)
    topk_dist.set_shape([query.shape[0], num_neighbors])
    topk_idx.set_shape([query.shape[0], num_neighbors])
    return topk_dist, topk_idx

  return tf_db, _search

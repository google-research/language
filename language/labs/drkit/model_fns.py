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
# Lint as: python3
"""Collection of model functions implementing different multihop variants."""

from bert import modeling
from language.labs.drkit import search_utils
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

DEFAULT_VALUE = -10000.


def ragged_sparse_lookup(ragged_ind,
                         ragged_val,
                         row_indices,
                         num_mentions,
                         mask=None,
                         fix_values_to_one=True):
  """Extract rows from sparse matrix and return as another sparse matrix.

  Args:
    ragged_ind: RaggedTensor mapping entities to mention indices. (E x M)
    ragged_val: RaggedTensor mapping entities to mention scores. (E x M)
    row_indices: Batch indices into rows of sparse matrix. (B)
    num_mentions: Scalar (M)
    mask: Boolean mask into the row_indices. Positions where this is false will
      not be retrieved. (B)
    fix_values_to_one: If true final result is binary.

  Returns:
    sp_vec: (B x M) sparse matrix of retrieved rows.
  """
  if mask is None:
    mask = tf.ones_like(row_indices, dtype=tf.bool)

  retained_idx = tf.squeeze(tf.where(mask), 1)
  row_indices = tf.boolean_mask(row_indices, mask)  # NNZ

  batch_idx = tf.gather(ragged_ind, row_indices).to_sparse()  # NNZ
  orig_idx = tf.gather(retained_idx, batch_idx.indices[:, 0])  # B
  batch_idx = tf.concat(
      (tf.expand_dims(orig_idx, 1), tf.expand_dims(batch_idx.values, 1)),
      1)  # B x 2

  if fix_values_to_one:
    return tf.SparseTensor(
        batch_idx,
        tf.ones((tf.shape(batch_idx)[0],), dtype=tf.float32),
        dense_shape=[tf.shape(row_indices)[0], num_mentions])
  else:
    batch_val = tf.gather(ragged_val, row_indices).values  # B
    return tf.SparseTensor(
        batch_idx,
        batch_val,
        dense_shape=[tf.shape(row_indices)[0], num_mentions])


def sparse_ragged_mul(sp_tensor,
                      ragged_ind,
                      ragged_val,
                      batch_size,
                      num_mentions,
                      reduce_fn="sum",
                      threshold=None,
                      fix_values_to_one=True):
  """Multiply two sparse matrices, one of which is in ragged form.

  Args:
    sp_tensor: SparseTensor of shape B x E.
    ragged_ind: RaggedTensor mapping entities to mention indices. (E x M)
    ragged_val: RaggedTensor mapping entities to mention scores. (E x M)
    batch_size: Scalar (B)
    num_mentions: Scalar (M)
    reduce_fn: Aggregation function to use (`max` or `sum`).
    threshold: Scalar. Only entries greater than this will be multiplied from
      the SparseTensor.
    fix_values_to_one: Whether to fix Ragged tensor values to 1.

  Returns:
    sp_vec: (B x M) sparse matrix of retrieved rows.
  """
  row_ind = sp_tensor.indices[:, 0]
  col_ind = sp_tensor.indices[:, 1]
  if threshold is not None:
    mask = tf.greater(sp_tensor.values, threshold)
    row_ind = tf.boolean_mask(row_ind, mask)  # NNZ
    col_ind = tf.boolean_mask(col_ind, mask)  # NNZ
    mask_val = tf.boolean_mask(sp_tensor.values, mask)  # NNZ

  gather_idx = tf.gather(ragged_ind, col_ind).to_sparse()  # NNZ x (M)
  row_gather_idx = tf.gather(row_ind, gather_idx.indices[:, 0])  # NNZ'
  out_sp_indices = tf.concat(
      (tf.expand_dims(row_gather_idx, 1), tf.expand_dims(gather_idx.values, 1)),
      1)

  gather_val = tf.gather(ragged_val, col_ind).to_sparse()  # NNZ x (M)
  row_gather_val = tf.gather(mask_val, gather_val.indices[:, 0])  # NNZ'
  if fix_values_to_one:
    out_sp_values = row_gather_val
  else:
    out_sp_values = row_gather_val * gather_val.values

  agg_indices, agg_values = aggregate_sparse_indices(out_sp_indices,
                                                     out_sp_values,
                                                     [batch_size, num_mentions],
                                                     reduce_fn)

  return tf.SparseTensor(
      agg_indices, agg_values, dense_shape=[batch_size, num_mentions])


def sparse_dense_mul(sp_mat, dense_mat):
  """Element-wise multiplication between sparse and dense tensors.

  Returns a sparse tensor. Limited broadcasting of dense_mat is supported.
  If rank(dense_mat) < rank(sparse_mat), then dense_mat is broadcasted on the
  rightmost dimensions to match sparse_mat.

  Args:
    sp_mat: SparseTensor.
    dense_mat: DenseTensor with rank <= sp_mat.

  Returns:
    SparseTensor.
  """
  rank = dense_mat.get_shape().ndims
  indices = sp_mat.indices[:, :rank]
  dense_values = tf.gather_nd(dense_mat, indices)
  return tf.SparseTensor(sp_mat.indices, sp_mat.values * dense_values,
                         sp_mat.dense_shape)


def batch_gather(params, indices):
  """Gather a batch of indices from a batch of params."""
  batch_size = tf.shape(indices)[0]
  num_idx = tf.shape(indices)[1]
  brange = tf.cast(tf.range(batch_size), indices.dtype)
  bindices = tf.tile(tf.expand_dims(brange, 1), (1, num_idx))
  gather_indices = tf.stack([bindices, indices], axis=2)
  return tf.gather_nd(params, gather_indices)


def convert_search_to_vector(dist, idx, batch_size, num_nn, vecsize):
  """Create vector from search indices and distances."""
  brange = tf.range(batch_size, dtype=tf.int32)
  bindices = tf.tile(tf.expand_dims(brange, 1), (1, num_nn))
  indices = tf.reshape(
      tf.stack([bindices, idx], axis=2), (batch_size * num_nn, 2))
  values = tf.reshape(dist, (batch_size * num_nn,))
  return tf.SparseTensor(
      indices=tf.cast(indices, dtype=tf.int64),
      values=values,
      dense_shape=[batch_size, vecsize])


def aggregate_sparse_indices(indices, values, shape, agg_fn="sum"):
  """Sums values corresponding to repeated indices.

  Returns the unique indices and their summed values.

  Args:
    indices: [num_nnz, rank] Tensor.
    values: [num_nnz] Tensor.
    shape: [rank] Tensor.
    agg_fn: Method to use for aggregation - `sum` or `max`.

  Returns:
    indices: [num_uniq, rank] Tensor.
    values: [num_uniq] Tensor.
  """
  # Linearize the indices.
  scaling_vec = tf.cumprod(tf.cast(shape, indices.dtype), exclusive=True)
  linearized = tf.linalg.matvec(indices, scaling_vec)
  # Get the unique indices, and their positions in the array
  y, idx = tf.unique(linearized)
  # Use the positions of the unique values as the segment ids to
  # get the unique values
  idx.set_shape([None])
  if agg_fn == "sum":
    values = tf.unsorted_segment_sum(values, idx, tf.shape(y)[0])
  elif agg_fn == "max":
    values = tf.unsorted_segment_max(values, idx, tf.shape(y)[0])
  elif agg_fn == "mean":
    values = tf.unsorted_segment_mean(values, idx, tf.shape(y)[0])
  # Go back to ND indices
  y = tf.expand_dims(y, 1)
  indices = tf.floormod(
      tf.floordiv(y, tf.expand_dims(scaling_vec, 0)),
      tf.cast(tf.expand_dims(shape, 0), indices.dtype))
  return indices, values


def ensure_values_in_mat(mat, values, dtype):
  """Ensure indices are present in the rows of mat.

  This replaces the last value in each row of mat with the corresponding
  value in values, IF that value is not in any of the remaining columns.

  Args:
    mat: [batch_size, M] Tensor.
    values: [batch_size] Tensor.
    dtype: Data type of mat.

  Returns:
    [batch_size, M] Tensor.
  """
  val_in_rows = tf.reduce_sum(
      tf.cast(tf.equal(mat[:, :-1], tf.expand_dims(values, 1)), dtype), axis=1)
  last_col = val_in_rows * mat[:, -1] + (1 - val_in_rows) * values
  return tf.concat((mat[:, :-1], tf.expand_dims(last_col, 1)), axis=1)


def sparse_reduce(sp_tensor, rank, agg_fn="sum", axis=-1):
  """Reduce SparseTensor along the given axis.

  Args:
    sp_tensor: SparseTensor of arbitrary rank.
    rank: Integer rank of the sparse tensor.
    agg_fn: Reduce function for aggregation.
    axis: Integer specifying axis to sum over.

  Returns:
    sp_tensor: SparseTensor of one less rank.
  """
  if axis < 0:
    axis = rank + axis
  axes_to_keep = tf.one_hot(
      axis, rank, on_value=False, off_value=True, dtype=tf.bool)
  indices_to_keep = tf.boolean_mask(sp_tensor.indices, axes_to_keep, axis=1)
  new_shape = tf.boolean_mask(sp_tensor.dense_shape, axes_to_keep)
  indices_to_keep.set_shape([None, rank - 1])
  indices, values = aggregate_sparse_indices(
      indices_to_keep, sp_tensor.values, new_shape, agg_fn=agg_fn)
  return tf.sparse.reorder(tf.SparseTensor(indices, values, new_shape))


def sp_mul(sp_a_indices, sp_a_values, sp_b, default_value=0.):
  """Element-wise multiply two SparseTensors.

  One SparseTensor has a fixed number of non-zero values, represented in the
  form of a B x K matrix of indices and the other in a B x K matrix of values.

  Args:
    sp_a_indices: B x K Tensor. Indices into M values.
    sp_a_values: B x K Tensor.
    sp_b: B x M SparseTensor.
    default_value: Scalar default value to use for indices which do not
      intersect.

  Returns:
    mul_values: B x K Tensor corresponding to indices in sp_a_indices. For
      indices also present in sp_b the output is the product of the values in
      sp_a_values. For those absent in sp_b, the output is default_value.
  """
  batch_size = tf.shape(sp_a_indices)[0]
  num_neighbors = tf.shape(sp_a_indices)[1]
  a_values = tf.reshape(sp_a_values, (batch_size * num_neighbors,))
  brange = tf.tile(
      tf.expand_dims(tf.cast(tf.range(batch_size), sp_a_indices.dtype), 1),
      (1, num_neighbors))
  nnz_indices = tf.stack([brange, sp_a_indices], axis=2)
  nnz_indices = tf.reshape(nnz_indices, (batch_size * num_neighbors, 2))

  # Linearize
  scaling_vector = tf.cumprod(tf.cast(tf.shape(sp_b), sp_b.indices.dtype))
  a1s = tf.linalg.matvec(
      tf.cast(nnz_indices, sp_b.indices.dtype), scaling_vector)
  b1s = tf.linalg.matvec(sp_b.indices, scaling_vector)

  # Get intersections.
  a_nonint, _ = tf.setdiff1d(a1s, b1s)
  a_int, a_int_idx = tf.setdiff1d(a1s, a_nonint)
  b_nonint, _ = tf.setdiff1d(b1s, a1s)
  b_int, b_int_idx = tf.setdiff1d(b1s, b_nonint)

  # Get sorting.
  a_int_sort_ind = tf.argsort(a_int)
  b_int_sort_ind = tf.argsort(b_int)

  # Multiply.
  int_vals = (
      tf.gather(a_values, tf.gather(a_int_idx, a_int_sort_ind)) *
      tf.gather(sp_b.values, tf.gather(b_int_idx, b_int_sort_ind)))
  int_vals_dense = tf.sparse_to_dense(
      sparse_indices=tf.expand_dims(tf.gather(a_int_idx, a_int_sort_ind), 1),
      output_shape=tf.shape(a_values),
      default_value=default_value,
      sparse_values=int_vals,
      validate_indices=False)

  return tf.reshape(int_vals_dense, (batch_size, num_neighbors))


def remove_from_sparse(sp_tensor, remove_indices):
  """Remove indices from SparseTensor if present."""
  # 1. create 1d index maps
  scaling_vector = tf.cumprod(
      tf.cast(tf.shape(sp_tensor), sp_tensor.indices.dtype))
  a1s = tf.linalg.matvec(sp_tensor.indices, scaling_vector)
  b1s = tf.linalg.matvec(remove_indices, scaling_vector)

  # 2. get relevant indices of sp_a
  int_idx = tf.setdiff1d(a1s, b1s)[1]
  to_retain = tf.sparse_to_dense(
      sparse_indices=int_idx,
      output_shape=tf.shape(a1s),
      default_value=0.0,
      sparse_values=1.0) > 0.5
  return tf.sparse.retain(sp_tensor, to_retain)


def sp_sp_matmul(sp_a, sp_b):
  """Element-wise multiply two SparseTensors of same shape."""
  sp_a = tf.sparse.reorder(sp_a)
  sp_b = tf.sparse.reorder(sp_b)

  # 1. create 1d index maps
  scaling_vector = tf.cumprod(tf.cast(tf.shape(sp_b), sp_b.indices.dtype))
  a1s = tf.linalg.matvec(sp_a.indices, scaling_vector)
  b1s = tf.linalg.matvec(sp_b.indices, scaling_vector)

  # 2. get relevant indices of sp_a
  int_idx = tf.setdiff1d(a1s, b1s)[1]
  to_retain = tf.sparse_to_dense(
      sparse_indices=int_idx,
      output_shape=tf.shape(a1s),
      default_value=0.0,
      sparse_values=1.0) < 0.5
  rsp_a = tf.sparse.retain(sp_a, to_retain)
  rsp_a = tf.sparse.reorder(rsp_a)

  # 3. get relevant indices of sp_b
  int_idx = tf.setdiff1d(b1s, a1s)[1]
  to_retain = tf.sparse_to_dense(
      sparse_indices=int_idx,
      output_shape=tf.shape(b1s),
      default_value=0.0,
      sparse_values=1.0) < 0.5
  rsp_b = tf.sparse.retain(sp_b, to_retain)
  rsp_b = tf.sparse.reorder(rsp_b)

  # 4. create output matrix
  return tf.SparseTensor(
      indices=rsp_b.indices,
      values=rsp_a.values * rsp_b.values,
      dense_shape=sp_b.dense_shape)


def entity_emb(entity_ind, entity_word_ids, entity_word_masks, word_emb_table,
               word_weights):
  """Get BOW embeddings for entities."""
  # [NNZ, max_entity_len]
  batch_entity_word_ids = tf.gather(entity_word_ids, entity_ind)
  batch_entity_word_masks = tf.gather(entity_word_masks, entity_ind)
  # [NNZ, max_entity_len, dim]
  batch_entity_word_emb = tf.gather(word_emb_table, batch_entity_word_ids)
  # [NNZ, max_entity_len, 1]
  batch_entity_word_weights = tf.gather(word_weights, batch_entity_word_ids)
  # [NNZ, dim]
  batch_entity_emb = tf.reduce_sum(
      batch_entity_word_emb * batch_entity_word_weights *
      tf.expand_dims(batch_entity_word_masks, 2),
      axis=1)
  return batch_entity_emb


def rescore_sparse(sp_mentions, tf_db, qrys):
  """Rescore mentions in sparse tensor with given batch of queries."""
  batch_ind = sp_mentions.indices[:, 0]
  mention_ind = sp_mentions.indices[:, 1]
  mention_vec = tf.gather(tf_db, mention_ind)
  batch_qrys = tf.gather(qrys, batch_ind)
  mention_scs = tf.reduce_sum(batch_qrys * mention_vec, 1)
  return tf.SparseTensor(
      indices=sp_mentions.indices,
      values=mention_scs,
      dense_shape=sp_mentions.dense_shape)


def follow(batch_entities,
           relation_st_qry,
           relation_en_qry,
           entity_word_ids,
           entity_word_masks,
           ent2ment_ind,
           ent2ment_val,
           ment2ent_map,
           word_emb_table,
           word_weights,
           mips_search_fn,
           tf_db,
           hidden_size,
           mips_config,
           qa_config,
           is_training,
           ensure_index=None):
  """Sparse implementation of the relation follow operation.

  Args:
    batch_entities: [batch_size, num_entities] SparseTensor of incoming entities
      and their scores.
    relation_st_qry: [batch_size, dim] Tensor representating start query vectors
      for dense retrieval.
    relation_en_qry: [batch_size, dim] Tensor representating end query vectors
      for dense retrieval.
    entity_word_ids: [num_entities, max_entity_len] Tensor holding word ids of
      each entity.
    entity_word_masks: [num_entities, max_entity_len] Tensor with masks into
      word ids above.
    ent2ment_ind: [num_entities, num_mentions] RaggedTensor mapping entities to
      mention indices which co-occur with them.
    ent2ment_val: [num_entities, num_mentions] RaggedTensor mapping entities to
      mention scores which co-occur with them.
    ment2ent_map: [num_mentions] Tensor mapping mentions to their entities.
    word_emb_table: [vocab_size, dim] Tensor of word embedddings.
    word_weights: [vocab_size, 1] Tensor of word weights.
    mips_search_fn: Function which accepts a dense query vector and returns the
      top-k indices closest to it.
    tf_db: [num_mentions, 2 * dim] Tensor of mention representations.
    hidden_size: Scalar dimension of word embeddings.
    mips_config: mipsConfig object.
    qa_config: QAConfig object.
    is_training: Boolean.
    ensure_index: [batch_size] Tensor of mention ids. Only needed if
      `is_training` is True.

  Returns:
    ret_mentions_ids: [batch_size, k] Tensor of retrieved mention ids.
    ret_mentions_scs: [batch_size, k] Tensor of retrieved mention scores.
    ret_entities_ids: [batch_size, k] Tensor of retrieved entities ids.
  """
  if qa_config.entity_score_threshold is not None:
    mask = tf.greater(batch_entities.values, qa_config.entity_score_threshold)
    batch_entities = tf.sparse.retain(batch_entities, mask)
  batch_size = batch_entities.dense_shape[0]
  batch_ind = batch_entities.indices[:, 0]
  entity_ind = batch_entities.indices[:, 1]
  entity_scs = batch_entities.values

  # Obtain BOW embeddings for the given set of entities.
  # [NNZ, dim]
  batch_entity_emb = entity_emb(entity_ind, entity_word_ids, entity_word_masks,
                                word_emb_table, word_weights)
  batch_entity_emb = batch_entity_emb * tf.expand_dims(entity_scs, axis=1)
  # [batch_size, dim]
  uniq_batch_ind, uniq_idx = tf.unique(batch_ind)
  agg_emb = tf.unsorted_segment_sum(batch_entity_emb, uniq_idx,
                                    tf.shape(uniq_batch_ind)[0])
  batch_bow_emb = tf.scatter_nd(
      tf.expand_dims(uniq_batch_ind, 1), agg_emb,
      tf.stack([batch_size, hidden_size], axis=0))
  batch_bow_emb.set_shape([None, hidden_size])
  if qa_config.projection_dim is not None:
    with tf.variable_scope("projection"):
      batch_bow_emb = contrib_layers.fully_connected(
          batch_bow_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=tf.AUTO_REUSE,
          scope="bow_projection")

  # Ragged sparse search.
  # [batch_size x num_mentions] sparse
  sp_mention_vec = sparse_ragged_mul(
      batch_entities,
      ent2ment_ind,
      ent2ment_val,
      batch_size,
      mips_config.num_mentions,
      qa_config.sparse_reduce_fn,
      threshold=qa_config.entity_score_threshold,
      fix_values_to_one=qa_config.fix_sparse_to_one)
  if is_training and qa_config.ensure_answer_sparse:
    ensure_indices = tf.stack([tf.range(batch_size), ensure_index], axis=-1)
    sp_ensure_vec = tf.SparseTensor(
        tf.cast(ensure_indices, tf.int64),
        tf.ones([batch_size]),
        dense_shape=[batch_size, mips_config.num_mentions])
    sp_mention_vec = tf.sparse.add(sp_mention_vec, sp_ensure_vec)
    sp_mention_vec = tf.SparseTensor(
        indices=sp_mention_vec.indices,
        values=tf.minimum(1., sp_mention_vec.values),
        dense_shape=sp_mention_vec.dense_shape)

  # Dense mips search.
  # [batch_size, 2 * dim]
  mips_qrys = tf.concat(
      [batch_bow_emb + relation_st_qry, batch_bow_emb + relation_en_qry],
      axis=1)
  with tf.device("/cpu:0"):
    # [batch_size, num_neighbors]
    _, ret_mention_ids = mips_search_fn(mips_qrys)
    if is_training and qa_config.ensure_answer_dense:
      ret_mention_ids = ensure_values_in_mat(ret_mention_ids, ensure_index,
                                             tf.int32)
    # [batch_size, num_neighbors, 2 * dim]
    ret_mention_emb = tf.gather(tf_db, ret_mention_ids)

  if qa_config.l2_normalize_db:
    ret_mention_emb = tf.nn.l2_normalize(ret_mention_emb, axis=2)
  # [batch_size, 1, num_neighbors]
  ret_mention_scs = tf.matmul(
      tf.expand_dims(mips_qrys, 1), ret_mention_emb, transpose_b=True)
  # [batch_size, num_neighbors]
  ret_mention_scs = tf.squeeze(ret_mention_scs, 1)
  # [batch_size, num_mentions] sparse
  dense_mention_vec = convert_search_to_vector(ret_mention_scs, ret_mention_ids,
                                               tf.cast(batch_size, tf.int32),
                                               mips_config.num_neighbors,
                                               mips_config.num_mentions)

  # Combine sparse and dense search.
  if (is_training and qa_config.train_with_sparse) or (
      (not is_training) and qa_config.predict_with_sparse):
    # [batch_size, num_mentions] sparse
    if qa_config.sparse_strategy == "dense_first":
      ret_mention_vec = sp_sp_matmul(dense_mention_vec, sp_mention_vec)
    elif qa_config.sparse_strategy == "sparse_first":
      with tf.device("/cpu:0"):
        ret_mention_vec = rescore_sparse(sp_mention_vec, tf_db, mips_qrys)
    else:
      raise ValueError("Unrecognized sparse_strategy %s" %
                       qa_config.sparse_strategy)
  else:
    # [batch_size, num_mentions] sparse
    ret_mention_vec = dense_mention_vec

  # Get entity scores and ids.
  # [batch_size, num_entities] sparse
  entity_indices = tf.cast(
      tf.gather(ment2ent_map, ret_mention_vec.indices[:, 1]), tf.int64)
  ret_entity_vec = tf.SparseTensor(
      indices=tf.concat(
          [ret_mention_vec.indices[:, 0:1],
           tf.expand_dims(entity_indices, 1)],
          axis=1),
      values=ret_mention_vec.values,
      dense_shape=[batch_size, qa_config.num_entities])

  return ret_entity_vec, ret_mention_vec, dense_mention_vec, sp_mention_vec


def compute_loss_sparse(logits, indices, weights=None):
  """Cross-entropy loss when only 1 label per example."""
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=indices, logits=logits)
  if weights is not None:
    row_sums = tf.squeeze(batch_gather(weights, tf.expand_dims(indices, 1)), 1)
    return tf.reduce_sum(row_sums * losses) / tf.maximum(
        tf.reduce_sum(row_sums), 1e-12), row_sums
  else:
    return tf.reduce_mean(losses), None


def compute_loss(logits, indices, weights=None):
  """Cross-entropy loss when multiple labels per example."""
  if weights is not None:
    indices = indices * weights
    indices = indices / tf.maximum(
        1e-12, tf.reduce_sum(indices, axis=1, keepdims=True))
    row_sums = tf.reduce_sum(indices, axis=1)
  losses = tf.nn.softmax_cross_entropy_with_logits(
      labels=indices, logits=logits)
  if weights is not None:
    return tf.reduce_sum(row_sums * losses) / tf.maximum(
        tf.reduce_sum(row_sums), 1e-12), row_sums
  else:
    return tf.reduce_mean(losses), None


def compute_loss_from_sptensors(sp_probs, sp_answers):
  """Cross-entropy loss when both answers and probabilities are sparse."""
  # First take log of the probabilities.
  mask = tf.greater(sp_probs.values, 1e-12)
  sp_probs = tf.sparse.retain(sp_probs, mask)
  sp_nll = tf.SparseTensor(
      indices=sp_probs.indices,
      values=-tf.log(sp_probs.values),
      dense_shape=sp_probs.dense_shape)
  return sp_sp_matmul(sp_nll, sp_answers)


def answer_in_retrieval(retrieved_entities, answer_index):
  """Check for which batch elements an answer is retrieved."""
  retrieved_index = batch_gather(answer_index, retrieved_entities)
  retrieved_scores = tf.reduce_sum(retrieved_index, axis=1)
  return tf.cast(tf.greater(retrieved_scores, 0.), tf.float32)


def get_bert_embeddings(model, layers_to_use, aggregation_fn, name="bert"):
  """Extract embeddings from BERT model."""
  all_hidden = model.get_all_encoder_layers()
  layers_hidden = [all_hidden[i] for i in layers_to_use]
  hidden_shapes = [
      modeling.get_shape_list(hid, expected_rank=3) for hid in all_hidden
  ]

  if len(layers_hidden) == 1:
    hidden_emb = layers_hidden[0]
    hidden_size = hidden_shapes[0][2]
  elif aggregation_fn == "concat":
    hidden_emb = tf.concat(layers_hidden, 2)
    hidden_size = sum([hidden_shapes[i][2] for i in layers_to_use])
  elif aggregation_fn == "average":
    hidden_size = hidden_shapes[0][2]
    assert all([shape[2] == hidden_size for shape in hidden_shapes
               ]), hidden_shapes
    hidden_emb = tf.add_n(layers_hidden) / len(layers_hidden)
  elif aggregation_fn == "attention":
    hidden_size = hidden_shapes[0][2]
    mixing_weights = tf.get_variable(
        name + "/mixing/weights", [len(layers_hidden)],
        initializer=tf.zeros_initializer())
    mixing_scores = tf.nn.softmax(mixing_weights)
    hidden_emb = tf.tensordot(
        tf.stack(layers_hidden, axis=-1), mixing_scores, [[-1], [0]])
  else:
    raise ValueError("Unrecognized aggregation function %s." % aggregation_fn)

  return hidden_emb, hidden_size


def shared_qry_encoder_v2(qry_input_ids, qry_input_mask, is_training,
                          use_one_hot_embeddings, bert_config, qa_config):
  """Embed query into a BOW and shared dense representation."""
  qry_model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=qry_input_ids,
      input_mask=qry_input_mask,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope="bert")
  qry_seq_emb, _ = get_bert_embeddings(
      qry_model, [qa_config.qry_num_layers - 2], "concat", name="qry")
  word_emb_table = qry_model.get_embedding_table()

  return qry_seq_emb, word_emb_table


def shared_qry_encoder(qry_input_ids, qry_input_mask, is_training,
                       use_one_hot_embeddings, bert_config, qa_config):
  """Embed query into a BOW and shared dense representation."""
  dropout = qa_config.dropout if is_training else 0.0
  attention_mask = modeling.create_attention_mask_from_input_mask(
      qry_input_ids, qry_input_mask)

  # Word embeddings.
  with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
      # Perform embedding lookup on the word ids.
      qry_word_emb, word_emb_table = modeling.embedding_lookup(
          input_ids=qry_input_ids,
          vocab_size=bert_config.vocab_size,
          embedding_size=bert_config.hidden_size,
          initializer_range=bert_config.initializer_range,
          word_embedding_name="word_embeddings",
          use_one_hot_embeddings=use_one_hot_embeddings)

  # question shared encoder
  with tf.variable_scope("qry/encoder", reuse=tf.AUTO_REUSE):
    qry_seq_emb = modeling.transformer_model(
        input_tensor=qry_word_emb,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=qa_config.qry_num_layers - 1,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False)

  return qry_seq_emb, word_emb_table


def layer_qry_encoder(qry_seq_emb,
                      qry_input_ids,
                      qry_input_mask,
                      is_training,
                      bert_config,
                      qa_config,
                      suffix="",
                      project=True):
  """Embed query into start and end vectors for dense retrieval for a hop."""
  dropout = qa_config.dropout if is_training else 0.0
  attention_mask = modeling.create_attention_mask_from_input_mask(
      qry_input_ids, qry_input_mask)

  # question start
  with tf.variable_scope("qry/start" + suffix, reuse=tf.AUTO_REUSE):
    qry_start_seq_emb = modeling.transformer_model(
        input_tensor=qry_seq_emb,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False)
    qry_start_emb = tf.squeeze(qry_start_seq_emb[:, 0:1, :], axis=1)

  # question end
  with tf.variable_scope("qry/end" + suffix, reuse=tf.AUTO_REUSE):
    qry_end_seq_emb = modeling.transformer_model(
        input_tensor=qry_seq_emb,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False)
    qry_end_emb = tf.squeeze(qry_end_seq_emb[:, 0:1, :], axis=1)

  if project and qa_config.projection_dim is not None:
    with tf.variable_scope("projection"):
      qry_start_emb = contrib_layers.fully_connected(
          qry_start_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=tf.AUTO_REUSE,
          scope="qry_projection")
      qry_end_emb = contrib_layers.fully_connected(
          qry_end_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=tf.AUTO_REUSE,
          scope="qry_projection")

  return qry_start_emb, qry_end_emb


def one_hop(qry_input_ids,
            qry_input_mask,
            qry_entity_ids,
            entity_ids,
            entity_mask,
            ent2ment_ind,
            ent2ment_val,
            ment2ent_map,
            is_training,
            use_one_hot_embeddings,
            bert_config,
            qa_config,
            mips_config,
            answer_mentions=None):
  """One hop of propagation from input to output entities."""
  # for question BOW embedding
  with tf.variable_scope("qry/bow"):
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())

  qry_seq_emb, word_emb_table = shared_qry_encoder(qry_input_ids,
                                                   qry_input_mask, is_training,
                                                   use_one_hot_embeddings,
                                                   bert_config, qa_config)

  qry_start_emb, qry_end_emb = layer_qry_encoder(qry_seq_emb, qry_input_ids,
                                                 qry_input_mask, is_training,
                                                 bert_config, qa_config)

  with tf.device("/cpu:0"):
    # mips search.
    tf_db, mips_search_fn = search_utils.create_mips_searcher(
        mips_config.ckpt_var_name, mips_config.ckpt_path,
        mips_config.num_neighbors)

  batch_size = tf.shape(qry_entity_ids)[0]
  batch_entities = tf.SparseTensor(
      indices=tf.concat([
          tf.cast(tf.expand_dims(tf.range(batch_size), 1), tf.int64),
          tf.cast(tf.expand_dims(qry_entity_ids, 1), tf.int64)
      ],
                        axis=1),
      values=tf.ones_like(qry_entity_ids, tf.float32),
      dense_shape=[batch_size, qa_config.num_entities])
  ret_entities, ret_mentions, dense_mention_vec, sp_mention_vec = follow(
      batch_entities, qry_start_emb, qry_end_emb, entity_ids, entity_mask,
      ent2ment_ind, ent2ment_val, ment2ent_map, word_emb_table, word_weights,
      mips_search_fn, tf_db, bert_config.hidden_size, mips_config, qa_config,
      is_training, answer_mentions)

  return ret_entities, ret_mentions, dense_mention_vec, sp_mention_vec


def multi_hop(qry_input_ids,
              qry_input_mask,
              qry_entity_ids,
              entity_ids,
              entity_mask,
              ent2ment_ind,
              ent2ment_val,
              ment2ent_map,
              is_training,
              use_one_hot_embeddings,
              bert_config,
              qa_config,
              mips_config,
              num_hops,
              exclude_set=None,
              bridge_mentions=None,
              answer_mentions=None):
  """Two hops of propagation from input to output entities."""
  # for question BOW embedding
  with tf.variable_scope("qry/bow"):
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())

  qry_seq_emb, word_emb_table = shared_qry_encoder_v2(qry_input_ids,
                                                      qry_input_mask,
                                                      is_training,
                                                      use_one_hot_embeddings,
                                                      bert_config, qa_config)

  batch_size = tf.shape(qry_input_ids)[0]
  if isinstance(qry_entity_ids, tf.SparseTensor):
    # Multiple entities per question. We need to re-score.
    with tf.name_scope("entity_linking"):
      batch_entity_emb = entity_emb(
          tf.cast(qry_entity_ids.values, tf.int64), entity_ids, entity_mask,
          word_emb_table, word_weights)
      qry_el_emb, _ = layer_qry_encoder(
          qry_seq_emb,
          qry_input_ids,
          qry_input_mask,
          is_training,
          bert_config,
          qa_config,
          suffix="_el",
          project=False)
      batch_qry_el_emb = tf.gather(qry_el_emb, qry_entity_ids.indices[:, 0])
      batch_entity_el_scs = tf.reduce_sum(batch_qry_el_emb * batch_entity_emb,
                                          -1)
      batch_entities_nosc = tf.SparseTensor(
          indices=tf.concat([
              qry_entity_ids.indices[:, 0:1],
              tf.cast(tf.expand_dims(qry_entity_ids.values, 1), tf.int64)
          ],
                            axis=1),
          values=batch_entity_el_scs,
          dense_shape=[batch_size, qa_config.num_entities])
      batch_entities = tf.sparse.softmax(tf.sparse.reorder(batch_entities_nosc))
  else:
    batch_entities = tf.SparseTensor(
        indices=tf.concat([
            tf.cast(tf.expand_dims(tf.range(batch_size), 1), tf.int64),
            tf.cast(tf.expand_dims(qry_entity_ids, 1), tf.int64)
        ],
                          axis=1),
        values=tf.ones_like(qry_entity_ids, tf.float32),
        dense_shape=[batch_size, qa_config.num_entities])
  ensure_mentions = bridge_mentions

  with tf.device("/cpu:0"):
    # mips search.
    tf_db, mips_search_fn = search_utils.create_mips_searcher(
        mips_config.ckpt_var_name, mips_config.ckpt_path,
        mips_config.num_neighbors)

  layer_mentions, layer_entities = [], []
  layer_dense, layer_sp = [], []
  for hop in range(num_hops):
    with tf.name_scope("hop_%d" % hop):
      qry_start_emb, qry_end_emb = layer_qry_encoder(
          qry_seq_emb,
          qry_input_ids,
          qry_input_mask,
          is_training,
          bert_config,
          qa_config,
          suffix="_%d" % hop)

      ret_entities, ret_mentions, dense_mention_vec, sp_mention_vec = follow(
          batch_entities, qry_start_emb, qry_end_emb, entity_ids, entity_mask,
          ent2ment_ind, ent2ment_val, ment2ent_map, word_emb_table,
          word_weights, mips_search_fn, tf_db, bert_config.hidden_size,
          mips_config, qa_config, is_training, ensure_mentions)
      if exclude_set is not None:
        batch_ind = tf.expand_dims(tf.range(batch_size), 1)
        exclude_indices = tf.concat([
            tf.cast(batch_ind, tf.int64),
            tf.cast(tf.expand_dims(exclude_set, 1), tf.int64)
        ],
                                    axis=1)
        ret_entities = remove_from_sparse(ret_entities, exclude_indices)
      scaled_entities = tf.sparse.reorder(ret_entities)
      scaled_entities = tf.SparseTensor(
          indices=scaled_entities.indices,
          values=scaled_entities.values / qa_config.softmax_temperature,
          dense_shape=scaled_entities.dense_shape)
      batch_entities = tf.sparse.softmax(scaled_entities)
      ensure_mentions = answer_mentions
      layer_mentions.append(ret_mentions)
      layer_entities.append(ret_entities)
      layer_dense.append(dense_mention_vec)
      layer_sp.append(sp_mention_vec)

  return (layer_entities, layer_mentions, layer_dense, layer_sp,
          batch_entities_nosc, qry_seq_emb)


def create_onehop_model(bert_config, qa_config, mips_config, is_training,
                        features, ent2ment_ind, ent2ment_val, ment2ent_map,
                        entity_ids, entity_mask, use_one_hot_embeddings,
                        summary_obj):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  qry_input_ids = features["qry_input_ids"]
  qry_input_mask = features["qry_input_mask"]
  qry_entity_ids = features["qry_entity_id"]
  batch_size = tf.shape(qry_input_ids)[0]

  answer_mentions = None
  answer_entities = None
  if is_training:
    answer_mentions = features["answer_mentions"]
    answer_entities = features["answer_entities"]

  ret_entities, ret_mentions, _, _ = one_hop(
      qry_input_ids,
      qry_input_mask,
      qry_entity_ids,
      entity_ids,
      entity_mask,
      ent2ment_ind,
      ent2ment_val,
      ment2ent_map,
      is_training,
      use_one_hot_embeddings,
      bert_config,
      qa_config,
      mips_config,
      answer_mentions=answer_mentions)

  if qa_config.supervision == "mention":
    logits = ret_mentions
    answer_index = answer_mentions
  elif qa_config.supervision == "entity":
    logits = ret_entities
    uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
        logits.indices, logits.values, logits.dense_shape,
        qa_config.entity_score_aggregation_fn)
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             logits.dense_shape)
    answer_index = answer_entities
  else:
    raise ValueError("Unrecognized supervision type %s" % qa_config.supervision)
  logits = tf.sparse.to_dense(
      logits, default_value=DEFAULT_VALUE, validate_indices=False)
  if is_training:
    logits_mask = tf.cast(tf.greater(logits, DEFAULT_VALUE + 1), tf.float32)
  answer_preds = tf.argmax(logits, axis=1)

  total_loss = None
  if is_training:
    total_loss, _ = compute_loss_sparse(
        logits, answer_index, weights=logits_mask)
    if summary_obj is not None:
      nnz_logits = tf.reduce_sum(tf.cast(tf.greater(logits, 0.), tf.float32))
      summary_obj.scalar(
          "train/nnz_logits",
          tf.expand_dims(nnz_logits / tf.cast(batch_size, tf.float32), 0))
      correct = tf.reduce_sum(
          tf.cast(
              tf.equal(answer_preds, tf.cast(answer_index, tf.int64)),
              tf.int32))
      summary_obj.scalar("train/num_correct", tf.expand_dims(correct, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))

  predictions = {
      "qas_ids": qas_ids,
      "logits": logits,
      "predictions": answer_preds,
  }

  return total_loss, predictions


def create_twohopcascade_model(bert_config,
                               qa_config,
                               mips_config,
                               is_training,
                               features,
                               ent2ment_ind,
                               ent2ment_val,
                               ment2ent_map,
                               entity_ids,
                               entity_mask,
                               use_one_hot_embeddings,
                               summary_obj,
                               num_hops=2):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  rel_input_ids, rel_input_mask = [], []
  for ii in range(num_hops):
    rel_input_ids.append(features["rel_input_ids_%d" % ii])
    rel_input_mask.append(features["rel_input_mask_%d" % ii])
  qry_entity_ids = features["qry_entity_id"]
  batch_size = tf.shape(rel_input_ids)[0]

  answer_mentions = None
  answer_entities = None
  bridge_mentions = None
  if is_training:
    answer_mentions = features["answer_mentions"]
    answer_entities = features["answer_entities"]
    bridge_mentions = features["bridge_mentions"]

  # for question BOW embedding
  with tf.variable_scope("qry/bow"):
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())

  batch_size = tf.shape(qry_entity_ids)[0]
  batch_entities = tf.SparseTensor(
      indices=tf.concat([
          tf.cast(tf.expand_dims(tf.range(batch_size), 1), tf.int64),
          tf.cast(tf.expand_dims(qry_entity_ids, 1), tf.int64)
      ],
                        axis=1),
      values=tf.ones_like(qry_entity_ids, tf.float32),
      dense_shape=[batch_size, qa_config.num_entities])
  ensure_mentions = bridge_mentions

  with tf.device("/cpu:0"):
    # mips search.
    tf_db, mips_search_fn = search_utils.create_mips_searcher(
        mips_config.ckpt_var_name, mips_config.ckpt_path,
        mips_config.num_neighbors)

  layer_mentions, layer_entities = [], []
  layer_dense, layer_sp = [], []
  for hop in range(num_hops):
    with tf.name_scope("hop_%d" % hop):
      qry_seq_emb, word_emb_table = shared_qry_encoder(rel_input_ids[hop],
                                                       rel_input_mask[hop],
                                                       is_training,
                                                       use_one_hot_embeddings,
                                                       bert_config, qa_config)
      qry_start_emb, qry_end_emb = layer_qry_encoder(qry_seq_emb,
                                                     rel_input_ids[hop],
                                                     rel_input_mask[hop],
                                                     is_training, bert_config,
                                                     qa_config)

      ret_entities, ret_mentions, dense_mention_vec, sp_mention_vec = follow(
          batch_entities, qry_start_emb, qry_end_emb, entity_ids, entity_mask,
          ent2ment_ind, ent2ment_val, ment2ent_map, word_emb_table,
          word_weights, mips_search_fn, tf_db, bert_config.hidden_size,
          mips_config, qa_config, is_training, ensure_mentions)
      scaled_entities = tf.sparse.reorder(ret_entities)
      scaled_entities = tf.SparseTensor(
          indices=scaled_entities.indices,
          values=scaled_entities.values / qa_config.softmax_temperature,
          dense_shape=scaled_entities.dense_shape)
      batch_entities = tf.sparse.softmax(scaled_entities)
      ensure_mentions = answer_mentions
      layer_mentions.append(ret_mentions)
      layer_entities.append(ret_entities)
      layer_dense.append(dense_mention_vec)
      layer_sp.append(sp_mention_vec)

  if qa_config.supervision == "mention":
    logits = layer_mentions[-1]
    answer_index = answer_mentions
  elif qa_config.supervision == "entity":
    logits = layer_entities[-1]
    uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
        logits.indices, logits.values, logits.dense_shape,
        qa_config.entity_score_aggregation_fn)
    uniq_entity_scs /= qa_config.softmax_temperature
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             logits.dense_shape)
    answer_index = answer_entities
  else:
    raise ValueError("Unrecognized supervision type %s" % qa_config.supervision)
  logits = tf.sparse.to_dense(
      logits, default_value=DEFAULT_VALUE, validate_indices=False)
  logits_mask = tf.cast(tf.greater(logits, DEFAULT_VALUE + 1), tf.float32)
  answer_preds = tf.argmax(logits, axis=1)

  total_loss = None
  if is_training:
    total_loss, _ = compute_loss_sparse(
        logits, answer_index, weights=logits_mask)
    if summary_obj is not None:
      nnz_logits = tf.reduce_sum(tf.cast(tf.greater(logits, 0.), tf.float32))
      summary_obj.scalar(
          "train/nnz_logits",
          tf.expand_dims(nnz_logits / tf.cast(batch_size, tf.float32), 0))
      correct = tf.reduce_sum(
          tf.cast(
              tf.equal(answer_preds, tf.cast(answer_index, tf.int64)),
              tf.int32))
      summary_obj.scalar("train/num_correct", tf.expand_dims(correct, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))

  predictions = {
      "qas_ids": qas_ids,
      "logits": logits,
      "predictions": answer_preds,
  }

  return total_loss, predictions


def create_twohop_model(bert_config,
                        qa_config,
                        mips_config,
                        is_training,
                        features,
                        ent2ment_ind,
                        ent2ment_val,
                        ment2ent_map,
                        entity_ids,
                        entity_mask,
                        use_one_hot_embeddings,
                        summary_obj,
                        num_hops=2):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  qry_input_ids = features["qry_input_ids"]
  qry_input_mask = features["qry_input_mask"]
  qry_entity_ids = features["qry_entity_id"]
  batch_size = tf.shape(qry_input_ids)[0]

  answer_mentions = None
  answer_entities = None
  bridge_mentions = None
  if is_training:
    answer_mentions = features["answer_mentions"]
    answer_entities = features["answer_entities"]
    bridge_mentions = features["bridge_mentions"]

  layer_entities, layer_mentions, _, _, _, _ = multi_hop(
      qry_input_ids,
      qry_input_mask,
      qry_entity_ids,
      entity_ids,
      entity_mask,
      ent2ment_ind,
      ent2ment_val,
      ment2ent_map,
      is_training,
      use_one_hot_embeddings,
      bert_config,
      qa_config,
      mips_config,
      num_hops=num_hops,
      bridge_mentions=bridge_mentions,
      answer_mentions=answer_mentions)

  if qa_config.supervision == "mention":
    logits = layer_mentions[-1]
    answer_index = answer_mentions
  elif qa_config.supervision == "entity":
    logits = layer_entities[-1]
    uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
        logits.indices, logits.values, logits.dense_shape,
        qa_config.entity_score_aggregation_fn)
    uniq_entity_scs /= qa_config.softmax_temperature
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             logits.dense_shape)
    answer_index = answer_entities
  else:
    raise ValueError("Unrecognized supervision type %s" % qa_config.supervision)
  logits = tf.sparse.to_dense(
      logits, default_value=DEFAULT_VALUE, validate_indices=False)
  logits_mask = tf.cast(tf.greater(logits, DEFAULT_VALUE + 1), tf.float32)
  answer_preds = tf.argmax(logits, axis=1)

  total_loss = None
  if is_training:
    total_loss, _ = compute_loss_sparse(
        logits, answer_index, weights=logits_mask)
    if summary_obj is not None:
      nnz_logits = tf.reduce_sum(tf.cast(tf.greater(logits, 0.), tf.float32))
      summary_obj.scalar(
          "train/nnz_logits",
          tf.expand_dims(nnz_logits / tf.cast(batch_size, tf.float32), 0))
      correct = tf.reduce_sum(
          tf.cast(
              tf.equal(answer_preds, tf.cast(answer_index, tf.int64)),
              tf.int32))
      summary_obj.scalar("train/num_correct", tf.expand_dims(correct, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))

  predictions = {
      "qas_ids": qas_ids,
      "logits": logits,
      "predictions": answer_preds,
  }

  return total_loss, predictions


def create_wikimovie_model(bert_config,
                           qa_config,
                           mips_config,
                           is_training,
                           features,
                           ent2ment_ind,
                           ent2ment_val,
                           ment2ent_map,
                           entity_ids,
                           entity_mask,
                           use_one_hot_embeddings,
                           summary_obj,
                           num_hops=1):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  qry_input_ids = features["qry_input_ids"]
  qry_input_mask = features["qry_input_mask"]
  qry_entity_ids = features["qry_entity_id"]
  batch_size = tf.shape(qry_input_ids)[0]

  answer_entities = None
  bridge_entities = None
  layer_answers = []
  if is_training:
    answer_entities = features["answer_entities"]
    answer_index = tf.sparse_to_dense(
        sparse_indices=tf.concat([
            answer_entities.indices[:, 0:1],
            tf.cast(tf.expand_dims(answer_entities.values, 1), tf.int64)
        ],
                                 axis=1),
        output_shape=[batch_size, qa_config.num_entities],
        sparse_values=tf.ones_like(answer_entities.values, dtype=tf.float32),
        validate_indices=False)
    answer_index = answer_index / tf.reduce_sum(
        answer_index, axis=1, keepdims=True)
    layer_answers.insert(0, answer_index)
    if num_hops > 1 and "bridge_entities_0" in features:
      for ii in range(num_hops - 2, -1, -1):
        bridge_entities = features["bridge_entities_%d" % ii]
        bridge_index = tf.sparse_to_dense(
            sparse_indices=tf.concat([
                bridge_entities.indices[:, 0:1],
                tf.cast(tf.expand_dims(bridge_entities.values, 1), tf.int64)
            ],
                                     axis=1),
            output_shape=[batch_size, qa_config.num_entities],
            sparse_values=tf.ones_like(
                bridge_entities.values, dtype=tf.float32),
            validate_indices=False)
        bridge_index = bridge_index / tf.reduce_sum(
            bridge_index, axis=1, keepdims=True)
        layer_answers.insert(0, bridge_index)

  layer_entities, layer_mentions, layer_dense, layer_sp, _, _ = multi_hop(
      qry_input_ids,
      qry_input_mask,
      qry_entity_ids,
      entity_ids,
      entity_mask,
      ent2ment_ind,
      ent2ment_val,
      ment2ent_map,
      is_training,
      use_one_hot_embeddings,
      bert_config,
      qa_config,
      mips_config,
      num_hops=num_hops,
      exclude_set=qry_entity_ids)

  if is_training:
    nrows = qa_config.train_batch_size
  else:
    nrows = qa_config.predict_batch_size

  def _to_ragged(sp_tensor):
    r_ind = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.indices[:, 1],
        nrows=nrows)
    r_val = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.values,
        nrows=nrows)
    return r_ind, r_val

  total_loss = []
  predictions = {"qas_ids": qas_ids}
  for hop in range(len(layer_mentions)):
    if qa_config.light:
      if hop == len(layer_mentions) - 1:
        logits = layer_entities[hop]
        uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
            logits.indices, logits.values, logits.dense_shape,
            qa_config.entity_score_aggregation_fn)
        logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                                 logits.dense_shape)
        logits = tf.sparse.to_dense(
            logits, default_value=DEFAULT_VALUE, validate_indices=False)
        answer_preds = tf.argmax(logits, axis=1)
    else:
      logits = layer_entities[hop]
      uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
          logits.indices, logits.values, logits.dense_shape,
          qa_config.entity_score_aggregation_fn)
      uniq_entity_scs /= qa_config.softmax_temperature
      logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                               logits.dense_shape)
      logits_mask = tf.sparse.to_dense(
          tf.SparseTensor(uniq_entity_ids, tf.ones_like(uniq_entity_scs),
                          logits.dense_shape),
          validate_indices=False)
      logits = tf.sparse.to_dense(
          logits, default_value=DEFAULT_VALUE, validate_indices=False)
      answer_preds = tf.argmax(logits, axis=1)

      sparse_ind, sparse_val = _to_ragged(layer_sp[hop])
      dense_ind, dense_val = _to_ragged(layer_dense[hop])
      mention_ind, mention_val = _to_ragged(layer_mentions[hop])
      entity_ind, entity_val = _to_ragged(layer_entities[hop])
      predictions.update({
          "sparse_%d" % hop: sparse_ind.to_tensor(default_value=-1),
          "dense_%d" % hop: dense_ind.to_tensor(default_value=-1),
          "mention_%d" % hop: mention_ind.to_tensor(default_value=-1),
          "entity_%d" % hop: entity_ind.to_tensor(default_value=-1),
          "sparse_scores_%d" % hop: sparse_val.to_tensor(default_value=-1),
          "dense_scores_%d" % hop: dense_val.to_tensor(default_value=-1),
          "mention_scores_%d" % hop: mention_val.to_tensor(default_value=-1),
          "entity_scores_%d" % hop: entity_val.to_tensor(default_value=-1),
      })

    if is_training:
      layer_loss, weights = compute_loss(
          logits, layer_answers[hop], weights=logits_mask)
      total_loss.append(layer_loss)
      if summary_obj is not None:
        nnz_logits = tf.reduce_sum(
            tf.cast(tf.greater(logits, DEFAULT_VALUE), tf.float32))
        summary_obj.scalar(
            "train/nnz_logits_%d" % hop,
            tf.expand_dims(nnz_logits / tf.cast(batch_size, tf.float32), 0))
        pred_answer_index = batch_gather(layer_answers[hop],
                                         tf.expand_dims(answer_preds, 1))
        correct = tf.reduce_sum(
            tf.cast(tf.greater(pred_answer_index, 0.), tf.int32))
        summary_obj.scalar("train/num_correct_%d" % hop,
                           tf.expand_dims(correct, 0))
        summary_obj.scalar("train/total_loss_%d" % hop,
                           tf.expand_dims(layer_loss, 0))
        summary_obj.scalar("train/ans_in_ret_%d" % hop,
                           tf.expand_dims(tf.reduce_sum(weights), 0))

  if total_loss and qa_config.intermediate_loss:
    total_loss = tf.add_n(total_loss)
  elif total_loss:
    total_loss = total_loss[-1]
  else:
    total_loss = None

  predictions.update({
      "logits": logits,
      "predictions": answer_preds,
  })

  return total_loss, predictions


def batch_multiply(sp_tensor, dense_vec):
  """Batch multiply a vector with a sparse tensor."""
  batch_indices = sp_tensor.indices[:, 0]
  batch_vec = tf.gather(dense_vec, batch_indices)
  return tf.SparseTensor(
      indices=sp_tensor.indices,
      values=sp_tensor.values * batch_vec,
      dense_shape=sp_tensor.dense_shape)


def create_hotpotqa_model(bert_config,
                          qa_config,
                          mips_config,
                          is_training,
                          features,
                          ent2ment_ind,
                          ent2ment_val,
                          ment2ent_map,
                          entity_ids,
                          entity_mask,
                          use_one_hot_embeddings,
                          summary_obj,
                          num_hops=2):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  qry_input_ids = features["qry_input_ids"]
  qry_input_mask = features["qry_input_mask"]
  batch_size = tf.shape(qry_input_ids)[0]
  qry_entity_ids = features["qry_entity_id"]
  if not isinstance(qry_entity_ids, tf.SparseTensor):
    # This assumes batch_size == 1.
    num_ents = features["num_entities"][0]
    qry_entity_ids = tf.SparseTensor(
        indices=tf.concat([
            tf.zeros((num_ents, 1), dtype=tf.int64),
            tf.expand_dims(tf.range(num_ents, dtype=tf.int64), 1)
        ],
                          axis=1),
        values=qry_entity_ids[0, :num_ents],
        dense_shape=[1, qa_config.num_entities])

  answer_entities = None
  if is_training:
    answer_entities = features["answer_entities"]
    answer_index = tf.SparseTensor(
        indices=tf.concat([
            answer_entities.indices[:, 0:1],
            tf.cast(tf.expand_dims(answer_entities.values, 1), tf.int64)
        ],
                          axis=1),
        values=tf.ones_like(answer_entities.values, dtype=tf.float32),
        dense_shape=[batch_size, qa_config.num_entities])

  layer_entities, _, _, _, el, qry_seq_emb = multi_hop(
      qry_input_ids,
      qry_input_mask,
      qry_entity_ids,
      entity_ids,
      entity_mask,
      ent2ment_ind,
      ent2ment_val,
      ment2ent_map,
      is_training,
      use_one_hot_embeddings,
      bert_config,
      qa_config,
      mips_config,
      num_hops=num_hops,
      exclude_set=None)
  layer_entities = [el] + layer_entities

  # Compute weights for each layer.
  with tf.name_scope("classifier"):
    qry_emb, _ = layer_qry_encoder(
        qry_seq_emb,
        qry_input_ids,
        qry_input_mask,
        is_training,
        bert_config,
        qa_config,
        suffix="_cl")
    output_weights = tf.get_variable(
        "cl_weights", [qa_config.projection_dim,
                       len(layer_entities)],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "cl_bias", [len(layer_entities)], initializer=tf.zeros_initializer())
    logits = tf.matmul(qry_emb, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)

  if is_training:
    nrows = qa_config.train_batch_size
  else:
    nrows = qa_config.predict_batch_size

  def _to_ragged(sp_tensor):
    r_ind = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.indices[:, 1],
        nrows=nrows)
    r_val = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.values,
        nrows=nrows)
    return r_ind, r_val

  def _layer_softmax(entities):
    uniq_entity_ids, uniq_entity_scs = aggregate_sparse_indices(
        entities.indices, entities.values, entities.dense_shape,
        qa_config.entity_score_aggregation_fn)
    uniq_entity_scs /= qa_config.softmax_temperature
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             entities.dense_shape)
    return tf.sparse.softmax(tf.sparse.reorder(logits))

  predictions = {"qas_ids": qas_ids}
  layer_entities_weighted = []
  for i, layer_entity in enumerate(layer_entities):
    ent_ind, ent_val = _to_ragged(layer_entity)
    predictions.update({
        "layer_%d_ent" % i: ent_ind.to_tensor(default_value=-1),
        "layer_%d_scs" % i: ent_val.to_tensor(default_value=-1),
    })
    layer_entities_weighted.append(
        batch_multiply(_layer_softmax(layer_entity), probabilities[:, i]))

  probs = tf.sparse.add(layer_entities_weighted[0], layer_entities_weighted[1])
  for i in range(2, len(layer_entities_weighted)):
    probs = tf.sparse.add(probs, layer_entities_weighted[i])

  probs_dense = tf.sparse.to_dense(
      probs, default_value=DEFAULT_VALUE, validate_indices=False)
  answer_preds = tf.argmax(probs_dense, axis=1)
  top_vals, top_idx = tf.nn.top_k(probs_dense, k=100, sorted=True)

  total_loss = None
  if is_training:
    sp_loss = compute_loss_from_sptensors(probs, answer_index)
    total_loss = tf.reduce_sum(sp_loss.values) / tf.cast(batch_size, tf.float32)
    num_answers_ret = tf.shape(sp_loss.values)[0]
    if summary_obj is not None:
      for i in range(len(layer_entities)):
        num_ents = tf.cast(tf.shape(layer_entities[i].indices)[0],
                           tf.float32) / tf.cast(batch_size, tf.float32)
        summary_obj.scalar("train/layer_weight_%d" % i,
                           tf.reduce_mean(probabilities[:, i], keepdims=True))
        summary_obj.scalar("train/num_entities_%d" % i,
                           tf.expand_dims(num_ents, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))
      summary_obj.scalar("train/ans_in_ret", tf.expand_dims(num_answers_ret, 0))
      summary_obj.scalar("train/total_prob_mass",
                         tf.reduce_sum(probs.values, keepdims=True))

  predictions.update({
      "layer_probs": probabilities,
      "top_vals": top_vals,
      "top_idx": top_idx,
      "predictions": answer_preds,
  })

  return total_loss, predictions


def create_hotpot_answer_model(bert_config, is_training, input_ids, input_mask,
                               segment_ids, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  cls_weights = tf.get_variable(
      "cls/squad/cls_weights", [3, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  cls_bias = tf.get_variable(
      "cls/squad/cls_bias", [3], initializer=tf.zeros_initializer())

  final_hidden_matrix = final_hidden[:, 0, :]
  qtype_logits = tf.matmul(final_hidden_matrix, cls_weights, transpose_b=True)
  qtype_logits = tf.nn.bias_add(qtype_logits, cls_bias)

  sp_weights = tf.get_variable(
      "cls/squad/sp_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  sp_bias = tf.get_variable(
      "cls/squad/sp_bias", [1], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  sp_logits = tf.matmul(final_hidden_matrix, sp_weights, transpose_b=True)
  sp_logits = tf.nn.bias_add(sp_logits, sp_bias)
  sp_logits = tf.reshape(
      tf.squeeze(sp_logits, axis=1), [batch_size, seq_length])

  return (start_logits, end_logits, qtype_logits, sp_logits)

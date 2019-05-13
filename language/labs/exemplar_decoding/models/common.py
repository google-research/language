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
"""Common utils for semiparametric generation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial  # pylint: disable=g-importing-member
import language.labs.exemplar_decoding.models.adam as adam
from language.labs.exemplar_decoding.models.hyperlstm import HyperLSTMCell
from language.labs.exemplar_decoding.utils.data import id2text
import tensorflow as tf


def dimension_value(dimension):
  if isinstance(dimension, tf.Dimension):
    return dimension.value


def get_rnn_cell(mode,
                 hps,
                 input_dim,
                 num_units,
                 num_layers=1,
                 dropout=0.,
                 mem_input=None,
                 use_beam=False,
                 cell_type="lstm",
                 reuse=None):
  """Construct RNN cells.

  Args:
    mode: train or eval. Keys from tf.estimator.ModeKeys.
    hps: Hyperparameters.
    input_dim: input size.
    num_units: hidden state size.
    num_layers: number of RNN layers.
    dropout: drop rate of RNN dropout.
    mem_input: mem_input
    use_beam: Use beam search or not.
    cell_type: [`lstm`, `hyperlsm`].
    reuse: Reuse option.

  Returns:
    RNN cell.
  """

  cells = []
  for i in xrange(num_layers):
    input_size = input_dim if i == 0 else num_units
    scale = 1.
    if cell_type == "lstm":
      cell = tf.contrib.rnn.LSTMCell(
          num_units=num_units,
          initializer=tf.orthogonal_initializer(scale),
          reuse=reuse)
    elif cell_type == "gru":
      cell = tf.contrib.rnn.GRUCell(
          num_units=num_units,
          kernel_initializer=tf.orthogonal_initializer(scale),
          reuse=reuse)
    elif cell_type == "hyper_lstm":
      cell = HyperLSTMCell(
          num_units=num_units,
          mem_input=mem_input,
          use_beam=use_beam,
          initializer=tf.orthogonal_initializer(scale),
          hps=hps,
          reuse=reuse)
    else:
      assert False
    if mode == tf.estimator.ModeKeys.TRAIN and dropout > 0.:
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell,
          input_size=input_size,
          output_keep_prob=1.0 - dropout,
          variational_recurrent=True,
          dtype=tf.float32)
    if hps.use_residual and num_layers > 1:
      cell = tf.nn.rnn_cell.ResidualWrapper(cell=cell)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells)
  return cell


def print_text(tf_sequences, vocab, use_bpe=False, predict_mode=False):
  """Print text."""
  def _print_separator():
    if not predict_mode:
      tf.logging.info("=" * 80)
  print_ops = [tf.py_func(_print_separator, [], [])]
  for name, tf_sequence, tf_length, convert2txt in tf_sequences:
    def _do_print(n, sequence, lengths, to_txt):
      if to_txt:
        s = sequence[0][:lengths[0]]
        output = id2text(s, vocab, use_bpe=use_bpe)
      else:
        output = " ".join(sequence[0])
      if not predict_mode:
        tf.logging.info("%s: %s", n, output)

    with tf.control_dependencies(print_ops):
      print_ops.append(tf.py_func(
          _do_print, [name, tf_sequence, tf_length, convert2txt], []))
  with tf.control_dependencies(print_ops):
    return tf.py_func(_print_separator, [], [])


def optimize_log_loss(decoder_tgt, decoder_outputs, weights, hps):
  """Optimize log loss.

  Args:
    decoder_tgt: gold outputs. [batch_size, len, vocab_size]
    decoder_outputs: predictions. [batch_size, len, vocab_size]
    weights: [batch_size, len] Mask.
    hps: hyperparams

  Returns:
    loss: Loss.
    train_op: Tensorflow Op for updating parameters.
  """

  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=decoder_tgt,
      logits=decoder_outputs.rnn_output)
  loss = tf.reduce_mean(loss * weights)
  # loss = tf.Print(loss, [loss])

  global_step = tf.train.get_global_step()
  values = [hps.learning_rate,
            hps.learning_rate / 5.,
            hps.learning_rate / 10.,
            hps.learning_rate / 25.,
            hps.learning_rate / 50.]
  boundaries = [hps.lr_schedule,
                int(hps.lr_schedule*1.5),
                hps.lr_schedule*2,
                int(hps.lr_schedule*2.5)]
  learning_rate = tf.train.piecewise_constant(
      global_step, boundaries, values)

  assert hps.trainer == "adam", "Only supporting Adam now."

  trainable_var_list = tf.trainable_variables()
  grads = tf.gradients(loss, trainable_var_list)
  gvs = list(zip(grads, trainable_var_list))

  grads = [g for g, _ in gvs]
  train_op = adam.adam(
      trainable_var_list,
      grads,
      learning_rate,
      partial(adam.warmup_constant),
      hps.total_steps,
      weight_decay=hps.weight_decay,
      max_grad_norm=hps.max_grad_norm,
      bias_l2=True)

  return loss, train_op


def sparse_map(ids, mask, vocab_size):
  """Create mapping from order 3 tensor to vocabulary for copying mechanism.

  This will return a SparseTensor of size (batch_size, vocab_size, num_cells).
  Each column of this sparse tensor will be non-zero at the indices specified by
  the `ids` in that cell.

  For rank 2 tensors expand dims along the axis=2 before passing.

  Args:
    ids: <tf.int32>[batch_size, num_cells, seq_len] Word ids of sequence in each
      cell.
    mask: <tf.float32>[batch_size, num_cells, seq_len] Mask for ids.
    vocab_size: Number of rows in resulting tensor.

  Returns:
    sp_tensor: tf.SparseTensor mapping num_cells to vocab_size.
  """
  batch_size, num_cells, seq_len = (
      tf.shape(ids)[0], tf.shape(ids)[1], tf.shape(ids)[2])

  nonzero = batch_size * num_cells * seq_len

  # Indices.
  ids_x = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=2),
      (1, num_cells, seq_len))
  sp_ids_x = tf.reshape(ids_x, (nonzero, 1))
  sp_ids_y = tf.reshape(ids, (nonzero, 1))
  ids_z = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(num_cells), axis=0), axis=2),
      (batch_size, 1, seq_len))
  sp_ids_z = tf.reshape(ids_z, (nonzero, 1))
  sp_ids = tf.concat([sp_ids_x, sp_ids_y, sp_ids_z], axis=1)

  # Values.
  lengths = tf.maximum(
      tf.reduce_sum(mask, axis=2, keepdims=True), 1.)
  values = mask / lengths
  sp_values = tf.reshape(values, (nonzero,))

  # SparseTensor.
  sp_tensor = tf.SparseTensor(
      tf.cast(sp_ids, tf.int64), sp_values,
      tf.cast(tf.stack([batch_size, vocab_size, num_cells]), tf.int64))

  return sp_tensor


def sparse_batched_matmul(x, y):
  """Batch multiply sparse tensor x with dense tensor y.

  Args:
    x: <tf.float32>[B, M, N] SparseTensor.
    y: <tf.float32>[B, N, K] DenseTensor.

  Returns:
    <tf.float32>[B, M, K] DenseTensor.
  """
  sp_indices = x.indices
  sp_values = x.values
  batch_size, num_row = x.dense_shape[0], x.dense_shape[1]
  num_col = tf.cast(tf.shape(y)[2], tf.int64)
  # Number of non-zero entries.
  num_nz = tf.cast(tf.shape(sp_indices)[0], tf.int64)

  # Fetch relevant values from y.
  # <tf.float32>[num_nz, num_col]
  lookup = tf.gather_nd(y, tf.stack(
      [sp_indices[:, 0], sp_indices[:, 2]], axis=1))

  # Reshape first two dimensions of x into a new SparseTensor of rank 2.
  # <tf.int32>[num_nz, 2]
  x_i = tf.stack([sp_indices[:, 0] * num_row + sp_indices[:, 1],
                  tf.cast(tf.range(num_nz), tf.int64)], axis=1)
  # <tf.float32>[batch_size * num_row, num_nz]
  sparse_2d = tf.SparseTensor(x_i, sp_values,
                              tf.stack([batch_size * num_row, num_nz], 0))

  # Multiply the new sparse tensor with the gathered values.
  # <tf.float32>[batch_size * num_row, num_col]
  dense_2d = tf.sparse_tensor_dense_matmul(sparse_2d, lookup)

  # Reshape back [batch_size, num_row, num_col]
  dense_3d = tf.reshape(dense_2d, tf.stack([batch_size, num_row, num_col], 0))

  return dense_3d


def sparse_tile_batch(tensor, multiplier):
  """Tile SparseTensor along batch dimension.

  This is a sparse equivalent for tf.contrib.seq2seq.tile_batch.

  NOTE: Only rank 3 tensors are supported currently.

  Args:
    tensor: A rank 3 SparseTensor.
    multiplier: Number of times to tile the batch dimension.

  Returns:
    tiled_tensor: A rank 3 SparseTensor representing the tiled dense tensor.
  """
  # Collect indices.
  indices = tensor.indices
  values = tensor.values
  batch_indices = indices[:, 0]
  nonbatch_indices = indices[:, 1:]
  batch_size = tensor.dense_shape[0]
  num_row, num_col = tensor.dense_shape[1], tensor.dense_shape[2]
  num_nonzero = tf.shape(indices)[0]

  # Tile non-batch dimensions. Every row is repeated `multiplier` times.
  # So given nonbatch_indices = [[0, 1], [2, 3]] and multiplier = 2,
  # new_nonbatch_indices = [[0, 1], [0, 1], [2, 3], [2, 3]].
  new_nonbatch_i = tf.reshape(
      tf.tile(tf.expand_dims(nonbatch_indices, 1), (1, multiplier, 1)),
      (num_nonzero * multiplier, -1))

  # Tile batch dimension. Every batch index `bi` is expanded into a range from
  # `bi * multiplier : (bi + 1) * multiplier`.
  new_batch_i = tf.reshape(
      multiplier * tf.expand_dims(batch_indices, 1) +
      tf.expand_dims(tf.range(multiplier, dtype=tf.int64), 0),
      (num_nonzero * multiplier, 1))

  # Concatenate to get new indices.
  new_indices = tf.concat([new_batch_i, new_nonbatch_i], 1)

  # Get new values. Tile these `multiplier` times.
  new_values = tf.reshape(
      tf.tile(tf.expand_dims(values, 1), (1, multiplier)),
      (num_nonzero * multiplier,))

  # Create new sparse tensor.
  tiled_tensor = tf.SparseTensor(new_indices,
                                 new_values,
                                 [batch_size * multiplier, num_row, num_col])

  return tiled_tensor

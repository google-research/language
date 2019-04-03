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
"""Model utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers

import tensorflow as tf


__all__ = [
    "GNMTAttentionMultiCell",
    "gnmt_residual_fn",
    "create_rnn_cell",
    "create_gnmt_rnn_cell",
    "build_unidirectional_rnn",
    "build_bidirectional_rnn",
    "make_sequences_compatible",
    "get_embeddings",
    "build_logits",
    "get_global_step",
]


def _single_cell(unit_type,
                 num_units,
                 forget_bias,
                 dropout,
                 mode,
                 residual_connection=False,
                 residual_fn=None,
                 trainable=True):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    single_cell = tf.contrib.rnn.LSTMCell(
        num_units,
        forget_bias=forget_bias,
        trainable=trainable)
  elif unit_type == "gru":
    single_cell = tf.contrib.rnn.GRUCell(
        num_units,
        trainable=trainable)
  elif unit_type == "layer_norm_lstm":
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True,
        trainable=trainable)
  elif unit_type == "nas":
    single_cell = tf.contrib.rnn.NASCell(
        num_units,
        trainable=trainable)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob).
  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))

  # Residual.
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

  return single_cell


def _cell_list(unit_type,
               num_units,
               num_layers,
               num_residual_layers,
               forget_bias,
               dropout,
               mode,
               single_cell_fn=None,
               residual_fn=None,
               trainable=True):
  """Create a list of RNN cells."""
  if single_cell_fn is None:
    single_cell_fn = _single_cell

  cell_list = []
  for i in range(num_layers):
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        residual_fn=residual_fn,
        trainable=trainable)
    cell_list.append(single_cell)

  return cell_list


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    scope = "multi_rnn_cell" if scope is None else scope
    with tf.variable_scope(scope):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)


def gnmt_residual_fn(inputs, outputs):
  """Residual function that handles different inputs and outputs inner dims.

  Args:
    inputs: A potentially nested structure of <tensor> [..., input_dim] that
      represents cell inputs.
    outputs: A potentially nested structure of <tensor> [..., output_dim] that
      represents cell outputs. Must have the same structure and number of
      dimensions as inputs and output_dim >= input_dim must hold.

  Returns:
    outputs + actual_inputs where actual_inputs are a nested structure of slices
    of the inputs along the last dimension up to the output_dim.
  """
  def split_input(inp, out):
    out_dim = out.get_shape().as_list()[-1]
    inp_dim = inp.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
  actual_inputs, _ = tf.contrib.framework.nest.map_structure(
      split_input, inputs, outputs)
  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())
  tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
  tf.contrib.framework.nest.map_structure(
      assert_shape_match, actual_inputs, outputs)
  return tf.contrib.framework.nest.map_structure(
      lambda inp, out: inp + out, actual_inputs, outputs)


def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    num_residual_layers,
                    forget_bias,
                    dropout,
                    mode,
                    attention_mechanism=None,
                    attention_num_heads=1,
                    attention_layer_size=None,
                    output_attention=False,
                    single_cell_fn=None,
                    trainable=True):
  """Returns an instance of an RNN cell.

  Args:
    unit_type: A string that specifies the type of the recurrent unit. Must be
      one of {"lstm", "gru", "lstm_norm", "nas"}.
    num_units: An integer for the numner of units per layer.
    num_layers: An integer for the number of recurrent layers.
    num_residual_layers: An integer for the number of residual layers.
    forget_bias: A float for the forget bias in LSTM cells.
    dropout: A float for the recurrent dropout rate.
    mode: TRAIN | EVAL | PREDICT
    attention_mechanism: An instance of tf.contrib.seq2seq.AttentionMechanism.
    attention_num_heads: An integer for the number of attention heads.
    attention_layer_size: Optional integer for the size of the attention layer.
    output_attention: A boolean indicating whether RNN cell outputs attention.
    single_cell_fn: A function for building a single RNN cell.
    trainable: A boolean indicating whether the cell weights are trainable.

  Returns:
    An RNNCell instance.
  """
  cell_list = _cell_list(
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      forget_bias=forget_bias,
      dropout=dropout,
      mode=mode,
      num_residual_layers=num_residual_layers,
      single_cell_fn=single_cell_fn,
      trainable=trainable)

  if len(cell_list) == 1:  # Single layer.
    cell = cell_list[0]
  else:  # Multiple layers.
    cell = tf.contrib.rnn.MultiRNNCell(cell_list)

  # Wrap with attention, if necessary.
  if attention_mechanism is not None:
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell, [attention_mechanism] * attention_num_heads,
        attention_layer_size=[attention_layer_size] * attention_num_heads,
        alignment_history=False,
        output_attention=output_attention,
        name="attention")

  return cell


def create_gnmt_rnn_cell(unit_type,
                         num_units,
                         num_layers,
                         num_residual_layers,
                         forget_bias,
                         dropout,
                         mode,
                         attention_mechanism,
                         attention_num_heads=1,
                         attention_layer_size=None,
                         output_attention=False,
                         single_cell_fn=None):
  """Returns an instance of an GNMT-style RNN cell.

  Args:
    unit_type: A string that specifies the type of the recurrent unit. Must be
      one of {"lstm", "gru", "lstm_norm", "nas"}.
    num_units: An integer for the numner of units per layer.
    num_layers: An integer for the number of recurrent layers.
    num_residual_layers: An integer for the number of residual layers.
    forget_bias: A float for the forget bias in LSTM cells.
    dropout: A float for the recurrent dropout rate.
    mode: TRAIN | EVAL | PREDICT
    attention_mechanism: An instance of tf.contrib.seq2seq.AttentionMechanism.
    attention_num_heads: An integer for the number of attention heads.
    attention_layer_size: Optional integer for the size of the attention layer.
    output_attention: A boolean indicating whether RNN cell outputs attention.
    single_cell_fn: A function for building a single RNN cell.

  Returns:
    An RNNCell instance.
  """
  cell_list = _cell_list(
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      forget_bias=forget_bias,
      dropout=dropout,
      mode=mode,
      num_residual_layers=num_residual_layers,
      single_cell_fn=single_cell_fn,
      residual_fn=gnmt_residual_fn)

  if attention_num_heads > 1:
    attention_mechanism = [attention_mechanism] * attention_num_heads
    attention_layer_size = [attention_layer_size] * attention_num_heads

  # Only wrap the bottom layer with the attention mechanism.
  attention_cell = cell_list.pop(0)
  attention_cell = tf.contrib.seq2seq.AttentionWrapper(
      attention_cell, attention_mechanism,
      attention_layer_size=attention_layer_size,
      alignment_history=False,
      output_attention=output_attention,
      name="gnmt_attention")

  cell = GNMTAttentionMultiCell(
      attention_cell, cell_list, use_new_attention=True)

  return cell


def build_unidirectional_rnn(sequences, length,
                             num_layers, num_residual_layers, num_units,
                             unit_type, forget_bias, dropout, mode,
                             trainable=True):
  """Builds sequences encoded with a unidirectional RNN."""
  cell = create_rnn_cell(
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=forget_bias,
      dropout=dropout,
      mode=mode,
      trainable=trainable)

  # Encode.
  outputs, final_state = tf.nn.dynamic_rnn(
      cell, sequences,
      sequence_length=length,
      dtype=tf.float32,
      time_major=False)

  return outputs, final_state


def build_bidirectional_rnn(sequences, length,
                            num_layers, num_residual_layers, num_units,
                            unit_type, forget_bias, dropout, mode,
                            trainable=True):
  """Builds sequences encoded with a bidirectional RNN."""
  # Create RNN cells.
  cells = {
      direction: create_rnn_cell(
          unit_type=unit_type,
          num_units=num_units,
          num_layers=num_layers,
          num_residual_layers=num_residual_layers,
          forget_bias=forget_bias,
          dropout=dropout,
          mode=mode,
          trainable=trainable)
      for direction in ["fw", "bw"]}

  # Encode.
  outputs, state = tf.nn.bidirectional_dynamic_rnn(
      cells["fw"], cells["bw"], sequences,
      sequence_length=length,
      dtype=tf.float32,
      time_major=False)

  # Concatenate bidirectional outputs.
  outputs = tf.concat(outputs, axis=2)

  return outputs, state


def make_sequences_compatible(seq1, seq2, op_type="max_pool"):
  """Preprocesses sequences to make them compatible for computing a loss.

  Args:
    seq1: <type> [batch_size, seq_len, ...].
    seq2: <type> [batch_size, seq_len, ...].
    op_type: String for the operation type. Must be one of
      {"avg_pool", "max_pool", "truncate"}.

  Returns:
    new_seq1, new_seq2: <type> [batch_size, ...] for preprocessed sequences.

  Raises:
    ValueError: if op_type is unknown.
  """
  if op_type not in {"avg_pool", "max_pool", "truncate"}:
    raise ValueError("Unknown op_type: %s" % op_type)

  if op_type == "avg_pool":
    new_seq1 = tf.reduce_mean(seq1, axis=1)
    new_seq2 = tf.reduce_mean(seq2, axis=1)
  elif op_type == "max_pool":
    new_seq1 = tf.reduce_max(seq1, axis=1)
    new_seq2 = tf.reduce_max(seq2, axis=1)
  elif op_type == "truncate":
    max_length = tf.minimum(tf.shape(seq1)[1], tf.shape(seq2)[1])
    new_seq1 = seq1[:, :max_length, :]
    new_seq2 = seq2[:, :max_length, :]

  return  new_seq1, new_seq2


def get_embeddings(modality, outer_scope, inner_scope="shared"):
  """Returns embeddings for the given modality stricly forcing reuse.

  Args:
    modality: A tensor2tensro.utils.modality.Modality object.
    outer_scope: A variable scope object used as the outer scope.
    inner_scope: A string used as the inner-most variable scope name.

  Returns:
    embeddings: <float32> [vocab_size, emb_size].
  """
  with tf.variable_scope(outer_scope, reuse=True):
    with tf.variable_scope(inner_scope, reuse=True):
      embeddings = modality._get_weights()  # pylint: disable=protected-access
  return embeddings


def build_logits(sequences, embeddings, vocab_size):
  """Builds logits for sequences and a given modality."""
  sequences_shape = common_layers.shape_list(sequences)
  sequences = tf.reshape(sequences, [-1, sequences_shape[-1]])
  # Compute logits.
  logits = tf.matmul(sequences, embeddings, transpose_b=True)
  logits_shape = sequences_shape[:-1] + [1, vocab_size]
  return tf.reshape(logits, logits_shape)


def get_global_step(hparams):
  """Returns the global optimization step."""
  step = tf.to_float(tf.train.get_or_create_global_step())
  multiplier = hparams.optimizer_multistep_accumulate_steps
  if not multiplier:
    return step
  return step / tf.to_float(multiplier)

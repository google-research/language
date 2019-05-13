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
"""`seq2seq` and `nn2seq` baseline models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple  # pylint: disable=g-importing-member
from language.labs.exemplar_decoding.models.attention import MyAttention
from language.labs.exemplar_decoding.models.common import get_rnn_cell
from language.labs.exemplar_decoding.models.common import sparse_map
from language.labs.exemplar_decoding.models.output_wrapper import OutputWrapper
from language.labs.exemplar_decoding.utils import tensor_utils
import language.labs.exemplar_decoding.utils.data as data
import tensorflow as tf


EncoderOutputs = namedtuple(
    "EncoderOutputs",
    ["embeddings", "att_context", "copy_context", "states"]
)

DecoderOutputs = namedtuple(
    "DecoderOutputs",
    ["decoder_outputs", "decoder_len"]
)


def _build_context(hps, encoder_outputs):
  """Compute feature representations for attention/copy.

  Args:
    hps: hyperparameters.
    encoder_outputs: outputs by the encoder RNN.

  Returns:
    Feature representation of [batch_size, seq_len, decoder_dim]
  """
  with tf.variable_scope("memory_context"):
    context = tf.layers.dense(
        encoder_outputs,
        units=hps.decoder_dim,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="memory_projector")
  return context


def encoder(features, mode, vocab, hps):
  """Model function.

  Atttention seq2seq model, augmented with an encoder
  over the targets of the nearest neighbors.

  Args:
    features: Dictionary of input Tensors.
    mode: train or eval. Keys from tf.estimator.ModeKeys.
    vocab: A list of strings of words in the vocabulary.
    hps: Hyperparams.

  Returns:
    Encoder outputs.
  """

  # [batch_size, src_len]
  src_inputs = features["src_inputs"]
  src_len = features["src_len"]

  with tf.variable_scope("embeddings"):
    embeddings = tf.get_variable(
        "embeddings",
        [vocab.size(), hps.emb_dim],
        dtype=tf.float32,
        initializer=tf.uniform_unit_scaling_initializer())

  # [batch_size, src_len, emb_dim]
  src_encoder_input_emb = tf.nn.embedding_lookup(embeddings, src_inputs)
  if mode == tf.estimator.ModeKeys.TRAIN and hps.emb_drop > 0.:
    src_encoder_input_emb = tf.nn.dropout(
        src_encoder_input_emb, keep_prob=1.0-hps.emb_drop)

  src_att_context, neighbor_att_context = None, None
  src_copy_context, neighbor_copy_context = None, None
  with tf.variable_scope("src_encoder"):

    # 2 * [batch_size, src_len, encoder_dim]
    src_encoder_outputs, src_encoder_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=get_rnn_cell(
            mode=mode, hps=hps,
            input_dim=hps.emb_dim,
            num_units=hps.encoder_dim,
            num_layers=hps.num_encoder_layers,
            dropout=hps.encoder_drop,
            cell_type=hps.rnn_cell),
        cell_bw=get_rnn_cell(
            mode=mode, hps=hps,
            input_dim=hps.emb_dim,
            num_units=hps.encoder_dim,
            num_layers=hps.num_encoder_layers,
            dropout=hps.encoder_drop,
            cell_type=hps.rnn_cell),
        inputs=src_encoder_input_emb,
        dtype=tf.float32,
        sequence_length=src_len)

    # [batch_size, src_len, 2*encoder_dim]
    src_encoder_outputs = tf.concat(src_encoder_outputs, 2)
    with tf.variable_scope("src_att_context"):
      src_att_context = _build_context(
          hps=hps,
          encoder_outputs=src_encoder_outputs)
    if hps.use_copy:
      with tf.variable_scope("src_copy_context"):
        src_copy_context = _build_context(
            hps=hps,
            encoder_outputs=src_encoder_outputs)

  if hps.model == "nn2seq":

    # [batch_size, neighbor_len]
    neighbor_inputs = features["neighbor_inputs"]
    neighbor_len = features["neighbor_len"]

    # [batch_size, neighbor_len, emb_dim]
    neighbor_input_emb = tf.nn.embedding_lookup(
        embeddings, neighbor_inputs)
    if mode == tf.estimator.ModeKeys.TRAIN and hps.emb_drop > 0.:
      neighbor_input_emb = tf.nn.dropout(
          neighbor_input_emb, keep_prob=1.0-hps.emb_drop)
    with tf.variable_scope("neighbor_encoder"):
      # 2 * [batch_size, neighbor_len, encoder_dim]
      neighbor_encoder_outputs, _ = \
          tf.nn.bidirectional_dynamic_rnn(
              cell_fw=get_rnn_cell(
                  mode=mode, hps=hps,
                  input_dim=hps.emb_dim,
                  num_units=hps.encoder_dim,
                  num_layers=1,
                  dropout=hps.encoder_drop,
                  cell_type=hps.rnn_cell),
              cell_bw=get_rnn_cell(
                  mode=mode, hps=hps,
                  input_dim=hps.emb_dim,
                  num_units=hps.encoder_dim,
                  num_layers=1,
                  dropout=hps.encoder_drop,
                  cell_type=hps.rnn_cell),
              inputs=neighbor_input_emb,
              dtype=tf.float32,
              sequence_length=neighbor_len)

      neighbor_encoder_outputs = tf.concat(neighbor_encoder_outputs, 2)
      with tf.variable_scope("neighbor_att_context"):
        neighbor_att_context = _build_context(
            hps=hps,
            encoder_outputs=neighbor_encoder_outputs)
      if hps.use_copy:
        with tf.variable_scope("neighbor_copy_context"):
          neighbor_copy_context = _build_context(
              hps=hps,
              encoder_outputs=neighbor_encoder_outputs)
  att_context, copy_context = None, None
  if hps.model == "nn2seq":
    att_context = tf.concat([src_att_context, neighbor_att_context], 1)
    if hps.use_copy:
      copy_context = tf.concat(
          [src_copy_context, neighbor_copy_context], 1)
  elif hps.model == "seq2seq":
    att_context = src_att_context
    if hps.use_copy:
      copy_context = src_copy_context
  else:
    assert False, "baseline `model` should be [`nn2seq`, `seq2seq`]."
  if hps.use_bridge:
    with tf.variable_scope("bridge"):
      out_dim = hps.num_decoder_layers * hps.decoder_dim
      if hps.rnn_cell == "lstm":
        fw_states, bw_states = src_encoder_states
        hs = tf.concat([fw_states[-1].h, bw_states[-1].h], axis=1)
        cs = tf.concat([fw_states[-1].c, bw_states[-1].c], axis=1)

        h_state = tf.layers.dense(
            hs, units=out_dim,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="h_layer")
        c_state = tf.layers.dense(
            cs, units=out_dim,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="c_layer")
      elif hps.rnn_cell == "gru":
        fw_states, bw_states = src_encoder_states
        hs = tf.concat([fw_states[-1], bw_states[-1]], axis=1)
        h_state = tf.layers.dense(
            hs, units=out_dim,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="h_layer")
        c_state = None
  else:
    h_state, c_state = None, None

  return EncoderOutputs(
      embeddings=embeddings,
      att_context=att_context,
      copy_context=copy_context,
      states=(h_state, c_state)
  )


def basic_decoder(features, mode, vocab, encoder_outputs, hps):
  """Decoder.

  Args:
    features: Dictionary of input Tensors.
    mode: train or eval. Keys from tf.estimator.ModeKeys.
    vocab: A list of strings of words in the vocabulary.
    encoder_outputs: output tensors from the encoder
    hps: Hyperparams.

  Returns:
    Decoder outputs
  """

  # [batch_size, src_len]

  embeddings = encoder_outputs.embeddings
  src_len, src_inputs = features["src_len"], features["src_inputs"]
  batch_size = tensor_utils.shape(src_len, 0)

  src_mask = tf.sequence_mask(src_len, tf.shape(src_inputs)[1])

  if hps.model == "nn2seq":
    neighbor_len, neighbor_inputs \
        = features["neighbor_len"], features["neighbor_inputs"]
    neighbor_mask = tf.sequence_mask(
        neighbor_len, tf.shape(neighbor_inputs)[1])
    inputs = tf.concat([src_inputs, neighbor_inputs], 1)
    mask = tf.concat([src_mask, neighbor_mask], axis=1)
    # lens = tf.shape(mask)[1] * tf.ones([batch_size], tf.int32)
  else:
    inputs = src_inputs
    # lens = src_len
    mask = src_mask

  sparse_inputs = None

  float_mask = tf.cast(mask, dtype=tf.float32)
  if hps.use_copy:
    sparse_inputs = sparse_map(
        tf.expand_dims(inputs, axis=2),
        tf.expand_dims(float_mask, axis=2),
        vocab.size())

  # [batch_size, dec_len]
  decoder_inputs = features["decoder_inputs"]

  # [batch_size, dec_len, emb_dim]
  decoder_input_emb = tf.nn.embedding_lookup(embeddings, decoder_inputs)
  if mode == tf.estimator.ModeKeys.TRAIN and hps.emb_drop > 0.:
    decoder_input_emb = tf.nn.dropout(
        decoder_input_emb, keep_prob=1.0-hps.emb_drop)

  def _decode(cell, helper):
    """Decode function.

    Args:
      cell: rnn cell
      helper: a helper instance from tf.contrib.seq2seq.

    Returns:
      decoded outputs and lengths.
    """
    with tf.variable_scope("decoder"):
      initial_state = cell.zero_state(batch_size, tf.float32)
      if hps.use_bridge:
        h_state, c_state = encoder_outputs.states
        if hps.rnn_cell == "lstm":
          initial_cell_state = tf.contrib.rnn.LSTMStateTuple(h_state, c_state)
        elif hps.rnn_cell == "gru":
          initial_cell_state = h_state
        initial_state = initial_state.clone(cell_state=(initial_cell_state,))

      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=initial_state)
    with tf.variable_scope("dynamic_decode", reuse=tf.AUTO_REUSE):
      decoder_outputs, _, decoder_len = tf.contrib.seq2seq.dynamic_decode(
          decoder=decoder,
          maximum_iterations=hps.max_dec_steps)
    return decoder_outputs, decoder_len

  att_context = encoder_outputs.att_context
  with tf.variable_scope("source_attention"):
    if hps.att_type == "luong":
      attention = tf.contrib.seq2seq.LuongAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None)
    elif hps.att_type == "bahdanau":
      attention = tf.contrib.seq2seq.BahdanauAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None)
    elif hps.att_type == "my":
      attention = MyAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None,
          mask=mask)

  with tf.variable_scope("rnn_decoder"):
    decoder_cell = get_rnn_cell(
        mode=mode, hps=hps,
        input_dim=hps.decoder_dim+hps.emb_dim,
        num_units=hps.decoder_dim,
        num_layers=hps.num_decoder_layers,
        dropout=hps.decoder_drop,
        cell_type=hps.rnn_cell)
  with tf.variable_scope("attention_wrapper"):
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention,
        attention_layer_size=hps.decoder_dim,
        alignment_history=hps.use_copy)

  with tf.variable_scope("output_projection"):
    weights = tf.transpose(embeddings) if hps.tie_embedding else None
    hidden_dim = hps.emb_dim if hps.tie_embedding else hps.decoder_dim
    decoder_cell = OutputWrapper(
        decoder_cell,
        hps.num_mlp_layers,
        hidden_dim=hidden_dim,
        output_size=vocab.size() if hps.tie_embedding else hps.output_size,
        weights=weights,
        dropout=hps.emb_drop,
        use_copy=hps.use_copy,
        encoder_emb=encoder_outputs.copy_context,
        sparse_inputs=sparse_inputs,
        mask=float_mask,
        hps=hps)

  if mode == tf.estimator.ModeKeys.TRAIN:
    if hps.sampling_probability > 0.:
      helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
          inputs=decoder_input_emb,
          sequence_length=features["decoder_len"],
          embedding=embeddings,
          sampling_probability=hps.sampling_probability)
    else:
      helper = tf.contrib.seq2seq.TrainingHelper(
          decoder_input_emb, features["decoder_len"])
    decoder_outputs, _ = _decode(decoder_cell, helper=helper)
    return DecoderOutputs(
        decoder_outputs=decoder_outputs,
        decoder_len=features["decoder_len"]), None

  # Used to compute loss.
  teacher_helper = tf.contrib.seq2seq.TrainingHelper(
      decoder_input_emb, features["decoder_len"])
  teacher_decoder_outputs, _ = _decode(decoder_cell, helper=teacher_helper)

  helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      embedding=embeddings,
      start_tokens=tf.fill(
          [batch_size], vocab.word2id(data.START_DECODING)),
      end_token=vocab.word2id(data.STOP_DECODING))
  decoder_outputs, decoder_len = _decode(decoder_cell, helper=helper)
  return (
      DecoderOutputs(
          decoder_outputs=decoder_outputs,
          decoder_len=decoder_len),
      DecoderOutputs(
          decoder_outputs=teacher_decoder_outputs,
          decoder_len=features["decoder_len"])
  )


def beam_decoder(features, mode, vocab, encoder_outputs, hps):
  """Beam search decoder.

  Args:
    features: Dictionary of input Tensors.
    mode: train or eval. Keys from tf.estimator.ModeKeys.
    vocab: A list of strings of words in the vocabulary.
    encoder_outputs: output tensors from the encoder
    hps: Hyperparams.

  Returns:
    Decoder outputs
  """
  assert mode is not tf.estimator.ModeKeys.TRAIN, "Not using beam in training."

  batch_size = tensor_utils.shape(features["src_len"], 0)
  embeddings = encoder_outputs.embeddings
  src_len, src_inputs = features["src_len"], features["src_inputs"]
  src_mask = tf.sequence_mask(src_len, tf.shape(src_inputs)[1])

  if hps.model == "nn2seq":
    neighbor_len, neighbor_inputs \
        = features["neighbor_len"], features["neighbor_inputs"]
    neighbor_mask = tf.sequence_mask(
        neighbor_len, tf.shape(neighbor_inputs)[1])
    inputs = tf.concat([src_inputs, neighbor_inputs], 1)
    mask = tf.concat([src_mask, neighbor_mask], axis=1)
  else:
    inputs = src_inputs
    mask = src_mask

  tiled_mask = tf.contrib.seq2seq.tile_batch(mask, multiplier=hps.beam_width)
  inputs = tf.contrib.seq2seq.tile_batch(inputs, multiplier=hps.beam_width)
  sparse_inputs = None
  if hps.use_copy:
    sparse_inputs = sparse_map(
        tf.expand_dims(inputs, axis=2),
        tf.expand_dims(tf.cast(tiled_mask, dtype=tf.float32), axis=2),
        vocab.size())

  def _beam_decode(cell):
    """Beam decode."""
    with tf.variable_scope("beam_decoder"):
      initial_state = cell.zero_state(
          batch_size=batch_size*hps.beam_width, dtype=tf.float32)
      if hps.use_bridge:
        h_state, c_state = encoder_outputs.states
        if hps.rnn_cell == "lstm":
          h_state = tf.contrib.seq2seq.tile_batch(
              h_state, multiplier=hps.beam_width)
          c_state = tf.contrib.seq2seq.tile_batch(
              c_state, multiplier=hps.beam_width)
          initial_cell_state = tf.contrib.rnn.LSTMStateTuple(h_state, c_state)
        elif hps.rnn_cell == "gru":
          h_state = tf.contrib.seq2seq.tile_batch(
              h_state, multiplier=hps.beam_width)
          initial_cell_state = h_state
        initial_state = initial_state.clone(cell_state=(initial_cell_state,))

      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          cell=cell,
          embedding=embeddings,
          start_tokens=tf.fill(
              [batch_size], vocab.word2id(data.START_DECODING)),
          end_token=vocab.word2id(data.STOP_DECODING),
          initial_state=initial_state,
          beam_width=hps.beam_width,
          length_penalty_weight=hps.length_norm)
    with tf.variable_scope("dynamic_decode", reuse=tf.AUTO_REUSE):
      decoder_outputs, _, decoder_len = tf.contrib.seq2seq.dynamic_decode(
          decoder=decoder,
          maximum_iterations=hps.max_dec_steps)
    return decoder_outputs, decoder_len

  # [batch_size*beam_width, src_len, encoder_dim]
  att_context = tf.contrib.seq2seq.tile_batch(
      encoder_outputs.att_context, multiplier=hps.beam_width)
  copy_context = None
  if hps.use_copy:
    copy_context = tf.contrib.seq2seq.tile_batch(
        encoder_outputs.copy_context, multiplier=hps.beam_width)
  with tf.variable_scope("source_attention", reuse=tf.AUTO_REUSE):
    if hps.att_type == "luong":
      attention = tf.contrib.seq2seq.LuongAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None)
    elif hps.att_type == "bahdanau":
      attention = tf.contrib.seq2seq.BahdanauAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None)
    elif hps.att_type == "my":
      attention = MyAttention(
          num_units=hps.decoder_dim,
          memory=att_context,
          memory_sequence_length=None,
          mask=tiled_mask)

  with tf.variable_scope("rnn_decoder", reuse=tf.AUTO_REUSE):
    decoder_cell = get_rnn_cell(
        mode=mode, hps=hps,
        input_dim=hps.decoder_dim+hps.emb_dim,
        num_units=hps.decoder_dim,
        num_layers=hps.num_decoder_layers,
        dropout=hps.decoder_drop,
        cell_type=hps.rnn_cell)
  with tf.variable_scope("attention_wrapper", reuse=tf.AUTO_REUSE):
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention,
        attention_layer_size=hps.decoder_dim,
        alignment_history=hps.use_copy)

  with tf.variable_scope("output_projection", reuse=tf.AUTO_REUSE):
    weights = tf.transpose(embeddings) if hps.tie_embedding else None
    hidden_dim = hps.emb_dim if hps.tie_embedding else hps.decoder_dim
    decoder_cell = OutputWrapper(
        decoder_cell,
        hps.num_mlp_layers,
        hidden_dim=hidden_dim,
        output_size=vocab.size() if hps.tie_embedding else hps.output_size,
        weights=weights,
        dropout=hps.emb_drop,
        use_copy=hps.use_copy,
        encoder_emb=copy_context,
        sparse_inputs=sparse_inputs,
        mask=tf.cast(tiled_mask, dtype=tf.float32),
        hps=hps,
        reuse=tf.AUTO_REUSE)

  decoder_outputs, decoder_len = _beam_decode(decoder_cell)

  return DecoderOutputs(
      decoder_outputs=decoder_outputs,
      decoder_len=decoder_len)

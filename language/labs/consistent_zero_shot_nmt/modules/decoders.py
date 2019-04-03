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
"""A collection of decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.labs.consistent_zero_shot_nmt.modules import attention_mechanisms
from language.labs.consistent_zero_shot_nmt.modules import attention_wrappers
from language.labs.consistent_zero_shot_nmt.modules import base
from language.labs.consistent_zero_shot_nmt.modules import helpers
from language.labs.consistent_zero_shot_nmt.utils import common_utils as U
from language.labs.consistent_zero_shot_nmt.utils import model_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers

import tensorflow as tf


__all__ = [
    "BasicRNNDecoder",
    "AttentiveRNNDecoder",
    "get",
]


class BasicRNNDecoder(base.AbstractNMTModule):
  """Basic multi-layer recurrent neural network decoder."""

  def __init__(self, name="BasicRNNDecoder"):
    super(BasicRNNDecoder, self).__init__(name=name)

  def _build(self, embeddings, inputs, inputs_length, hiddens, hiddens_length,
             enc_state, mode, hparams, decoder_hparams=None):
    if decoder_hparams is None:
      decoder_hparams = tf.contrib.training.HParams(
          auxiliary=False)

    batch_size = common_layers.shape_list(hiddens)[0]

    # Build RNN cell.
    rnn_cell = self._build_rnn_cell(
        embeddings=embeddings,
        sequences=hiddens,
        sequences_length=hiddens_length,
        mode=mode,
        hparams=hparams)

    # Build initial state.
    initial_state = self._build_init_state(
        batch_size=batch_size,
        enc_state=enc_state,
        rnn_cell=rnn_cell,
        mode=mode,
        hparams=hparams)

    # Build helper.
    helper = self._build_helper(
        batch_size=batch_size,
        embeddings=embeddings,
        inputs=inputs,
        inputs_length=inputs_length,
        mode=mode,
        hparams=hparams,
        decoder_hparams=decoder_hparams)

    # Build decoder.
    decoder = self._build_decoder(
        helper=helper,
        rnn_cell=rnn_cell,
        initial_state=initial_state,
        mode=mode,
        hparams=hparams)

    return decoder

  def _build_attention(self, memory, memory_sequence_length, mode, hparams):
    """Builds attention mechanism for attending over the hiddens."""
    return attention_mechanisms.get(
        attention_type=hparams.attention_mechanism,
        num_units=hparams.attention_layer_size,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        scope="attention_over_hiddens")

  def _build_rnn_cell(self, embeddings, sequences, sequences_length,
                      mode, hparams):
    """Builds RNN cell for decoding."""
    del embeddings  # Unused.

    # Build attention.
    attention_mechanism = self._build_attention(
        memory=sequences,
        memory_sequence_length=sequences_length,
        mode=mode,
        hparams=hparams)

    # Choose attention architecture.
    if hparams.attention_gnmt:
      create_rnn_cell_fn = model_utils.create_gnmt_rnn_cell
    else:
      create_rnn_cell_fn = model_utils.create_rnn_cell

    # Create base RNN cell with attention.
    rnn_cell = create_rnn_cell_fn(
        attention_mechanism=attention_mechanism,
        attention_layer_size=hparams.attention_layer_size,
        output_attention=(hparams.output_attention == 1),
        unit_type=hparams.rnn_unit_type,
        num_units=hparams.hidden_size,
        num_layers=hparams.dec_num_layers,
        num_residual_layers=hparams.dec_num_residual_layers,
        forget_bias=hparams.rnn_forget_bias,
        dropout=hparams.dropout,
        mode=mode)

    return rnn_cell

  def _build_init_state(self, batch_size, enc_state, rnn_cell, mode, hparams):
    """Builds initial states for the given RNN cells."""
    # Build init state.
    init_state = rnn_cell.zero_state(batch_size, tf.float32)

    if hparams.pass_hidden_state:
      # Non-GNMT RNN cell returns AttentionWrappedState.
      if isinstance(init_state, tf.contrib.seq2seq.AttentionWrapperState):
        init_state = init_state.clone(cell_state=enc_state)
      # GNMT RNN cell returns a tuple state.
      elif isinstance(init_state, tuple):
        init_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(init_state, enc_state))
      else:
        ValueError("RNN cell returns zero states of unknown type: %s"
                   % str(type(init_state)))

    return init_state

  def _build_helper(self, batch_size, embeddings, inputs, inputs_length,
                    mode, hparams, decoder_hparams):
    """Builds a helper instance for BasicDecoder."""
    # Auxiliary decoding mode at training time.
    if decoder_hparams.auxiliary:
      start_tokens = tf.fill([batch_size], text_encoder.PAD_ID)
      # helper = helpers.FixedContinuousEmbeddingHelper(
      #     embedding=embeddings,
      #     start_tokens=start_tokens,
      #     end_token=text_encoder.EOS_ID,
      #     num_steps=hparams.aux_decode_length)
      helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
          embedding=embeddings,
          start_tokens=start_tokens,
          end_token=text_encoder.EOS_ID,
          softmax_temperature=None)
    # Continuous decoding.
    elif hparams.decoder_continuous:
      # Scheduled mixing.
      if mode == tf.estimator.ModeKeys.TRAIN and hparams.scheduled_training:
        helper = helpers.ScheduledContinuousEmbeddingTrainingHelper(
            inputs=inputs,
            sequence_length=inputs_length,
            mixing_concentration=hparams.scheduled_mixing_concentration)
      # Pure continuous decoding (hard to train!).
      elif mode == tf.estimator.ModeKeys.TRAIN:
        helper = helpers.ContinuousEmbeddingTrainingHelper(
            inputs=inputs,
            sequence_length=inputs_length)
      # EVAL and PREDICT expect teacher forcing behavior.
      else:
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=inputs,
            sequence_length=inputs_length)
    # Standard decoding.
    else:
      # Scheduled sampling.
      if mode == tf.estimator.ModeKeys.TRAIN and hparams.scheduled_training:
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=inputs,
            sequence_length=inputs_length,
            embedding=embeddings,
            sampling_probability=hparams.scheduled_sampling_probability)
      # Teacher forcing (also for EVAL and PREDICT).
      else:
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=inputs,
            sequence_length=inputs_length)
    return helper

  def _build_decoder(self, helper, rnn_cell, initial_state, mode, hparams):
    """Builds a decoder instance."""
    del mode  # Unused.
    del hparams  # Unused.
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=rnn_cell,
        helper=helper,
        initial_state=initial_state)
    return decoder


class AttentiveRNNDecoder(BasicRNNDecoder):
  """Decodes by attending over the embedding vocabulary."""

  def __init__(self, name="AttentiveRNNDecoder"):
    super(AttentiveRNNDecoder, self).__init__(name=name)

  def _wrap_with_attention(self, cell, memory, hparams):
    # Get decoding attention mechanism.
    with tf.variable_scope("decoding_attention", reuse=tf.AUTO_REUSE):
      attention_mechanism = attention_wrappers.FixedMemoryLuongAttention(
          num_units=hparams.hidden_size,
          memory=memory,
          memory_sequence_length=None,
          scale=True,
          name="FixedMemoryAttention")

    # Wrap RNN cell.
    wrapped_cell = attention_wrappers.FixedMemoryAttentionWrapper(
        cell, attention_mechanism,
        attention_layer_size=None,
        cell_input_fn=lambda inputs, attention: inputs,
        output_attention=True)

    return wrapped_cell

  def _build_rnn_cell(self, embeddings, sequences, sequences_length,
                      mode, hparams):
    """Builds attentive RNN cell for decoding."""
    # Build RNN cell.
    rnn_cell = super(AttentiveRNNDecoder, self)._build_rnn_cell(
        embeddings=embeddings,
        sequences=sequences,
        sequences_length=sequences_length,
        mode=mode,
        hparams=hparams)

    # Wrap cell with attention over the target embedding vocabulary.
    memory = tf.expand_dims(embeddings, 0)
    rnn_cell = self._wrap_with_attention(
        cell=rnn_cell,
        memory=memory,
        hparams=hparams)

    return rnn_cell

  def _build_init_state(self, batch_size, enc_state, rnn_cell, mode, hparams):
    """Builds initial states for the given RNN cells."""
    # Build init state.
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    inner_state = init_state.cell_state

    if hparams.pass_hidden_state:
      # Non-GNMT RNN cell returns AttentionWrappedState.
      if isinstance(inner_state, tf.contrib.seq2seq.AttentionWrapperState):
        init_state = init_state.clone(
            cell_state=inner_state.clone(cell_state=enc_state))
      # GNMT RNN cell returns a tuple state.
      elif isinstance(init_state.cell_state, tuple):
        init_state = init_state.clone(cell_state=tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(inner_state, enc_state)))
      else:
        ValueError("RNN cell returns zero states of unknown type: %s"
                   % str(type(init_state)))

    return init_state


def get(decoder_type):
  """Returns a decoder instance of the specified type."""
  if decoder_type == U.DEC_BASIC:
    decoder = BasicRNNDecoder()
  elif decoder_type == U.DEC_ATTENTIVE:
    decoder = AttentiveRNNDecoder()
  else:
    raise ValueError("Unknown decoder type: %s. The type must be one of %s."
                     % (decoder_type, str(U.DEC_TYPES)))
  return decoder

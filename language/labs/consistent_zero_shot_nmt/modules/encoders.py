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
"""A collection of encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from language.labs.consistent_zero_shot_nmt.modules import base
from language.labs.consistent_zero_shot_nmt.utils import common_utils as U
from language.labs.consistent_zero_shot_nmt.utils import model_utils
import tensorflow as tf


__all__ = [
    "BaseRNNEncoder",
    "UniRNNEncoder",
    "BiRNNEncoder",
    "GNMTEncoder",
    "get",
]


class EncoderOutput(
    collections.namedtuple("EncoderOutput", ("outputs", "final_state"))):
  """Final outputs returned by the encoder.

  Args:
    outputs: <float32> [batch_size, seq_len, hidden_size] for encoded sequences.
    final_state: Nested structure of <float32> [batch_size, hidden_size] tensors
      for the final hidden state.
  """
  pass


class BaseRNNEncoder(base.AbstractNMTModule):
  """Base class for RNN encoders."""

  def __init__(self, name="BaseRNNEncoder"):
    super(BaseRNNEncoder, self).__init__(name=name)

  def _build(self, inputs, inputs_length, mode, hparams):

    # Build encoded sequences.
    # TODO(alshedivat): remove this spurious variable scope eventually.
    with tf.variable_scope("encoder"):
      outputs, final_state = self._build_encoded_sequences(
          sequences=inputs,
          length=inputs_length,
          mode=mode,
          hparams=hparams)

    return EncoderOutput(outputs=outputs, final_state=final_state)

  @abc.abstractmethod
  def _build_encoded_sequences(self, sequences, length, mode, hparams):
    """Must be implemented by a subclass."""
    raise NotImplementedError


class UniRNNEncoder(BaseRNNEncoder):
  """Encoder based on a unidirectional (multi-layer) recurrent neural network.
  """

  def __init__(self, name="UniRNNEncoder"):
    super(UniRNNEncoder, self).__init__(name=name)

  def _build_encoded_sequences(self, sequences, length, mode, hparams):
    return model_utils.build_unidirectional_rnn(
        sequences=sequences,
        length=length,
        num_layers=hparams.enc_num_layers,
        num_residual_layers=hparams.enc_num_residual_layers,
        num_units=hparams.hidden_size,
        unit_type=hparams.rnn_unit_type,
        forget_bias=hparams.rnn_forget_bias,
        dropout=hparams.dropout,
        mode=mode)


class BiRNNEncoder(BaseRNNEncoder):
  """Encoder based on a bidirectional (multi-layer) recurrent neural network.
  """

  def __init__(self, name="BiRNNEncoder"):
    super(BiRNNEncoder, self).__init__(name=name)

  def _build_encoded_sequences(self, sequences, length, mode, hparams):
    """Build sequences of encoded vectors."""
    outputs, (final_states_fw, final_states_bw) = (
        model_utils.build_bidirectional_rnn(
            sequences=sequences,
            length=length,
            unit_type=hparams.rnn_unit_type,
            num_layers=hparams.enc_num_layers,
            num_residual_layers=hparams.enc_num_residual_layers,
            num_units=hparams.hidden_size,
            forget_bias=hparams.rnn_forget_bias,
            dropout=hparams.dropout,
            mode=mode))

    # Concatenate forward and backward states.
    if hparams.enc_num_layers > 1:
      concat_state = []
      for layer_id in range(hparams.enc_num_layers):
        concat_state.append(final_states_fw[layer_id])  # forward
        concat_state.append(final_states_bw[layer_id])  # backward
      final_state = tuple(concat_state)

    return outputs, final_state


class GNMTEncoder(BaseRNNEncoder):
  """Encoder based on multi-layer RNN with GNMT attention architecture."""

  def __init__(self, name="GNMTEncoder"):
    super(GNMTEncoder, self).__init__(name=name)

  def _build_encoded_sequences(self, sequences, length, mode, hparams):
    num_bi_layers = 1
    num_uni_layers = hparams.enc_num_layers - num_bi_layers

    # Build bidirectional layers.
    bi_outputs, bi_state = model_utils.build_bidirectional_rnn(
        sequences=sequences,
        length=length,
        num_layers=num_bi_layers,
        num_residual_layers=0,  # no residual connection
        num_units=hparams.hidden_size,
        unit_type=hparams.rnn_unit_type,
        forget_bias=hparams.rnn_forget_bias,
        dropout=hparams.dropout,
        mode=mode)

    # Build unidirectional layers.
    outputs, state = model_utils.build_unidirectional_rnn(
        sequences=bi_outputs,
        length=length,
        num_layers=num_uni_layers,
        num_residual_layers=hparams.enc_num_residual_layers,
        num_units=hparams.hidden_size,
        unit_type=hparams.rnn_unit_type,
        forget_bias=hparams.rnn_forget_bias,
        dropout=hparams.dropout,
        mode=mode)

    # Pass all encoder states to the decoder except the first in the bi-layer.
    state = (bi_state[1],) + ((state,) if num_uni_layers == 1 else state)

    return outputs, state


def get(encoder_type):
  """Returns an encoder instance of the specified type."""
  if encoder_type == U.ENC_UNI:
    encoder = UniRNNEncoder()
  elif encoder_type == U.ENC_BI:
    encoder = BiRNNEncoder()
  elif encoder_type == U.ENC_GNMT:
    encoder = GNMTEncoder()
  else:
    raise ValueError("Unknown encoder type: %s. The type must be one of %s."
                     % (encoder_type, str(U.ENC_TYPES)))
  return encoder

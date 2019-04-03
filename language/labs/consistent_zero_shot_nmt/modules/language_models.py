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
"""A collection of language models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from language.labs.consistent_zero_shot_nmt.modules import base
from language.labs.consistent_zero_shot_nmt.utils import common_utils as U
from language.labs.consistent_zero_shot_nmt.utils import model_utils


__all__ = [
    "BaseLanguageModel",
    "Left2RightLanguageModel",
    "get",
]


class BaseLanguageModel(base.AbstractNMTModule):
  """Base class for language models."""

  def __init__(self, name="BaseLanguageModel"):
    super(BaseLanguageModel, self).__init__(name=name)

  def _build(self, inputs, inputs_length, mode, hparams, trainable=False):
    # Build encoded sequences.
    outputs, _ = self._build_encoded_sequences(
        sequences=inputs,
        length=inputs_length,
        mode=mode,
        hparams=hparams,
        trainable=trainable)
    return outputs

  @abc.abstractmethod
  def _build_encoded_sequences(self, sequences, length, mode, hparams,
                               trainable=True):
    raise NotImplementedError("AbstractMethod")


class Left2RightLanguageModel(BaseLanguageModel):
  """Language model that factorizes over the sequences left-to-right."""

  def __init__(self, name="Left2RigthLanguageModel"):
    super(Left2RightLanguageModel, self).__init__(name=name)

  def _build_encoded_sequences(self, sequences, length, mode, hparams,
                               trainable=True):
    return model_utils.build_unidirectional_rnn(
        sequences=sequences,
        length=length,
        num_layers=hparams.lm_num_layers,
        num_residual_layers=hparams.lm_num_residual_layers,
        num_units=hparams.hidden_size,
        unit_type=hparams.rnn_unit_type,
        forget_bias=hparams.rnn_forget_bias,
        dropout=hparams.dropout,
        mode=mode,
        trainable=trainable)


def get(language_model_type):
  """Returns a language model instance of the specified type."""
  if language_model_type == "left2right":
    language_model = Left2RightLanguageModel()
  else:
    raise ValueError("Unknown LM type: %s. The type must be one of %s."
                     % (language_model_type, str(U.LM_TYPES)))
  return language_model

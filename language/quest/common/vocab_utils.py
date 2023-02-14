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
"""Utilities for dealing with T5 sentence piece model."""

from sentencepiece import SentencePieceProcessor


class T5SpmWrapper(object):
  """Wrapper for T5 sentence piece model."""

  def __init__(self, sp_model):
    self.sp = SentencePieceProcessor()
    self.sp.Load(sp_model)

  def tokenize(self, input_string):
    """Return list of tokens for input."""
    return self.sp.EncodeAsPieces(input_string)

  def truncate(self, input_string, num_tokens):
    """Truncate input to be `num_tokens`."""
    tokens = self.sp.EncodeAsPieces(input_string)
    truncated_tokens = tokens[:num_tokens]
    return self.sp.DecodePieces(truncated_tokens)

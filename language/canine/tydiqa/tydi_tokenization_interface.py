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
"""A fork of BERT's tokenizer that tracks byte offsets.

This module does not depend on TensorFlow and should be re-usable within your
favorite ML/DL framework.
"""

import abc

import typing



# For CANINE, all `SubTokens` will always be codepoints.
TokenizedIdsWithOffsets = typing.NamedTuple("TokenizedIdsWithOffsets", [
    ("subtokens", List[int]),
    ("start_bytes", List[int]),
    ("limit_bytes", List[int]),
    ("offsets_to_subtoken", Dict[int, int]),
])


class TokenizerWithOffsets(metaclass=abc.ABCMeta):
  """Tokenizes an input string into a sequence of IDs and tracks byte offsets."""

  def tokenize(self, text):
    """Tokenizes a piece of `text` and returns a list of vocab IDs."""
    wordpieces, _, _, _ = self.tokenize_with_offsets(text)
    return wordpieces

  @abc.abstractmethod
  def tokenize_with_offsets(self, text):
    """Tokenizes a piece of `text` and returns IDs and offset information."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_passage_marker(self, i):
    """Returns a marker for the start of passage `i`.

    Args:
      i: The passage index.  This method will be called to create the strings
        that will be inserted into the article text before the start of each
        passage. Because it will be inserted into the text, it will need to be
        handled during tokenization, often (but not necessarily) by recognizing
        it as a single unit that can be looked up from the "special markers"
        section of a vocabulary (much like a [CLS] or [SEP] marker).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_vocab_id(self, key, default = None):
    raise NotImplementedError()

  # Only needed by debugging code.
  @abc.abstractmethod
  def id_to_string(self, i):
    raise NotImplementedError()

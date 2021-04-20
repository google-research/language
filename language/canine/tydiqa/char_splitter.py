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
"""Character-level preprocessing for CANINE."""



from language.canine import special_codepoints
from language.canine.tydiqa import data
from language.canine.tydiqa import tydi_tokenization_interface


class CharacterSplitter(tydi_tokenization_interface.TokenizerWithOffsets):
  """A character splitter that preserves byte offsets.

  This implements the `TokenizerWithOffsets` interface to demonstrate how to
  retrofit legacy tokenization code with character splitting.
  """

  def __init__(self):
    next_private_use = max(special_codepoints.SPECIAL_CODEPOINTS) + 1

    # Special symbols that should be given pseudo-codepoints from the "private
    # use area", following those in the standard CANINE SPECIAL_CODEPOINTS set.
    tydiqa_symbols = ["[Q]"]

    # Creates a mapping for looking up the IDs of special symbols.
    self._special_codepoints: Dict[Text, int] = {}
    for codepoint, name in special_codepoints.SPECIAL_CODEPOINTS.items():
      self._special_codepoints[name] = codepoint
    for codepoint, name in enumerate(tydiqa_symbols, next_private_use):
      self._special_codepoints[name] = codepoint
    next_private_use += len(tydiqa_symbols)

    self._passage_0_codepoint = next_private_use

    # Creates a mapping for looking up the string forms of special symbol IDs.
    self._special_codepoint_strings: Dict[int, Text] = {
        codepoint: name for name, codepoint in self._special_codepoints.items()
    }

  def tokenize_with_offsets(
      self, text):
    result = tydi_tokenization_interface.TokenizedIdsWithOffsets([], [], [], {})
    byte_index = 0
    for char_index, c in enumerate(text):
      result.subtokens.append(ord(c))
      result.start_bytes.append(byte_index)
      for _ in range(data.byte_len(c)):
        result.offsets_to_subtoken[byte_index] = char_index
        byte_index += 1
      result.limit_bytes.append(byte_index - 1)
    return result

  def get_passage_marker(self, i):
    return chr(self._passage_0_codepoint + i)

  def get_vocab_id(self, key, default = None):
    """Gets the vocab id of `key`."""
    if key in self._special_codepoints:
      return self._special_codepoints[key]
    try:
      return ord(key)
    except TypeError:
      raise ValueError(f"invalid vocab key: '{key}'")

  def id_to_string(self, i):
    if i in self._special_codepoint_strings:
      return self._special_codepoint_strings[i]
    try:
      return chr(i)
    except TypeError:
      raise ValueError(f"invalid id: {i}")

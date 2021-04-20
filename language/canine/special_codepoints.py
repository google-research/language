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
"""Constants defining canonical codepoints for special, pseudo-characters."""



# Padding is always index zero. This means that the NULL character is
# technically not embeddable. This seems fine according to all reasonable
# interpretations of the NULL character as a past-end-of-string marker.
PAD = 0

CLS = 0xE000
SEP = 0xE001
BOS = 0xE002
MASK = 0xE003
RESERVED = 0xE004

# Maps special codepoints to human-readable names.
SPECIAL_CODEPOINTS: Dict[int, Text] = {
    # Special symbols are represented using codepoints values that are valid,
    # but designated as "Private Use", meaning that they will never by assigned
    # characters by the Unicode Consortium, and are thus safe for use here.
    #
    # NOTE: Do *NOT* add any sort of [UNK_CHAR] here. They are explicitly
    # excluded and should fail with a hard error.
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

# Maps special codepoint human-readable names to their codepoint values.
SPECIAL_CODEPOINTS_BY_NAME: Dict[Text, int] = {
    name: codepoint for codepoint, name in SPECIAL_CODEPOINTS.items()
}

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
"""The vocabulary class."""

from GroundedScan import vocabulary


class RelationVocabulary(vocabulary.Vocabulary):
  """The vocabulary class that includes location prepositions.

  Allow random sampling of nonce-vocabulary or setting user-defined words. See
  https://github.com/LauraRuis/groundedSCAN/blob/master/GroundedScan/vocabulary.py
  for more details.
  """

  LOCATION_PREPS = frozenset({
      "next to", "east of", "north of", "west of", "south of", "north east of",
      "north west of", "south west of", "south east of"
  })

  def __init__(self, intransitive_verbs, transitive_verbs, adverbs, nouns,
               color_adjectives, size_adjectives, location_preps):
    all_words = list(intransitive_verbs.keys()) + list(
        transitive_verbs.keys()) + list(adverbs.keys()) + list(
            nouns.keys()) + list(color_adjectives.keys()) + list(
                size_adjectives.keys()) + list(location_preps.keys())
    all_unique_words = set(all_words)
    assert len(all_words) == len(
        all_unique_words
    ), "Overlapping vocabulary (the same string used twice)."

    super().__init__(intransitive_verbs, transitive_verbs, adverbs, nouns,
                     color_adjectives, size_adjectives)
    self._location_preps = location_preps
    self._translation_table.update(self._location_preps)
    self._translate_to = {
        semantic_word: word
        for word, semantic_word in self._translation_table.items()
    }

  def get_location_preps(self):
    return list(self._location_preps.keys()).copy()

  @classmethod
  def initialize(cls, intransitive_verbs, transitive_verbs, adverbs, nouns,
                 color_adjectives, size_adjectives, location_preps):
    """Initialize vocabulary."""
    intransitive_verbs = cls.bind_words_to_meanings(
        intransitive_verbs, set(cls.INTRANSITIVE_VERBS.copy()))
    transitive_verbs = cls.bind_words_to_meanings(
        transitive_verbs, set(cls.TRANSITIVE_VERBS.copy()))
    nouns = cls.bind_words_to_meanings(nouns, set(cls.NOUNS.copy()))
    color_adjectives = cls.bind_words_to_meanings(
        color_adjectives, set(cls.COLOR_ADJECTIVES.copy()))
    size_adjectives = cls.bind_words_to_meanings(
        size_adjectives, set(cls.SIZE_ADJECTIVES.copy()))
    adverbs = cls.bind_words_to_meanings(adverbs, set(cls.ADVERBS.copy()))
    location_preps = cls.bind_words_to_meanings(location_preps,
                                                set(cls.LOCATION_PREPS.copy()))
    return cls(intransitive_verbs, transitive_verbs, adverbs, nouns,
               color_adjectives, size_adjectives, location_preps)

  def to_representation(self):
    representation = super().to_representation()
    representation.update({"location_preps": self._location_preps})
    return representation

  @classmethod
  def from_representation(cls, representation):
    return cls(representation["intransitive_verbs"],
               representation["transitive_verbs"], representation["adverbs"],
               representation["nouns"], representation["color_adjectives"],
               representation["size_adjectives"],
               representation["location_preps"])

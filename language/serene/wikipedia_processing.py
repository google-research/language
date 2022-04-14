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
"""Tools to convert the jsonl-WikiDump into protocol buffer format."""

import re

import unicodedata

from language.serene import fever_pb2

Json = Mapping[Text, Any]

# The WikiDump shipping with FEVER provides the `lines` in the following format,
# in a single string:
#   <sent_num>\t<sent>(\t<mention>\t<entity>)*(\n<sent_num>\t<sent>(\t<mention>\t<entity>)*
# Unfortunately, <sent>, <mention>, and <entity> may themselves contain
# `\n`, which makes naively breaking by new-lines insufficient.  Instead,
# we split by /\n\d+\t/, which will not only split the string into the
# individual <sent>/(<mention>/<entity>)s, but also consume the sentence
# number; as these are simply incrementing from 0 to one, we can trivially
# reconstruct them.
#
# CAVEAT: There is a tiny number of cases where even this breaks down,
# because some mentions may match `\n\d+\t` (e.g., `\1927\t1927 in sports`
# from the "Duluth_Kelleys/Eskimos" article).  Getting these cases correct
# would require actually incrementally parsing the string and with
# expectations about the next sentence number; and even then, may in theory
# break and require back-tracking.  As these complications only seem to
# apply to 5 out of 5,416,537 articles in the dump, we will, at least for
# the time, gracefully disregard these articles.
_LINE_SPLIT_REGEX = re.compile('\n\\d+\t')


def _nfc_normalize(text):
  return unicodedata.normalize('NFC', text)


def _page_name_to_title(page_name):
  return page_name.replace('_', ' ')


def _parse_sentence(sentence_line):
  """Parses a single `sentence` string from the FEVER WikipediaDump."""
  columns: List[Text] = sentence_line.split('\t')

  if len(columns) % 2 != 1:
    raise ValueError('Invalid sentence line, expected (\tmention\tentity)*, '
                     'got: %s' % (sentence_line))

  sentence = fever_pb2.WikipediaDump.Sentence()
  sentence.text = columns[0]

  for i in range(1, len(columns), 2):
    sentence.entities.append(
        fever_pb2.WikipediaDump.Entity(
            mention=columns[i], entity=columns[i + 1]))

  return sentence


def _parse_sentences(
    lines_text):
  """Parses the `lines` values from the FEVER Wikipedia dump."""
  sentences: Dict[int, fever_pb2.WikipediaDump.Sentence] = {}

  if not lines_text:
    return sentences

  # Prepend "\n" so we do not need to special case the first line by having
  # it also match the split pattern.
  lines: List[Text] = _LINE_SPLIT_REGEX.split('\n' + lines_text)
  # Drop the initial empty string.
  lines = lines[1:]

  for sentence_number, sentence_line in enumerate(lines):
    sentence = _parse_sentence(sentence_line)
    if sentence.text:
      sentences[sentence_number] = sentence

  return sentences


def parse_wiki_dump_from_json(json_value):
  return fever_pb2.WikipediaDump(
      id=_nfc_normalize(json_value['id']),
      title=_page_name_to_title(_nfc_normalize(json_value['id'])),
      text=_nfc_normalize(json_value['text']),
      sentences=_parse_sentences(_nfc_normalize(json_value['lines'])))

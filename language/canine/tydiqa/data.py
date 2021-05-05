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
"""Python representations of the TyDi QA primary task data.

This module does not contain any model-specific code nor preprocessing
heuritics. It's used for deserializing the data.

This module does not depend on TensorFlow and should be re-usable within your
favorite ML/DL framework.
"""

import collections
import enum

from typing import Any, Mapping, Optional, Sequence, Text

TextSpan = collections.namedtuple("TextSpan", "byte_positions text")


class AnswerType(enum.IntEnum):
  """Type of TyDi answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  MINIMAL = 3
  PASSAGE = 4


class Language(enum.IntEnum):
  """Names of languages contained in TyDi dataset."""
  ARABIC = 0
  BENGALI = 1
  FINNISH = 2
  INDONESIAN = 3
  JAPANESE = 4
  SWAHILI = 5
  KOREAN = 6
  RUSSIAN = 7
  TELUGU = 8
  THAI = 9
  ENGLISH = 10


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls,
              type_: AnswerType,
              text: Optional[Text] = None,
              offset: Optional[int] = None):
    return super(Answer, cls).__new__(cls, type_, text, offset)


class TyDiExample(object):
  """A single training/test example.

  Typically created by `to_tydi_example`. This class is a fairly straightforward
  serialization of the dict-based entry format created in
  `create_entry_from_json`.
  """

  def __init__(self,
               example_id: int,
               language_id: Language,
               question: Text,
               contexts: Text,
               plaintext: Text,
               context_to_plaintext_offset: Sequence[int],
               answer: Optional[Answer] = None,
               start_byte_offset: Optional[int] = None,
               end_byte_offset: Optional[int] = None):
    self.example_id = example_id

    # A member of the `Language` enumeration as converted by `get_language_id`.
    self.language_id = language_id

    # `question` and `contexts` are the preprocessed question and plaintext
    # with special tokens appended by `create_entry_from_json`. All candidate
    # contexts have been concatenated in `contexts`.
    self.question = question
    self.contexts = contexts

    # `plaintext` is the original article plaintext from the corpus.
    self.plaintext = plaintext

    # `context_to_plaintext_offset` gives a mapping from byte indices in
    # `context` to byte indices in `plaintext`.
    self.context_to_plaintext_offset = context_to_plaintext_offset

    # The following attributes will be `None` for non-training examples.
    # For training, the *offset attributes are derived from the TyDi entry's
    # `start_offset` attribute via `make_tydi_answer`. They are offsets within
    # the original plaintext.
    self.answer = answer
    self.start_byte_offset = start_byte_offset
    self.end_byte_offset = end_byte_offset


def byte_str(text):
  return text.encode("utf-8")


def byte_len(text):
  # Python 3 encodes text as character sequences, not byte sequences
  # (like Python 2).
  return len(byte_str(text))


def byte_slice(text, start, end, errors="replace"):
  # Python 3 encodes text as character sequences, not byte sequences
  # (like Python 2).
  return byte_str(text)[start:end].decode("utf-8", errors=errors)


def has_passage_answer(a):
  return a["passage_answer"]["candidate_index"] >= 0


def get_first_annotation(json_dict, max_passages):
  """Returns the first minimal or passage answer in the example.

  Returns the annotation with the earliest minimal answer span. If no annotation
  has a minimal answer span, then the annotation with the earliest passage
  answer will be returned.

  Args:
    json_dict: annotated example.
    max_passages: see FLAGS.max_passages.

  Returns:
    annotation: (dict) selected annotation.
    annotated_idx: (int) index of the first annotated candidate.
    annotated_span: (tuple) byte offset of the start and end token of the
      minimal answer. The end token is exclusive. This index is relative to this
      particular passage's plaintext, not the full plaintext.
  """

  if "annotations" not in json_dict:
    return None, -1, (-1, -1)

  positive_annotations = sorted(
      [a for a in json_dict["annotations"] if has_passage_answer(a)],
      key=lambda a: a["passage_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["minimal_answer"]:
      # Check if it is a non null answer.
      start_byte_offset = a["minimal_answer"]["plaintext_start_byte"]
      if start_byte_offset < 0:
        continue

      idx = a["passage_answer"]["candidate_index"]
      if idx >= max_passages:
        continue
      end_byte_offset = a["minimal_answer"]["plaintext_end_byte"]
      return a, idx, (global_to_local_offset(json_dict, idx, start_byte_offset),
                      global_to_local_offset(json_dict, idx, end_byte_offset))

  for a in positive_annotations:
    idx = a["passage_answer"]["candidate_index"]
    if idx >= max_passages:
      continue
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)


def get_text_span(example, span):
  """Returns the text in the example's document in the given span."""
  byte_positions = []
  # `text` is a byte string since `document_plaintext` is also a byte string.
  start = span["plaintext_start_byte"]
  end = span["plaintext_end_byte"]
  text = byte_slice(example["document_plaintext"], start, end)
  for i in range(span["plaintext_start_byte"], span["plaintext_end_byte"]):
    byte_positions.append(i)
  return TextSpan(byte_positions, text)


def global_to_local_offset(json_dict, candidate_idx, byte_index):
  """Converts a byte index within the article to the byte offset within the candidate."""
  global_start = json_dict["passage_answer_candidates"][candidate_idx][
      "plaintext_start_byte"]
  return byte_index - global_start


def get_candidate_text(json_dict, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(json_dict["passage_answer_candidates"]):
    raise ValueError("Invalid index for passage candidate: {}".format(idx))

  return get_text_span(json_dict, json_dict["passage_answer_candidates"][idx])


def candidates_iter(json_dict):
  """Yields the candidates that should not be skipped in an example."""
  for idx, cand in enumerate(json_dict["passage_answer_candidates"]):
    yield idx, cand


def make_tydi_answer(contexts: Text, answer: Mapping[Text, Any]) -> Answer:
  """Makes an Answer object following TyDi conventions.

  Args:
    contexts: String containing the context.
    answer: Dictionary with `span_start` and `input_text` fields.

  Returns:
    An Answer object. If the Answer type is YES or NO or PASSAGE, the text
    of the answer is the passage answer. If the answer type is UNKNOWN,
    the text of the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]

  if (answer["candidate_id"] == -1 or start >= byte_len(contexts) or
      end > byte_len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "passage":
    answer_type = AnswerType.PASSAGE
  else:
    answer_type = AnswerType.MINIMAL

  return Answer(
      answer_type, text=byte_slice(contexts, start, end), offset=start)


def get_language_id(input_text: Text) -> Language:
  """Maps string language id into integer."""
  if input_text.lower() == "arabic":
    language_id = Language.ARABIC
  elif input_text.lower() == "finnish":
    language_id = Language.FINNISH
  elif input_text.lower() == "indonesian":
    language_id = Language.INDONESIAN
  elif input_text.lower() == "japanese":
    language_id = Language.JAPANESE
  elif input_text.lower() == "korean":
    language_id = Language.KOREAN
  elif input_text.lower() == "russian":
    language_id = Language.RUSSIAN
  elif input_text.lower() == "swahili":
    language_id = Language.SWAHILI
  elif input_text.lower() == "thai":
    language_id = Language.THAI
  elif input_text.lower() == "telugu":
    language_id = Language.TELUGU
  elif input_text.lower() == "bengali":
    language_id = Language.BENGALI
  elif input_text.lower() == "english":
    language_id = Language.ENGLISH
  else:
    raise ValueError("Invalid language <%s>" % input_text)
  return language_id


def to_tydi_example(entry: Mapping[Text, Any],
                    is_training: bool) -> TyDiExample:
  """Converts a TyDi 'entry' from `create_entry_from_json` to `TyDiExample`."""

  if is_training:
    answer = make_tydi_answer(entry["contexts"], entry["answer"])
    start_byte_offset = answer.offset
    end_byte_offset = answer.offset + byte_len(answer.text)
  else:
    answer = None
    start_byte_offset = None
    end_byte_offset = None

  return TyDiExample(
      example_id=int(entry["id"]),
      language_id=get_language_id(entry["language"]),
      question=entry["question"]["input_text"],
      contexts=entry["contexts"],
      plaintext=entry["plaintext"],
      context_to_plaintext_offset=entry["context_to_plaintext_offset"],
      answer=answer,
      start_byte_offset=start_byte_offset,
      end_byte_offset=end_byte_offset)

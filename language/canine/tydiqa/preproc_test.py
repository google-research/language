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
import pprint
from typing import Text

from language.canine.tydiqa import char_splitter
from language.canine.tydiqa import preproc
from language.canine.tydiqa import tydi_tokenization_interface
import tensorflow.compat.v1 as tf


# For test_srcdir
flags = tf.flags
FLAGS = flags.FLAGS


def _print_dict(d):
  """Can be used on, e.g. a `json_dict` or `result`."""
  print(pprint.PrettyPrinter().pformat(d).replace("°", "@"))


def make_tokenizer() -> tydi_tokenization_interface.TokenizerWithOffsets:
  return char_splitter.CharacterSplitter()


_JSON_MIN_ANSWER = {
    "document_title":
        "Zebra finch",
    "document_url":
        "https://en.wikipedia.org/wiki/Zebra%20finch",
    "example_id":
        111,
    "language":
        "english",
    "document_plaintext":  #
        # Passage 0
        "The zebra finch is the most common estrildid finch. The bird has "
        "been introduced to Puerto Rico.\n"
        # Passage 1
        "The body temperature (as measured from the cloaca) of the zebra "
        "finch may vary from 38 to 44 °C.\n"
        # Passage 2
        "The zebra finch was first collected in 1801 during Nicolas "
        "Baudin's expedition to Australia. It was described in 1817 by "
        "Louis Jean Pierre Vieillot in his Nouveau Dictionnaire d'Histoire "
        "Naturelle.\n"
        # Passage 3
        "Morphological differences between the subspecies. Males do not "
        "have the fine barring found on the throat and upper breast.\n"
        # Passage 4
        "Symmetry of both plumage, like chest bands, and artificial "
        "features, like leg bands, are preferred by the female.\n"
        # Passage 5
        "Nest predators of the zebra finch include the tiger snake.",
    "question_text":
        "Where are a zebra finch's stripes located?",
    "passage_answer_candidates": [{
        "plaintext_start_byte": 0,
        "plaintext_end_byte": 96,
    }, {
        "plaintext_start_byte": 97,
        "plaintext_end_byte": 194,
    }, {
        "plaintext_start_byte": 195,
        "plaintext_end_byte": 392,
    }, {
        "plaintext_start_byte": 393,
        "plaintext_end_byte": 515,
    }, {
        "plaintext_start_byte": 516,
        "plaintext_end_byte": 629,
    }, {
        "plaintext_start_byte": 630,
        "plaintext_end_byte": 688,
    }],
    "annotations": [
        {
            "annotation_id": 222,
            "minimal_answer": {
                # "chest"
                "plaintext_start_byte": 547,
                "plaintext_end_byte": 552,
            },
            "passage_answer": {
                "candidate_index": 4
            },
            "yes_no_answer": "NONE"
        },
        {
            "annotation_id": 333,
            "minimal_answer": {
                "plaintext_start_byte": -1,
                "plaintext_end_byte": -1,
            },
            "passage_answer": {
                "candidate_index": 3
            },
            "yes_no_answer": "NONE"
        },
        {
            "annotation_id": 444,
            "minimal_answer": {
                # "throat and upper breast"
                "plaintext_start_byte": 491,
                "plaintext_end_byte": 514,
            },
            "passage_answer": {
                "candidate_index": 3
            },
            "yes_no_answer": "NONE"
        },
        {
            "annotation_id": 555,
            "minimal_answer": {
                # "throat"
                "plaintext_start_byte": 491,
                "plaintext_end_byte": 497,
            },
            "passage_answer": {
                "candidate_index": 3
            },
            "yes_no_answer": "NONE"
        }
    ],
}

_JSON_PASSAGE_ANSWER = {
    "document_title":
        "Zebra finch",
    "document_url":
        "https://en.wikipedia.org/wiki/Zebra%20finch",
    "example_id":
        200,
    "language":
        "english",
    "document_plaintext":  #
        # Passage 0
        "The zebra finch is the most common estrildid finch.\n"
        # Passage 1
        "The body temperature may vary from 38 to 44 °C.\n"
        # Passage 2
        "Nest predators include the tiger snake.",
    "question_text":
        "Something without a minimal answer?",
    "passage_answer_candidates": [{
        "plaintext_start_byte": 0,
        "plaintext_end_byte": 51,
    }, {
        "plaintext_start_byte": 52,
        "plaintext_end_byte": 100,
    }, {
        "plaintext_start_byte": 101,
        "plaintext_end_byte": 140,
    }],
    "annotations": [{
        "annotation_id": 300,
        "minimal_answer": {
            "plaintext_start_byte": -1,
            "plaintext_end_byte": -1,
        },
        "passage_answer": {
            "candidate_index": 2
        },
        "yes_no_answer": "NONE"
    }, {
        "annotation_id": 400,
        "minimal_answer": {
            "plaintext_start_byte": -1,
            "plaintext_end_byte": -1,
        },
        "passage_answer": {
            "candidate_index": 1
        },
        "yes_no_answer": "NONE"
    }],
}

_JSON_NO_ANSWER = {
    "document_title":
        "Zebra finch",
    "document_url":
        "https://en.wikipedia.org/wiki/Zebra%20finch",
    "example_id":
        200,
    "language":
        "english",
    "document_plaintext":  #
        # Passage 0
        "The zebra finch is the most common estrildid finch.\n"
        # Passage 1
        "The body temperature may vary from 38 to 44 °C.",
    "question_text":
        "Something without a minimal answer?",
    "passage_answer_candidates": [{
        "plaintext_start_byte": 0,
        "plaintext_end_byte": 51,
    }, {
        "plaintext_start_byte": 52,
        "plaintext_end_byte": 100,
    }],
    "annotations": [],
}


class PreprocTest(tf.test.TestCase):

  def assertCreateEntryFromJsonResult(
      self,
      json_dict,
      result,
      expected_context: Text,
      expected_answer_type: Text,
      expected_passage_answer_index: int,
      expected_min_span_start: int,
      expected_min_span_end: int,
      expected_min_span_text: Text,
  ):
    self.assertAllEqual(
        set(result),
        set([
            "id", "name", "language", "plaintext", "question", "contexts",
            "answer", "context_to_plaintext_offset", "has_correct_context"
        ]))

    # Assert that certain fields are copied from the input.
    self.assertEqual(result["id"], str(json_dict["example_id"]))
    self.assertEqual(result["name"], json_dict["document_title"])
    self.assertEqual(result["language"], json_dict["language"])
    self.assertEqual(result["plaintext"], json_dict["document_plaintext"])
    self.assertEqual(result["question"]["input_text"],
                     json_dict["question_text"])

    # Assert that the article text is properly augmented, including the
    # addition of special passage markers.
    self.assertEqual(result["contexts"], expected_context)

    # Assert that the correct answer information is retrieved, and that the
    # answer span byte offsets into `contexts` have been computed correctly.
    self.assertAllEqual(
        result["answer"], {
            "candidate_id": expected_passage_answer_index,
            "input_text": expected_answer_type,
            "span_start": expected_min_span_start,
            "span_end": expected_min_span_end,
            "span_text": expected_min_span_text,
        })

    context_bytes = result["contexts"].encode()
    plaintext_bytes = json_dict["document_plaintext"].encode()
    context_to_plaintext_offset = result["context_to_plaintext_offset"]

    # Assert that `contexts` actually contains the expected answer at the
    # location given by the computed span offsets.
    self.assertEqual(
        context_bytes[expected_min_span_start:expected_min_span_end].decode(),
        expected_min_span_text)

    # Assert that the context-to-plaintext mapping exactly covers the bytes
    # of `contexts`.
    self.assertLen(context_to_plaintext_offset, len(context_bytes))

    # Assert that the plaintext and 'contexts' bytes actually match when
    # `context_to_plaintext_offset` says they should.
    mapped_context_bytes, mapped_plaintext_bytes = (
        zip(*[(context_bytes[ci], plaintext_bytes[pi])
              for ci, pi in enumerate(context_to_plaintext_offset)
              if pi != -1]))
    self.assertAllEqual(mapped_context_bytes, mapped_plaintext_bytes)

  def test_create_entry_from_json_min_answer(self):
    json_dict = _JSON_MIN_ANSWER
    result = preproc.create_entry_from_json(
        json_dict,
        max_passages=45,
        max_position=45,
        tokenizer=make_tokenizer(),
        fail_on_invalid=True)

    # Checks that passage markers generated by TyDiTokenizer.get_passage_marker
    # are inserted by preproc.create_entry_from_json.
    self.assertCreateEntryFromJsonResult(
        json_dict=json_dict,
        result=result,
        expected_context=(
            "\ue006 The zebra finch is the most common estrildid finch. "
            "The bird has been introduced to Puerto Rico. "
            "\ue007 The body temperature (as measured from the cloaca) "
            "of the zebra finch may vary from 38 to 44 °C. "
            "\ue008 The zebra finch was first collected in 1801 during "
            "Nicolas Baudin's expedition to Australia. It was described in "
            "1817 by Louis Jean Pierre Vieillot in his Nouveau Dictionnaire "
            "d'Histoire Naturelle. "
            "\ue009 Morphological differences between the subspecies. "
            "Males do not have the fine barring found on the throat and upper "
            "breast. "
            "\ue00a Symmetry of both plumage, like chest bands, and "
            "artificial features, like leg bands, are preferred by the female. "
            "\ue00b Nest predators of the zebra finch include the tiger "
            "snake."),
        expected_answer_type="minimal",
        expected_passage_answer_index=3,
        expected_min_span_start=507,
        expected_min_span_end=530,
        expected_min_span_text="throat and upper breast")

  def test_create_entry_from_json_passage_answer(self):
    json_dict = _JSON_PASSAGE_ANSWER
    result = preproc.create_entry_from_json(
        json_dict,
        max_passages=45,
        max_position=45,
        tokenizer=make_tokenizer(),
        fail_on_invalid=True)

    # Checks that passage markers generated by TyDiTokenizer.get_passage_marker
    # are inserted by preproc.create_entry_from_json.
    self.assertCreateEntryFromJsonResult(
        json_dict=json_dict,
        result=result,
        expected_context=(
            "\ue006 The zebra finch is the most common estrildid finch. "
            "\ue007 The body temperature may vary from 38 to 44 °C. "
            "\ue008 Nest predators include the tiger snake."),
        expected_answer_type="passage",
        expected_passage_answer_index=1,
        expected_min_span_start=60,
        expected_min_span_end=108,
        expected_min_span_text="The body temperature may vary from 38 to 44 °C."
    )

  def test_create_entry_from_json_no_answer(self):
    json_dict = _JSON_NO_ANSWER
    result = preproc.create_entry_from_json(
        json_dict,
        max_passages=45,
        max_position=45,
        tokenizer=make_tokenizer(),
        fail_on_invalid=True)

    # Checks that passage markers generated by TyDiTokenizer.get_passage_marker
    # are inserted by preproc.create_entry_from_json.
    self.assertCreateEntryFromJsonResult(
        json_dict=json_dict,
        result=result,
        expected_context=(
            "\ue006 The zebra finch is the most common estrildid finch. "
            "\ue007 The body temperature may vary from 38 to 44 °C."),
        expected_answer_type="passage",
        expected_passage_answer_index=-1,
        expected_min_span_start=-1,
        expected_min_span_end=-1,
        expected_min_span_text="")


if __name__ == "__main__":
  tf.test.main()

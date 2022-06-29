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
"""Unit tests for rendering_utils.py."""

import re

from absl.testing import absltest
from absl.testing import parameterized
from language.fruit import rendering_utils
import numpy as np


class RenderAllTest(absltest.TestCase):

  def test_render_all(self):
    assert rendering_utils.render_all([1, "hi"], sep="-") == "1-hi"


class InvertibleDelimiterConstructorTest(absltest.TestCase):

  def test_bad_constructor_raises_error(self):
    # Checks that inversion validation works properly.
    with self.assertRaises(ValueError):
      rendering_utils.InvertibleDelimiterConstructor(
          fstring="[{uid}]", regex=re.compile(r"\((?P<uid>\d+)\)"))


class InvertibleDelimiterRangeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.default_constructor = rendering_utils.square_bracket_delimiter_constructor
    self.default_delimiter_range = rendering_utils.InvertibleDelimiterRange(
        self.default_constructor,
        end=10,
    )

  def test_iter(self):
    for i, delimiter in enumerate(self.default_delimiter_range):
      self.assertEqual(f"[{i}]", delimiter)

  def test_in_range(self):
    self.assertTrue(self.default_delimiter_range._in_range(5))
    self.assertFalse(self.default_delimiter_range._in_range(100))

  def test_finditer(self):
    matches = list(self.default_delimiter_range.finditer("[0]"))
    self.assertLen(matches, 1)
    self.assertEqual(matches[0].group("uid"), "0")

  def test_finduids(self):
    uids = self.default_delimiter_range.finduids("[0] [1] asdf [2]")
    self.assertListEqual(uids, [0, 1, 2])

  def test_split(self):
    chunks = self.default_delimiter_range.split("front [0] back")
    self.assertListEqual(chunks, ["front", "back"])
    chunks = self.default_delimiter_range.split(
        "front [0] back", keep_delims=True)
    self.assertListEqual(chunks, ["front", "[0] back"])

  def test_remove_delims(self):
    cleaned = self.default_delimiter_range.remove_delims("front [0] back")
    self.assertEqual(cleaned, "front back")

  def test_copy_replace(self):
    src = ["middle"]
    edited = self.default_delimiter_range.copy_replace("front [0] back", src)
    self.assertEqual(edited, "front middle back")

  def test_null_constructor(self):
    # Checks that regex operations behave as expected with null constructors
    delimiter_range = rendering_utils.InvertibleDelimiterRange(
        rendering_utils.null_delimiter_constructor, end=10)
    for delimiter in delimiter_range:
      matches = list(delimiter_range.finditer(delimiter))
      uids = delimiter_range.finduids(delimiter)
      self.assertEmpty(matches)
      self.assertEmpty(uids)


class _MockTokenizer:

  def tokenize(self, x):
    return np.array(x.split(" "))


class _MockSentenceTokenizer:

  def span_tokenize(self, x):
    start = 0
    spans = []
    for match in re.finditer(r"\.", x):
      spans.append((start, match.end()))
      start = match.end()
    if start != len(x):
      spans.append((start, len(x)))
    return spans


class SentenceMatchingTest(absltest.TestCase):

  def test_iou_score(self):
    sent_a = "a b c"
    sent_b = "a b d"
    score = rendering_utils.iou_score(sent_a, sent_b)
    self.assertEqual(score, 0.5)

  def test_score_matrix(self):
    sents_a = ["a b c", "d e f"]
    sents_b = ["a b d", "c e f"]
    observed = rendering_utils.score_matrix(sents_a, sents_b)
    expected = np.array([[2 / 4, 1 / 5], [1 / 5, 2 / 4]])
    np.testing.assert_almost_equal(observed, expected)

  def test_match_sents(self):
    # Test 1: Perfect match
    sents = ["a b c", "d e f"]
    best_seq, best_score = rendering_utils.match_sents(sents, sents)
    self.assertEqual(best_seq, [0, 1])
    self.assertEqual(best_score, 2.0)

    # Test 2: Extra sent_a w/ perfect matches
    big_sents = ["a b c", "skip", "d e f"]
    best_seq, best_score = rendering_utils.match_sents(big_sents, sents)
    self.assertEqual(best_seq, [0, None, 1])
    self.assertEqual(best_score, 2.0)

    # Test 3: Extra sent_b w/ perfect matches
    best_seq, best_score = rendering_utils.match_sents(sents, big_sents)
    self.assertEqual(best_seq, [0, 2])
    self.assertEqual(best_score, 2.0)


class Seq2SeqInputBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.default_tokenizer = _MockTokenizer()
    self.default_sentence_tokenizer = _MockSentenceTokenizer()
    self.default_task = rendering_utils.Task.diff
    self.default_delimiter_range_pair = (
        rendering_utils.get_default_delimiter_range_pair(
            task=self.default_task,
            delimiter_type=rendering_utils.DelimiterType.text,
        ))

  def _get_builder(self, **kwargs):
    task = kwargs.pop("task", self.default_task)
    delimiter_range_pair = kwargs.pop("delimiter_range_pair",
                                      self.default_delimiter_range_pair)
    return rendering_utils.Seq2SeqInputBuilder(
        task=task,
        sentence_tokenizer=self.default_sentence_tokenizer,
        tokenizer=self.default_tokenizer,
        delimiter_range_pair=delimiter_range_pair,
        **kwargs,
    )

  def test_convert_to_evidence(self):
    builder = self._get_builder()
    test_mention = {
        "title": "John Lennon",
        "section": "INTRODUCTION",
        "text": "I am the walrus",
        "entities": [{
            "id": "walrus",
            "start": 9,
            "end": 15
        }],
    }
    test_annotated_mentions = [{"mention": test_mention}]
    evidence = builder._convert_to_evidence(test_annotated_mentions)
    # Check list has correct length
    self.assertLen(evidence, 1)
    # Check string and entity ids
    e = evidence.pop()
    self.assertSetEqual(e.entity_ids, {"John Lennon", "walrus"})
    self.assertEqual(str(e), "(0) John Lennon INTRODUCTION I am the walrus")

  def test_convert_source_sentences(self):
    builder = self._get_builder()
    test_source = {"text": "Sentence One. Sentence Two."}
    source_sentences = builder._convert_source_sentences(test_source)
    observed_strings = [str(x) for x in source_sentences]
    expected_strings = [
        "[0] Sentence One.",
        "[1] Sentence Two.",
    ]
    self.assertListEqual(observed_strings, expected_strings)

  def test_convert_target_sentences(self):
    builder = self._get_builder()
    test_target = {
        "text":
            "Sentence One. Sentence Two.",
        "added_entities": [
            {
                "id": "e1",
                "start": 0,
                "end": 8
            },
            {
                "id": "e2",
                "start": 13,
                "end": 21
            },
        ]
    }
    target_sentences = builder._convert_target_sentences(test_target)
    observed_strings = [str(x) for x in target_sentences]
    expected_strings = [
        "Sentence One.",
        "Sentence Two.",
    ]
    self.assertListEqual(observed_strings, expected_strings)
    # Check that entities associated with correct sentences.
    self.assertSetEqual(target_sentences[0].entity_ids, {"e1"})
    self.assertSetEqual(target_sentences[1].entity_ids, {"e2"})

  def test_get_generatable_methods(self):
    builder = self._get_builder()
    delimiter_iter = iter(builder.evidence_delimiter_range)

    test_target = {
        "text":
            "Sentence One. Sentence Two.",
        "added_entities": [
            {
                "id": "e1",
                "start": 0,
                "end": 8
            },
            {
                "id": "e2",
                "start": 13,
                "end": 21
            },
        ]
    }
    test_evidence = [
        rendering_utils.Evidence(
            delimiter=next(delimiter_iter),
            title="Foo",
            section="Omaha",
            text="Mention of e1.",
            entity_ids={"e1"},
        )
    ]
    test_mention = {
        "title": "Foo",
        "section": "Omaha",
        "text": "Mention of E1.",
        "entities": [{
            "id": "e1",
            "start": 11,
            "end": 13
        }]
    }
    test_annotated_mentions = [{"mention": test_mention}]
    generatable_entity_ids = builder._get_generatable_entity_ids(
        target=test_target, evidence=test_evidence)
    observed = builder._get_generatable_surfaces(
        target=test_target,
        annotated_mentions=test_annotated_mentions,
        generatable_entity_ids=generatable_entity_ids)
    expected = {"e1": ["Sentence", "E1"]}
    self.assertSetEqual(set(observed.keys()), {"e1"})
    self.assertSetEqual(set(observed["e1"]), set(expected["e1"]))

  def test_get_length(self):
    builder = self._get_builder()
    length = builder._get_length("I am a banana")
    self.assertEqual(length, 4)

  def test_get_controllable_inputs_targets(self):
    builder = self._get_builder(
        task=rendering_utils.Task.controllable,
        evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        include_source=False,
        include_evidence=True,
        include_distractors=True,
    )
    sentence_delimiter_iter = iter(builder.sentence_delimiter_range)
    evidence_delimiter_iter = iter(builder.evidence_delimiter_range)
    source_sentences = [
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Deleted.",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="This sentence is mostly the same but is.",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
    ]
    target_sentences = [
        rendering_utils.Sentence(
            delimiter="",
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="Intended to trip up cruncher.",
            entity_ids={"e1"},
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="This sentence is mostly the same but is replaced.",
            entity_ids={"e1"},
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="Same. ",
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="E1 new sentence.",
            entity_ids={"e2"},
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
    ]
    evidence = [
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence1",
            section="INTRO",
            text="Included evidence for replacement.",  # length: 2
            entity_ids={"e1"},
        ),
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence1",
            section="INTRO",
            text="Included distractor.",  # length: 2
            entity_ids=set(),
        ),
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence2",
            section="Morghul",
            text="Included evidence for addition.",
            entity_ids={"e2"},
        ),
    ]
    inputs_targets = list(
        builder._get_controllable_inputs_targets(source_sentences,
                                                 target_sentences, evidence))
    self.assertLen(inputs_targets, 1)
    observed = inputs_targets.pop()
    expected = {
        "inputs":
            ("[0] Same. [DELETION] [1] Deleted. [2] Same. [ADDITION] (0) [EDIT]"
             " [3] (0) This sentence is mostly the same but is. [4] Same. "
             "[ADDITION] (2) [CONTEXT] (0) Evidence1 INTRO Included evidence "
             "for replacement. (1) Evidence1 INTRO Included distractor. (2) "
             "Evidence2 Morghul Included evidence for addition."),
        "targets": (
            "[0] [2] Intended to trip up cruncher. This sentence is mostly the "
            "same but is replaced. [4] E1 new sentence."),
    }
    self.assertDictEqual(observed, expected)

  def test_get_inputs_targets(self):
    # Max input length set so that last piece of evidence should be truncated.
    builder = self._get_builder(
        include_source=False,
        include_evidence=True,
        include_distractors=True,
        max_input_length=17,
    )
    sentence_delimiter_iter = iter(builder.sentence_delimiter_range)
    evidence_delimiter_iter = iter(builder.evidence_delimiter_range)
    source_sentences = [
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Sentence 0. ",  # length: 2
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
        rendering_utils.Sentence(
            delimiter=next(sentence_delimiter_iter),
            text="Sentence 1.",  # length: 2
            evidence_marker_type=rendering_utils.EvidenceMarkerType.empty,
        ),
    ]
    target_sentences = [
        rendering_utils.Sentence(
            delimiter="",
            text="Sentence 0.",
            entity_ids=set(),
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
        rendering_utils.Sentence(
            delimiter="",
            text="E1 new sentence.",
            entity_ids={"e1"},
            evidence_marker_type=rendering_utils.EvidenceMarkerType.reference,
        ),
    ]
    evidence = [
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence1",
            section="INTRO",
            text="Included evidence.",  # length: 2
            entity_ids={"e1"},
        ),
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence1",
            section="INTRO",
            text="Included distractor.",  # length: 2
            entity_ids=set(),
        ),
        rendering_utils.Evidence(
            delimiter=next(evidence_delimiter_iter),
            title="Evidence2",
            section="Morghul",
            text="Excluded evidence.",
            entity_ids={"e1"},
        ),
    ]
    inputs_targets = list(
        builder._get_inputs_targets(source_sentences, target_sentences,
                                    evidence))
    self.assertLen(inputs_targets, 1)
    observed = inputs_targets.pop()
    expected = {
        "inputs":
            ("[0] Sentence 0. [1] Sentence 1. [CONTEXT] (0) Evidence1 INTRO "
             "Included evidence. (1) Evidence1 INTRO Included distractor."),
        "targets": "[0] (0) E1 new sentence.",
    }
    self.assertDictEqual(observed, expected)

  @parameterized.product(
      task=list(rendering_utils.Task),
      evidence_marker_type=list(rendering_utils.EvidenceMarkerType),
      delimiter_type=list(rendering_utils.DelimiterType))
  def test_tasks_delimiters_and_normalization(self, task, evidence_marker_type,
                                              delimiter_type):
    delimiter_range_pair = rendering_utils.get_default_delimiter_range_pair(
        task, delimiter_type)
    builder = self._get_builder(
        task=task,
        delimiter_range_pair=delimiter_range_pair,
        evidence_marker_type=evidence_marker_type,
    )
    article_pair = {
        "source_article": {
            "title": "test",
            "ns": 0,
            "id": 1234,
            "text": "This is deleted text. This is kept text.",
            "entities": [],
            "added_entities": [],
        },
        "target_article": {
            "title": "test",
            "ns": 0,
            "id": 1234,
            "text": "E1 is new. This is kept text.",
            "entities": [{
                "id": "e1",
                "start": 0,
                "end": 2
            }],
            "added_entities": [{
                "id": "e1",
                "start": 0,
                "end": 2
            }],
        },
        "updated":
            True,
        "annotated_mentions": [{
            "mention": {
                "title": "foo",
                "section": "INTRODUCION",
                "text": "E1 is here too.",
                "entities": [{
                    "id": "e1",
                    "start": 0,
                    "end": 2
                }],
                "added_entities": [{
                    "id": "e1",
                    "start": 0,
                    "end": 2
                }],
                "is_update": True,
                "key": 12345,
            },
            "label": 1
        }]
    }

    output = list(builder(article_pair))
    self.assertLen(output, 1)
    output = output.pop()

    # Although normalize is not a method of Seq2SeqBuilder we test that all
    # possible ways of creating inputs can be normalized back to the original
    # text.
    id_ = output["id"]
    inputs = output["inputs"]
    targets = output["targets"]
    clean_source, clean_edited = rendering_utils.normalize(
        inputs,
        targets,
        builder.delimiter_range_pair,
        builder.task,
    )
    self.assertEqual(id_, article_pair["target_article"]["id"])
    self.assertEqual(clean_source, article_pair["source_article"]["text"])
    self.assertEqual(clean_edited, article_pair["target_article"]["text"])


if __name__ == "__main__":
  absltest.main()

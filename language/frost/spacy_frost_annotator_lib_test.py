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
"""Tests for spacy_frost_annotator_lib."""

from absl.testing import absltest
from language.frost import spacy_frost_annotator_lib


class SpacyFrostAnnotatorTest(absltest.TestCase):

  def test_on_example(self):
    # Spacy Forst annotator function
    frost_processor = spacy_frost_annotator_lib.get_spacy_frost_processor_fn()

    test_sentence_1 = "This is a test sentence."
    self.assertEqual(
        frost_processor(test_sentence_1),
        "[CONTENT]  [SUMMARY] " + test_sentence_1)

    test_sentence_2 = "This is a test sentence. This is another test sentence."
    self.assertEqual(
        frost_processor(test_sentence_2),
        "[CONTENT]  |||  [SUMMARY] " + test_sentence_2)

    test_sentence_3 = "This is the first sentence. This is the second sentence."
    self.assertEqual(
        frost_processor(test_sentence_3),
        "[CONTENT] first ||| second [SUMMARY] " + test_sentence_3)

    test_sentence_4 = ("Sally Forrest, an actress-dancer who graced the "
                       "silver screen throughout the '40s and '50s in MGM "
                       "musicals and films died on March 15.")
    self.assertEqual(
        frost_processor(test_sentence_4),
        ("[CONTENT] Sally Forrest | the '40s and '50s | MGM | "
         "March 15 [SUMMARY] ") + test_sentence_4)

    test_sentence_5 = ("Sally Forrest, an actress-dancer who graced the "
                       "silver screen throughout the '40s and '50s in MGM "
                       "musicals and films died on March 15.\nForrest, whose "
                       "birth name was Katherine Feeney, had long battled "
                       "cancer.\nA San Diego native, Forrest became a "
                       "protege of Hollywood trailblazer Ida Lupino, who "
                       "cast her in starring roles in films.")
    self.assertEqual(
        frost_processor(test_sentence_5),
        ("[CONTENT] Sally Forrest | the '40s and '50s | MGM | "
         "March 15 ||| Forrest | Katherine Feeney ||| San Diego "
         "| Forrest | Hollywood | Ida Lupino [SUMMARY] ") + test_sentence_5)


if __name__ == "__main__":
  absltest.main()

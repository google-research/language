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
"""Tests for wiki_utils."""

from absl.testing import absltest
from language.fruit import wiki_utils


class WikiUtilsTest(absltest.TestCase):

  def test_validate_fields(self):
    self.assertFalse(wiki_utils.validate_fields({"ns": "0"}))
    self.assertTrue(
        wiki_utils.validate_fields({
            "ns": "0",
            "title": "tests that pass",
            "text": "written by someone other than me."
        }))

  def test_remove_element(self):
    text = "Keep [[Removable:extra stuff]] also keep."
    observed = wiki_utils.remove_element(text, "Removable")
    expected = "Keep  also keep."
    self.assertEqual(expected, observed)

  def test_process_wikitext(self):
    text = "'''This''' all&nbsp;stays == Not this"
    observed = wiki_utils.process_wikitext(text, True, True)
    expected = "This all stays "
    self.assertEqual(expected, observed)

  def test_split_tables(self):
    input_text = r"Text {| |+ Caption |- ! Header |- | Row || |}"
    text, tables = wiki_utils.split_tables(input_text)
    self.assertEqual(text, "Text ")
    self.assertListEqual(tables, ["{| |+ Caption |- ! Header |- | Row || |}"])

  def test_process_table(self):
    table = ("{|\n|+ Caption\n"
             "|-\n! HeaderA !! HeaderB\n"
             "|-\nRow1a || Row1b\n"
             "|-\nRow2a || Row2b\n"
             "|}")
    expected = [
        ("[CAPTION] Caption [HEADER] [COL] HeaderA [COL] HeaderB "
         "[ROW] Row1a [COL] Row1b\n"),
        ("[CAPTION] Caption [HEADER] [COL] HeaderA [COL] HeaderB "
         "[ROW] Row2a [COL] Row2b\n"),
    ]
    observed = list(wiki_utils.process_table(table))
    self.assertListEqual(expected, observed)

  def test_clean_wikilink(self):
    self.assertEqual(
        wiki_utils.clean_wikilink("george washington"), "George_washington")

  def test_process_wikilinks(self):
    text = "[[george washington|The president]]"
    clean_text, links = wiki_utils.process_wikilinks(text)
    self.assertEqual(clean_text, "The president")
    self.assertListEqual(links, [{
        "id": "George_washington",
        "start": 0,
        "end": 13
    }])


if __name__ == "__main__":
  absltest.main()

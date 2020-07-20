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
# Lint as: python3
"""Test utilities for tables to text."""

from absl.testing import absltest
from language.totto import table_to_text_utils


class TableToTextUtilsTest(absltest.TestCase):
  """Test class for testing table to text utilities."""

  def _get_toy_table(self):
    """Get toy example for testing."""
    row1 = [{
        "column_span": 1,
        "is_header": True,
        "row_span": 1,
        "value": "Year"
    }, {
        "column_span": 1,
        "is_header": True,
        "row_span": 1,
        "value": "Title"
    }]

    row2 = [{
        "column_span": 1,
        "is_header": False,
        "row_span": 1,
        "value": "2020"
    }, {
        "column_span": 1,
        "is_header": False,
        "row_span": 1,
        "value": "totto"
    }]

    table = [row1, row2]
    return table

  def test_get_highlighted_cells(self):
    """Tests whether the references are returned as expected."""
    table = self._get_toy_table()
    cell_indices = [[0, 0], [1, 1]]
    subtable = table_to_text_utils.get_highlighted_subtable(table, cell_indices)
    assert len(subtable) == 2
    assert subtable[0] == table[0][0]
    assert subtable[1] == table[1][1]

  def test_get_table_parent_format(self):
    """Tests get_table_parent_format function."""
    table = self._get_toy_table()
    table_page_title = "Tables to Texts"
    table_section_title = "Structured Data"
    table_section_text = "ToTTo is a new dataset"
    table_parent_format = table_to_text_utils.get_table_parent_format(
        table, table_page_title, table_section_title, table_section_text)

    true_str = ("header|||Year\theader|||Title\tcell|||2020\tcell|||totto\t" +
                "page_title|||Tables to Texts\t" +
                "section_title|||Structured Data\t" +
                "section_text|||ToTTo is a new dataset")
    assert table_parent_format == true_str

  def test_get_subtable_parent_format(self):
    """Tests get_table_parent_format function."""
    table = self._get_toy_table()
    subtable = table_to_text_utils.get_highlighted_subtable(
        table, [[0, 0], [1, 1]])
    table_page_title = "Tables to Texts"
    table_section_title = "Structured Data"
    subtable_parent_format = table_to_text_utils.get_subtable_parent_format(
        subtable, table_page_title, table_section_title)

    true_str = ("header|||Year\tcell|||totto\t" +
                "page_title|||Tables to Texts\t" +
                "section_title|||Structured Data")
    assert subtable_parent_format == true_str
    print(subtable_parent_format)


if __name__ == "__main__":
  absltest.main()

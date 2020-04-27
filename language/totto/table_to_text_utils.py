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
"""Utilities for tables-to-text."""
import copy


def add_adjusted_offsets(table):
  """Add adjusted column offsets to take into account multi-column cells."""
  adjusted_table = []
  for row in table:
    real_col_index = 0
    adjusted_row = []
    for cell in row:
      adjusted_cell = copy.deepcopy(cell)
      adjusted_cell["adjusted_col_start"] = real_col_index
      adjusted_cell["adjusted_col_end"] = (
          adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
      real_col_index += adjusted_cell["column_span"]
      adjusted_row.append(adjusted_cell)
    adjusted_table.append(adjusted_row)
  return adjusted_table


def get_row_headers(adjusted_table, row_index, col_index):
  """Heuristic to find row headers."""
  row_headers = []
  row = adjusted_table[row_index]
  for i in range(0, col_index):
    if row[i]["is_header"]:
      row_headers.append(row[i])
  return row_headers


def get_col_headers(adjusted_table, row_index, col_index):
  """Heuristic to find column headers."""
  adjusted_cell = adjusted_table[row_index][col_index]
  adjusted_col_start = adjusted_cell["adjusted_col_start"]
  adjusted_col_end = adjusted_cell["adjusted_col_end"]
  col_headers = []
  for r in range(0, row_index):
    row = adjusted_table[r]
    for cell in row:
      if (cell["adjusted_col_start"] < adjusted_col_end and
          cell["adjusted_col_end"] > adjusted_col_start):
        if cell["is_header"]:
          col_headers.append(cell)

  return col_headers


def get_highlighted_subtable(table, cell_indices, with_headers=False):
  """Extract out the highlighted part of a table, optionally with headers."""
  highlighted_table = []

  adjusted_table = add_adjusted_offsets(table)

  for (row_index, col_index) in cell_indices:
    cell = table[row_index][col_index]
    if with_headers:
      # Heuristically obtain all the row/column headers for this cell.
      row_headers = get_row_headers(adjusted_table, row_index, col_index)
      col_headers = get_col_headers(adjusted_table, row_index, col_index)
    else:
      row_headers = []
      col_headers = []

    highlighted_cell = {
        "cell": cell,
        "row_headers": row_headers,
        "col_headers": col_headers
    }
    highlighted_table.append(highlighted_cell)

  return highlighted_table


def get_table_parent_format(table, table_page_title, table_section_title,
                            table_section_text):
  """Convert table to format required by PARENT."""
  table_parent_array = []

  # Table values.
  for row in table:
    for cell in row:
      # For highlighted tables the cell is nested.
      if "cell" in cell:
        cell = cell["cell"]

      if cell["is_header"]:
        attribute = "header"
      else:
        attribute = "cell"
      value = cell["value"].strip()
      if value:
        value = value.replace("|", "-")
        entry = "%s|||%s" % (attribute, value)
        table_parent_array.append(entry)

  # Page title.
  if table_page_title:
    table_page_title = table_page_title.replace("|", "-")
    entry = "%s|||%s" % ("page_title", table_page_title)
    table_parent_array.append(entry)

  # Section title.
  if table_section_title:
    table_section_title = table_section_title.replace("|", "-")
    entry = "%s|||%s" % ("section_title", table_section_title)
    table_parent_array.append(entry)

  # Section text.
  if table_section_text:
    table_section_text = table_section_text.replace("|", "-")
    entry = "%s|||%s" % ("section_text", table_section_text)
    table_parent_array.append(entry)

  table_parent_str = "\t".join(table_parent_array)
  return table_parent_str


def get_subtable_parent_format(subtable, table_page_title, table_section_title):
  """Convert subtable to format required by PARENT."""
  table_parent_array = []
  # Table values.
  for item in subtable:
    cell = item["cell"]
    if cell["is_header"]:
      attribute = "header"
    else:
      attribute = "cell"
    value = cell["value"].strip()
    if value:
      value = value.replace("|", "-")
      entry = "%s|||%s" % (attribute, value)
      table_parent_array.append(entry)

  # Page title.
  if table_page_title:
    table_page_title = table_page_title.replace("|", "-")
    entry = "%s|||%s" % ("page_title", table_page_title)
    table_parent_array.append(entry)

  # Section title.
  if table_section_title:
    table_section_title = table_section_title.replace("|", "-")
    entry = "%s|||%s" % ("section_title", table_section_title)
    table_parent_array.append(entry)

  table_parent_str = "\t".join(table_parent_array)
  return table_parent_str

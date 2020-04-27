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
"""Utilities to visualize a ToTTo example."""


def get_table_style():
  return """<style>
             table { border-collapse: collapse; }
             th, td {
               word-wrap: break-word;
               max-width: 100%;
               font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
               border-bottom: 1px solid #ddd;
               padding: 5px;
               text-align: left;
             }
            tr:hover {background: #f4f4f4;}
            tr:hover .highlighted {background: repeating-linear-gradient(
                    45deg,
                    #ffff99,
                    #ffff99 10px,
                    #f4f4f4 10px,
                    #f4f4f4 20px
                  );}
           .highlighted { background-color: #ffff99; }
          </style>"""


def get_cell_html(cell, highlight):
  """Get html string for a table cell."""
  if highlight:
    color_str = """ class="highlighted" """
  else:
    color_str = ""

  is_header = cell["is_header"]
  cell_symbol = "td"

  if is_header:
    cell_symbol = "th"

  start_marker = "<%s%s>" % (cell_symbol, color_str)
  end_marker = "</%s>" % cell_symbol

  col_span = cell["column_span"]
  row_span = cell["row_span"]
  start_marker = "<%s%s colspan=%d rowspan=%d >" % (cell_symbol, color_str,
                                                    col_span, row_span)

  val = cell["value"]
  cell_html = start_marker + " " + val + " " + end_marker
  return cell_html


def get_table_html(table, highlighted_cells):
  """Get html for a table and a subset of highlighted cells."""
  table_str = "<table>\n"
  for r_index, row in enumerate(table):
    row_str = "<tr> "
    for c_index, cell in enumerate(row):
      if [r_index, c_index] in highlighted_cells:
        cell_html = get_cell_html(cell, True)
      else:
        cell_html = get_cell_html(cell, False)
      row_str += cell_html

    row_str += "</tr>\n"
    table_str += row_str

  table_str += "</table>"
  return table_str


def get_html_header(title):
  style_str = get_table_style()
  header = "<!doctype html> <html> <head>%s</head><body><h2>%s</h2><br>" % (
      style_str, title)
  return header


def get_example_html(json_example):
  """Get an HTML string for this json example."""
  annotations = json_example["sentence_annotations"]
  table_html = ""
  final_sentences = []

  table_page_title = json_example["table_page_title"]
  table_section_title = json_example["table_section_title"]
  table_section_text = json_example["table_section_text"]
  highlighted_cells = json_example["highlighted_cells"]
  table_html = get_table_html(json_example["table"], highlighted_cells)

  if not table_section_text:
    table_section_text = "<i> None </i>"

  for annotation in annotations:
    final_sentences.append(annotation["final_sentence"])

  html_header = get_html_header(table_page_title)
  all_final_sentences = "<h3>Sentence(s)</h3>" + "<br> ".join(final_sentences)

  html_str = (
      "%s <b>Section Title</b>: %s <br><b>Table Section Text</b>: %s <br> %s "
      "<br> %s </body></html>" %
      (html_header, table_section_title, table_section_text, table_html,
       all_final_sentences))

  return html_str

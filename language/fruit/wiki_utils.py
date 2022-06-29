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
r"""Utilities for processing Wikipedia article dumps."""

import re

from xml.etree import ElementTree

RE_TABLE = re.compile(r"{\|(?:(?!{\||\|}).)*\|}", re.DOTALL)
RE_TEMPLATE = re.compile(r"{{(?:(?!{{|}}).)*}}", re.DOTALL)
RE_REF = re.compile(r"<ref([^<]+?\/>|.*?<\/ref>)", re.DOTALL)
RE_COMMENT = re.compile(r"<!--(?:(?!<!--|-->).)*-->", re.DOTALL)
RE_HTML = re.compile(r"<.*?>", re.DOTALL)
RE_MENTION = re.compile(r"(\[\[)(?P<link>.+?)(\|(?P<surface>.+?))?(\]\])")
RE_TITLE = re.compile(r"^(?P<equals>={2,})(?P<title>[^=]+)={2,}$", re.MULTILINE)
RE_BREADCRUMB = re.compile(r"[^\\]#")
RE_DISAMBIGUATION = re.compile(r"{{disambiguation}}", re.IGNORECASE)
XMLNS = "{http://www.mediawiki.org/xml/export-0.10/}"


def validate_fields(fields):
  """Checks the validity of fields extracted from a <PAGE> element.

  Args:
    fields: A dictionary of fields extracted from a <PAGE> element.

  Returns:
    True if fields correspond to a valid article or redirect entry.
  """
  # If the namespace is not 0, then we are looking at something like an image
  # or a template, and do not want to further process it.
  if fields.get("ns") != "0":
    return False
  # We also want to check that the page has a title and text (if there is no
  # text then we are looking at some kind of upload).
  if not ("title" in fields and "text" in fields):
    return False
  return True


def generate_pages(xml_file):
  """Generates pages from Wikipedia XML file as dictionaries."""
  context = ElementTree.iterparse(xml_file, events=("start", "end"))
  context = iter(context)
  _, root = next(context)
  fields = {}
  depth = -1
  for event, element in context:
    ## Top-level logic
    if event == "start" and has_tag(element, "page"):
      fields = {}
      depth = 0
    elif event == "end" and has_tag(element, "page"):
      if validate_fields(fields):
        yield fields
      root.clear()  # Prevents memory issues.
    elif event == "start":
      depth += 1
    elif event == "end":
      depth -= 1
    ## Fields
    if event == "end" and has_tag(element, "title"):
      fields["title"] = element.text
    elif event == "end" and has_tag(element, "text"):
      fields["text"] = element.text
    elif event == "end" and has_tag(element, "redirect"):
      fields["redirect"] = element.attrib["title"]
    elif event == "end" and has_tag(element, "ns"):
      fields["ns"] = element.text
    # Using depth to ensure we get only the top-level page id, and not some
    # other id (like a revision id).
    elif event == "end" and has_tag(element, "id") and depth == 0:
      fields["id"] = int(element.text)


def has_tag(element, tag):
  """Detects whether an XML element has the given tag."""
  if element.tag != f"{XMLNS}{tag}":
    return False
  return True


def remove_element(text, element):
  """Removes double square bracket elements from the text.

  This gets rid of things that otherwise look like entities, e.g.:
    remove_element(text, 'File')
  removes all of the '[[File:...]]'s from the text.

  Args:
    text: The text to process.
    element: A string containing the type of element to remove, e.g., File,
      Image, etc.

  Returns:
    A string containing the processed text.
  """
  while f"[[{element}" in text:
    start = text.find(f"[[{element}")
    num_brackets = 0
    i = 0
    for char in text[start:]:
      if char == "[":
        num_brackets += 1
      elif char == "]":
        num_brackets -= 1
      if num_brackets == 0:
        break
      i += 1
    end = start + i + 1
    text = text[:start] + text[end:]
  return text


def process_wikitext(
    text,
    keep_tables = True,
    truncate = False,
):
  """Processes Wikitext to human readable text."""
  if text is None:
    return ""
  if RE_DISAMBIGUATION.search(text):
    return "DISAMBIGUATION"
  # Replace non-breaking spaces
  text = text.replace("&nbsp;", " ")
  # Remove tables
  if not keep_tables:
    text = RE_TABLE.sub("", text)
  # Remove templates
  n = 1
  while n:
    text, n = RE_TEMPLATE.subn("", text)
  # Remove comments
  n = 1
  while n:
    text, n = RE_COMMENT.subn("", text)
  # Remove references
  text = RE_REF.sub("", text)
  # Remove other HTML tags
  text = RE_HTML.sub("", text)
  # Remove elements from the text, e.g., files and images, that may get confused
  # with annotated mentions.
  text = remove_element(text, "File")
  text = remove_element(text, "file")
  text = remove_element(text, "Image")
  text = remove_element(text, "image")
  text = remove_element(text, "Category")
  text = remove_element(text, "category")
  # Remove bolding
  text = text.replace("'''", "")
  # Strip text
  text = text.strip()
  # If truncating return everything before first heading.
  if truncate:
    text = text.split("==")[0]
  return text


def split_tables(text):
  """Splits tables from the text if they exist."""
  tables = RE_TABLE.findall(text)
  text = RE_TABLE.sub("", text)
  return text, tables


RE_TAB_DELIM = re.compile(r"^(\{\||\|\}).*$", re.MULTILINE)
RE_TAB_CAPTION = re.compile(r"(?P<start>\|\+)(?P<caption>.+?)(?=[^\\](!|\|))",
                            re.DOTALL)
RE_TAB_STYLE = re.compile(r"[^\s!]+=.*?\|")
RE_TAB_ROW = re.compile(r"\|-")
RE_TAB_COL = re.compile(r"(\|\||!!|^\s*?!|^\s*?\|)", re.MULTILINE)
RE_WHITESPACE = re.compile(r"\s{2,}")


def process_table(table):
  """Processes table markup."""
  # Remove outer delimiters and styling
  table = RE_TAB_DELIM.sub("", table)
  table = RE_TAB_STYLE.sub("", table)

  # Detect optional caption
  caption = RE_TAB_CAPTION.search(table)
  if caption:
    caption_string = "[CAPTION] " + caption.group("caption")
    table = RE_TAB_CAPTION.sub("", table)
  else:
    caption_string = ""

  # Clean styling from start of table & split into rows
  rows = RE_TAB_ROW.split(table)

  # Detect optional header row (could be either of first two rows).
  header_string = ""
  row_start = 0
  for i, row in enumerate(rows[:2]):
    num_header_cells = len(re.findall("!", row))
    if num_header_cells > 0:
      header_string = "[HEADER] " + RE_TAB_COL.sub(" [COL] ", row)
      row_start = i + 1

  # Generate rows
  for row in rows[row_start:]:
    row_string = "[ROW] " + RE_TAB_COL.sub(" [COL] ", row)
    output_string = " ".join((caption_string, header_string, row_string))
    output_string = RE_WHITESPACE.sub(" ", output_string)
    yield output_string


def split_article(text):
  """Splits article into sections, with special logic for dealing with tables."""
  parents = []
  titles = ["INTRODUCTION"]
  contents = []
  prev_depth = 1
  start = 0
  for match in RE_TITLE.finditer(text):
    title = match.group("title").strip()
    depth = len(match.group("equals"))
    if prev_depth:
      pops = prev_depth - depth + 1
      if pops > 0:
        for _ in range(pops):
          try:
            parents.pop()
          except IndexError:
            continue
    parents.append(title)
    title = " - ".join(parents)
    titles.append(title)
    end = match.start()
    content = text[start:end]
    contents.append(content)
    start = match.end()
    prev_depth = depth
  if start:
    content = text[start:end]
    contents.append(content)
  for title, content in zip(titles, contents):
    content, tables = split_tables(content)
    yield title, content.strip(" \n\t")
    for i, table in enumerate(tables):
      for j, row in enumerate(process_table(table)):
        yield f"{title} - Table-{i}-{j}", row.strip(" \n\t")


def clean_wikilink(link):
  """Applies media wiki link formatting (e.g., stripping, capitalizing, etc.)"""
  clean_link = link.strip()
  try:
    clean_link = clean_link[0].upper() + clean_link[1:]
  except IndexError:
    return link
  clean_link = RE_BREADCRUMB.split(clean_link)[0]
  clean_link = clean_link.replace(" ", "_")
  return clean_link


def process_wikilinks(text):
  """Returns clean text and extracted wikilinks from wikitext."""
  clean_text = text
  links = []
  match = RE_MENTION.search(clean_text)
  while match:
    link = match.group("link")
    surface = match.group("surface")
    replacement = surface if surface else link
    start = match.start()
    end = match.start() + len(replacement)
    links.append({"id": clean_wikilink(link), "start": start, "end": end})
    try:
      clean_text = "".join(
          [clean_text[:match.start()], replacement, clean_text[match.end():]])
    except IndexError:
      raise Exception(f"text: {clean_text}, match: {match}")
    match = RE_MENTION.search(clean_text)
  return clean_text, links

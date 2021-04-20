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
"""Functions for logging debug info for use during model dev cycle."""

from absl import logging

from language.canine.tydiqa import data


def is_int_list(value):
  """Checks if a value's type is a list of integers."""
  return value and isinstance(value, list) and isinstance(value[0], int)


def is_unicode_list(value):
  """Checks if a value's type is a list of Unicode strings."""
  if value and isinstance(value, list):
    return isinstance(value[0], str)
  return False


def is_valid_unicode(text):
  """Check if a string is valid unicode. Did we slice on an invalid boundary?"""
  try:
    text.decode("utf-8")
    return True
  except UnicodeDecodeError:
    return False


def log_debug_info(filename, line_no, entry, debug_info, id_to_string_fn):
  """Logs `debug_info` for debugging purposes."""

  # Enable when debugging experimental new things.
  extremely_verbose = False

  def sanitize_char(c):
    """Optionally normalize chars we don't want in log messages."""
    # Don't like having too many newlines in your debugging log output?
    # Change this.
    remove_newlines = False
    if c == "\r":
      if remove_newlines:
        return " "
      return "\r"
    if c == "\n":
      if remove_newlines:
        return " "
      return "\n"
    return c

  def sanitize(s):
    return "".join(sanitize_char(c) for c in s)

  doc = entry["plaintext"]

  if "json" in debug_info:
    json_elem = debug_info["json"]
  else:
    json_elem = None
    logging.info("No 'json' key in `debug_info`.")

  if "tydi_example" in debug_info:
    tydi_example = debug_info["tydi_example"]
  else:
    tydi_example = None
    logging.info("No 'tydi_example' key in `debug_info`.")

  offset_to_wp = None
  doc_wp = None
  logging.info("=== Logging example %s:%d ===", filename, line_no)

  window = 20
  for i in range(0, data.byte_len(entry["contexts"]), window):
    span_text = data.byte_slice(
        entry["contexts"], i, i + window, errors="replace")
    doc_offsets = entry["context_to_plaintext_offset"][i:i + window]
    # Now double-check that those doc offsets actually match the text we expect.
    recovered_doc = [
        data.byte_slice(doc, i, i + 1, errors="replace")
        for i in doc_offsets
        if i != -1
    ]
    if extremely_verbose:
      logging.info("context_to_doc: %d: %s (%s) %s", i,
                   sanitize(span_text), " ".join(str(x) for x in doc_offsets),
                   sanitize(recovered_doc))

  for key, value in debug_info.items():
    if key == "offset_to_wp":
      offset_to_wp = value
      continue
    # Convert wordpiece vocab IDs back into readable text.
    if is_int_list(value) and "wp_ids" in key:
      value = [id_to_string_fn(word_id) for word_id in value]
    # Convert Unicode escapes to readable text.
    if is_unicode_list(value):
      value = [word.encode("utf-8") for word in value]

    if key == "all_doc_wp_ids":
      doc_wp = value

    # Represent lists as plaintext.
    if isinstance(value, list):
      value = " ".join(str(item) for item in value)
    value = str(value)
    logging.info("%s: %s", key, value)

  if offset_to_wp is not None:
    for i in range(0, data.byte_len(entry["contexts"]), window):
      wp_slice = []
      for byte_offset in range(i, i + window):
        if byte_offset in offset_to_wp:
          wp_offset = offset_to_wp[byte_offset]
          wp_slice.append(doc_wp[wp_offset])
        else:
          wp_slice.append("-1")
      context_slice = data.byte_slice(
          entry["contexts"], i, i + window, errors="replace")
      logging.info("context_to_wp: %d: %s (%s)", i, sanitize(context_slice),
                   " ".join(str(x) for x in wp_slice))

  if "searched_offset_to_wp" in debug_info:
    logging.info("searched_offset_to_wp: %s",
                 " ".join(str(i) for i in debug_info["searched_offset_to_wp"]))

  if json_elem:
    logging.info(
        "json.annotations[0].minimal_answer.plaintext_start_byte: %d",
        json_elem["annotations"][0]["minimal_answer"]["plaintext_start_byte"])
    logging.info(
        "json.annotations[0].minimal_answer.plaintext_end_byte: %d",
        json_elem["annotations"][0]["minimal_answer"]["plaintext_end_byte"])
    min_ans_sp = json_elem["annotations"][0]["minimal_answer"]
    min_ans_text = data.byte_slice(
        json_elem["document_plaintext"],
        min_ans_sp["plaintext_start_byte"],
        min_ans_sp["plaintext_end_byte"],
        errors="replace")
    min_ans_text_in_context = data.byte_slice(
        json_elem["document_plaintext"],
        min_ans_sp["plaintext_start_byte"] - 100,
        min_ans_sp["plaintext_end_byte"] + 100,
        errors="replace")
  logging.info("minimal answer text (from json): %s", min_ans_text)
  logging.info("minimal answer text in context: %s", min_ans_text_in_context)

  logging.info("entry.answer.span_start: %d", entry["answer"]["span_start"])
  logging.info("entry.answer.span_end: %d", entry["answer"]["span_end"])
  logging.info("entry.answer.span_text: %s", entry["answer"]["span_text"])
  if tydi_example:
    # Non-train examples may not have offsets.
    if tydi_example.start_byte_offset:
      logging.info("tydi_example.start_byte_offset: %d",
                   tydi_example.start_byte_offset)
      logging.info("tydi_example.end_byte_offset: %d",
                   tydi_example.end_byte_offset)
      tydi_example_min_ans_text = data.byte_slice(
          entry["contexts"],
          tydi_example.start_byte_offset,
          tydi_example.end_byte_offset,
          errors="replace")
      logging.info(
          "minimal answer text (from TyDiExample byte offsets in `contexts`): %s",
          tydi_example_min_ans_text)
  logging.info("^^^ End example ^^^")

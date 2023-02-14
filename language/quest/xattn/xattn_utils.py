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
"""Common constants and functions for xattn model."""

# Labels used for T5 output.
POS_LABEL = "relevant"
NEG_LABEL = "not relevant"

# Input format for T5.
INPUT_FORMAT = "query: {query}, doc: {doc}"


def get_example(
    query,
    doc_title,
    doc_title_to_text,
    spm_wrapper,
    is_relevant,
    context_size
):
  """Adds a tuple representing an example to `outputs`."""
  if doc_title not in doc_title_to_text:
    raise Exception("Missing document title: %s" % doc_title)

  doc_text = doc_title + " " + doc_title_to_text[doc_title]
  truncated_text = spm_wrapper.truncate(doc_text, context_size)
  input_string = INPUT_FORMAT.format(
      query=query, doc=truncated_text)
  output_string = (
      POS_LABEL if is_relevant else NEG_LABEL)
  return (input_string, output_string)

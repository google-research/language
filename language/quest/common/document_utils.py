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
"""Utilities for reading and writing documents files."""

import dataclasses

from language.quest.common import jsonl_utils


@dataclasses.dataclass(frozen=True)
class Document:
  """Represents a document with its title and text."""
  # Document title (should be unique in corpus).
  title: str
  # Document text.
  text: str


def read_documents(filepath, limit=None):
  documents_json = jsonl_utils.read(filepath, limit=limit, verbose=True)
  return [Document(**document) for document in documents_json]


def write_documents(filepath, documents):
  documents_json = [dataclasses.asdict(document) for document in documents]
  jsonl_utils.write(filepath, documents_json)

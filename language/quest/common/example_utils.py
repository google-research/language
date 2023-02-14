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
"""Utilities for reading and writing examples and predictions files."""

import dataclasses
import typing

from language.quest.common import jsonl_utils


@dataclasses.dataclass(frozen=False)
class ExampleMetadata:
  """Optional metadata used for analysis."""
  # The template used to synthesize this example.
  template: typing.Optional[str] = None
  # The domain of the example (e.g. films, books, plants, animals).
  domain: typing.Optional[str] = None
  # Fluency labels
  fluency: typing.Optional[typing.Sequence[bool]] = None
  # Meaning labels
  meaning: typing.Optional[typing.Sequence[bool]] = None
  # Naturalness labels
  naturalness: typing.Optional[typing.Sequence[bool]] = None
  # The following fields are dictionaries keyed by document title.
  # The sequences can contain multiple values for replicated annotations.
  relevance_ratings: typing.Optional[typing.Dict[str,
                                                 typing.Sequence[str]]] = None
  evidence_ratings: typing.Optional[typing.Dict[str,
                                                typing.Sequence[str]]] = None
  # The nested value is a map from query substring to document substring.
  attributions: typing.Optional[typing.Dict[str, typing.Sequence[typing.Dict[
      str, str]]]] = None


@dataclasses.dataclass(frozen=False)
class Example:
  """Represents a query paired with a set of documents."""
  query: str
  docs: typing.Iterable[str]
  # TODO(cmalaviya,petershaw): Move this to `metadata`.
  original_query: typing.Optional[str] = None
  # Scores can be optionally included if the examples are generated from model
  # predictions. Indexes of `scores` should correspond to indexes of `docs`.
  scores: typing.Optional[typing.Iterable[float]] = None
  # Optional metadata.
  metadata: typing.Optional[ExampleMetadata] = None


def read_examples(filepath):
  examples_json = jsonl_utils.read(filepath)
  examples = [Example(**example) for example in examples_json]
  for example in examples:
    example.metadata = ExampleMetadata(**example.metadata)
  return examples


def write_examples(filepath, examples):
  examples_json = [dataclasses.asdict(example) for example in examples]
  jsonl_utils.write(filepath, examples_json)

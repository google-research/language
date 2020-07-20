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
"""Types for fever data."""

import dataclasses

Json = Dict[Text, Any]


# An evidence set contains a list of tuples, each representing one line
# of evidence
# First two ints are IDs competition runners use, then the wiki page, then the
# sentence number.
@dataclasses.dataclass
class Evidence:
  annotation_id: Optional[int]
  evidence_id: int
  # fever_identifier: not actually a url, but page title
  wikipedia_url: Optional[Text]
  sentence_id: Optional[int]


# This must go after Evidence, otherwise python cannot parse it
EvidenceSet = List[Evidence]
EvidenceFromJson = Tuple[Optional[int], int, Optional[Text], Optional[int]]


@dataclasses.dataclass
class FeverMetrics:
  strict_score: float
  accuracy_score: float
  precision: float
  recall: float
  f1: float
  n_examples: int

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
"""Utilities for identifying identical substrings in sources and targets."""

from language.compgen.nqg.model.qcfg import qcfg_rule


def _in_matched_range(start_idx, end_idx, matched_ranges):
  """Return True if provided indices overlap any spans in matched_ranges."""
  for range_start_idx, range_end_idx in matched_ranges:
    if not (end_idx <= range_start_idx or start_idx >= range_end_idx):
      return True
  return False


def _find_exact_matches(source, target):
  """Returns longest non-overlapping sub-strings shared by source and target."""
  source_len = len(source)
  target_len = len(target)

  matches = set()
  matched_source_ranges = set()
  matched_target_ranges = set()
  for sequence_len in range(max(target_len, source_len), 0, -1):
    for source_start_idx in range(0, source_len - sequence_len + 1):
      source_end_idx = source_start_idx + sequence_len
      if _in_matched_range(source_start_idx, source_end_idx,
                           matched_source_ranges):
        continue
      for target_start_idx in range(0, target_len - sequence_len + 1):
        target_end_idx = target_start_idx + sequence_len
        if _in_matched_range(target_start_idx, target_end_idx,
                             matched_target_ranges):
          continue

        source_span = source[source_start_idx:source_end_idx]
        target_span = target[target_start_idx:target_end_idx]

        if source_span == target_span:
          matches.add(tuple(source_span))
          matched_source_ranges.add((source_start_idx, source_end_idx))
          matched_target_ranges.add((target_start_idx, target_end_idx))

  return matches


def get_exact_match_rules(dataset):
  """Return set of rules for terminal sequences in both source and target."""

  matches = set()
  for source_str, target_str in dataset:
    source = source_str.split()
    target = target_str.split()
    matches.update(_find_exact_matches(source, target))

  exact_match_rules = set()
  for match in matches:
    rule = qcfg_rule.QCFGRule(source=tuple(match), target=tuple(match), arity=0)
    exact_match_rules.add(rule)

  return exact_match_rules

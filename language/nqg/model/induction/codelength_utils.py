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
"""Utilities for computing codelengths over QCFG rules."""

import collections
import math
import random

from language.nqg.model.induction import rule_utils

from language.nqg.model.qcfg import qcfg_parser
from language.nqg.model.qcfg import qcfg_rule


def rule_codelength(rule, config):
  """Computes the codelength for a given rule."""
  length = 0.0
  for token in rule.source + rule.target:
    if token in {qcfg_rule.NT_1, qcfg_rule.NT_2}:
      length += config.non_terminal_codelength
    else:
      length += config.terminal_codelength
  return length


def _aggregate_counts(child_counts):
  """Return aggregated node count as int."""
  if not child_counts:
    return 1
  elif len(child_counts) == 1:
    return child_counts[0]
  elif len(child_counts) == 2:
    return child_counts[0] * child_counts[1]
  else:
    raise ValueError


def _get_num_all_derivations(source, rules, verbose):
  """Return total number of derivations for any target."""

  def node_fn(unused_span_begin, unused_span_end, unused_rule, children):
    """Represent nodes as integer counts of possible derivations."""
    return _aggregate_counts(children)

  def postprocess_fn(nodes):
    """Merge and sum all nodes."""
    return [sum(nodes)]

  outputs = qcfg_parser.parse(
      source,
      rules,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_fn,
      verbose=verbose)

  if len(outputs) != 1:
    raise ValueError
  num_outputs = outputs[0]
  return num_outputs


def _get_num_target_derivations(source, target, rules, verbose):
  """Return number of derivations of target."""
  goal_target_string = " ".join(target)

  def node_fn(unused_span_begin, unused_span_end, rule, children):
    """Represent nodes as (target string, int count of possible derivations)."""
    target_strings = [target_string for target_string, _ in children]
    new_target_string = qcfg_rule.apply_target(rule, target_strings)

    child_counts = [child_count for _, child_count in children]

    count = _aggregate_counts(child_counts)

    return (new_target_string, count)

  def postprocess_fn(nodes):
    """Discard nodes that cannot reach goal and aggregate counts."""
    counts_dict = collections.defaultdict(int)
    for target_string, count in nodes:
      # Discard any targets that are not substrings of goal target.
      if target_string not in goal_target_string:
        continue
      counts_dict[target_string] += count
    return [
        (target_string, count) for target_string, count in counts_dict.items()
    ]

  outputs = qcfg_parser.parse(
      source,
      rules,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_fn,
      verbose=verbose)

  for target_string, count in outputs:
    if target_string == goal_target_string:
      return count

  raise ValueError("No target derivation for example (%s, %s)" %
                   (source, target))


def _target_codelength(source, target, rules, verbose=False):
  """Return codelength for encoding `target` given `source` and `rules`.

  The codelength of the target is computed as -log_2(P(y|x)).

  For P(y|x) we use a naive uniform distribution over derivations, such that:

  P(y|x) = # of derivations of <x,y> / # of derivations of <x,?>,

  where ? is any target strings.

  We therefore run a QCFG parser twice to determine the numberator and
  denominator counts.

  Args:
    source: Tuple of source tokens.
    target: Tuple of target tokens.
    rules: Set of QCFGRule instances.
    verbose: Print debug logging if True.

  Returns:
    Float representing codelength for encoding `target` given `source` and
    `rules`.
  """
  num_derivations = _get_num_all_derivations(source, rules, verbose=verbose)
  num_target_derivations = _get_num_target_derivations(
      source, target, rules, verbose=verbose)

  # Note log(B/A) = -log(A/B).
  codelength = math.log2(float(num_derivations) / float(num_target_derivations))
  if verbose:
    print("(%s, %s): %s derivations, %s target derivations, %s codelength" %
          (source, target, num_derivations, num_target_derivations, codelength))
  return codelength


def _find_relevant_examples(dataset, rule):
  """Find examples in `dataset` where `rule` could be used in a derivation."""
  # TODO(petershaw): This could potentially be more effecient by pre-indexing
  # the dataset sources in a data structure such as a Trie.
  examples = []
  for source_str, target_str in dataset:
    source = source_str.split()
    target = target_str.split()
    if rule_utils.rhs_can_maybe_derive(rule.source, source):
      examples.append((source, target))
  return examples


def get_dataset_encoding_delta(sample_size,
                               examples,
                               current_rules,
                               candidate_rule_to_add,
                               candidate_rules_to_remove,
                               verbose=False):
  """Approximate increase in codelength to encode dataset."""
  # Make a copy of the ruleset and add/remove candidates.
  new_rules = current_rules.copy()
  for rule_to_remove in candidate_rules_to_remove:
    new_rules.remove(rule_to_remove)
  new_rules.add(candidate_rule_to_add)

  relevant_examples = _find_relevant_examples(examples, candidate_rule_to_add)
  num_relevant_examples = len(relevant_examples)
  sample = False

  if verbose:
    print("%s relevant rules." % num_relevant_examples)

  # If configured, sample rules for effeciency.
  if sample_size and num_relevant_examples > sample_size:
    random.shuffle(relevant_examples)

    relevant_examples = relevant_examples[:sample_size]
    sample = True

  # Compute the increase in target codelength summed across the sample.
  delta = 0
  for source, target in relevant_examples:
    new_codelength = _target_codelength(
        source, target, new_rules, verbose=verbose)
    original_codelength = _target_codelength(
        source, target, current_rules, verbose=verbose)
    delta += (new_codelength - original_codelength)

  # Estimate delta across entire set based on our sample.
  if sample:
    scale_factor = float(num_relevant_examples) / float(sample_size)
    delta *= scale_factor
    if verbose:
      print("Scaling delta by %s." % scale_factor)

  if verbose:
    print("Delta: %s." % delta)
  return delta

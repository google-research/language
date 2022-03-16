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
"""Utilities for computing grammar induction objective.

The objective has two terms which both decompose over the rules in the grammar:
(1) The "size" of each grammar rule, based on the number of terminal and
nonterminal tokens.
(2) A "penalty" for each grammar rule, based on how well it fits the given
examples.
"""

import math
import random

from language.compgen.csl.induction import rule_utils
from language.compgen.csl.qcfg import qcfg_rule


def _get_rule_size(rule, config):
  """Returns the "size" based on number of terminal and nonterminal tokens."""
  size = 0.0
  for token in rule.source + rule.target:
    if qcfg_rule.is_nt(token):
      size += config["non_terminal_coef"]
    else:
      size += config["terminal_coef"]
  return size


def _get_rule_penalty(rule, config, examples, verbose=False):
  """Returns a "penalty" for the rule based on the given set of examples."""
  # TODO(petershaw): This could potentially be more effecient by pre-indexing
  # the examples in a data structure such as a Trie.
  # TODO(petershaw): Could also consider sub-sampling the dataset for the
  # purpose of computing these correlations.

  # Optionally compute over a sample of examples only.
  sample_size = config.get("sample_size", 0)

  num_examples_match_source = 0
  num_examples_match_target = 0
  num_examples_match_source_and_target = 0
  for source_str, target_str in examples:
    source = tuple(source_str.split())
    target = tuple(target_str.split())
    match_source = rule_utils.rhs_can_maybe_derive(rule.source, source)
    match_target = rule_utils.rhs_can_maybe_derive(rule.target, target)
    if match_source:
      num_examples_match_source += 1
    if match_target:
      num_examples_match_target += 1
    if match_source and match_target:
      num_examples_match_source_and_target += 1

    if sample_size and num_examples_match_source_and_target >= sample_size:
      # Break early if using sample size and found sufficient sample.
      break

  # Ensure that at least one example is found.
  if not num_examples_match_source_and_target:
    print("Rule did not match any examples.")
    # TODO(petershaw): Raise instead?
    return 0.0
  if not num_examples_match_source:
    raise ValueError("num_examples_match_source: %s" %
                     num_examples_match_source)
  if not num_examples_match_target:
    raise ValueError("num_examples_match_target: %s" %
                     num_examples_match_target)

  if verbose:
    print("_get_rule_cost: %s" % rule)
    print("num_examples_match_source: %s" % num_examples_match_source)
    print("num_examples_match_target: %s" % num_examples_match_target)
    print("num_examples_match_source_and_target: %s" %
          num_examples_match_source_and_target)

  cost = 0.0
  p_source_given_target = (
      float(num_examples_match_source_and_target) / num_examples_match_target)
  cost -= (
      config["source_given_target_coef"] * math.log2(p_source_given_target))
  p_target_given_source = (
      float(num_examples_match_source_and_target) / num_examples_match_source)
  cost -= (
      config["target_given_source_coef"] * math.log2(p_target_given_source))
  return cost


class ObjectiveCalculator(object):
  """Computes and caches objective computations."""

  def __init__(self, examples, config):
    self.examples = examples
    self.config = config
    # Dictionaries to cache repeated computation.
    self.rule_to_size = {}
    self.rule_to_penalty = {}
    # Shuffle examples if using sampling.
    if config.get("sample_size", 0):
      random.shuffle(self.examples)

  def _get_rule_size(self, rule):
    """Computes the size for a given rule."""
    if rule not in self.rule_to_size:
      self.rule_to_size[rule] = _get_rule_size(rule, self.config)
    return self.rule_to_size[rule]

  def _get_rule_penalty(self, rule):
    """Computes the penalty for a given rule."""
    if rule not in self.rule_to_penalty:
      self.rule_to_penalty[rule] = _get_rule_penalty(rule, self.config,
                                                     self.examples)
    return self.rule_to_penalty[rule]

  def get_candidate_size_delta(self, candidate_rule, rules_to_remove):
    """Computes the delta in grammar size."""
    delta = 0.0
    delta -= self._get_rule_size(candidate_rule)
    for rule in rules_to_remove:
      delta += self._get_rule_size(rule)
    return delta

  def get_candidate_penalty_delta(self, candidate_rule, rules_to_remove):
    """Computes delta based on rule penalties."""
    delta = 0.0
    delta -= self._get_rule_penalty(candidate_rule)
    for rule in rules_to_remove:
      delta += self._get_rule_penalty(rule)
    return delta

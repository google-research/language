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
"""Utility to sample sources and targets in induced QCFG."""

import random

from language.compgen.csl.model.data import parsing_utils
from language.compgen.csl.qcfg import qcfg_rule


def _convert_to_qcfg(nested_rule):
  """Convert nested JointRule to QCFG source and target."""
  sources = []
  targets = []
  rule = nested_rule[0]
  idx_to_source = {}
  idx_to_target = {}
  for nt_idx, child_rule in enumerate(nested_rule[1:]):
    source, target = _convert_to_qcfg(child_rule)
    idx_to_source[nt_idx + 1] = source
    idx_to_target[nt_idx + 1] = target
  for symbol in rule.source:
    if qcfg_rule.is_nt_fast(symbol):
      index = qcfg_rule.get_nt_index(symbol)
      sources.extend(idx_to_source[index])
    else:
      sources.append(symbol)
  for symbol in rule.target:
    if qcfg_rule.is_nt_fast(symbol):
      index = qcfg_rule.get_nt_index(symbol)
      targets.extend(idx_to_target[index])
    else:
      targets.append(symbol)
  return sources, targets


def _uniform_score_fn(unused_parent_rule, unused_nt_idx, unused_child_rule):
  return 1


class QCFGSampler(object):
  """Class to sample sources and targets."""

  def __init__(self, qcfg_rules, max_recursion=10, min_recursion=1):
    """Sample source and target in both QCFG and target CFG.

    Args:
      qcfg_rules: A list of QCFGRule instance.
      max_recursion: Attempt to limit the derivation tree depth to this number.
        There are cases where this number may not be a strict bound, e.g. if
        certain nonterminals cannot be expanded by a rule with no RHS
        nonterminals.
      min_recursion: Minimum recursion depth.
    """
    # Dict of NT symbol to joint_rule.
    self.nt_rules = []
    self.t_rules = []

    for rule in qcfg_rules:
      if rule.arity > 0:
        self.nt_rules.append(rule)
      else:
        self.t_rules.append(rule)

    self.max_recursion = max_recursion
    self.min_recursion = min_recursion

  def __str__(self):
    rep_str = ["nt_rules: %s" % self.nt_rules]
    rep_str.append("t_rules: %s" % self.t_rules)
    return "\n".join(rep_str)

  def __repr__(self):
    return self.__str__()

  def _sample_rule(self, score_fn, parent_rule, nt_idx, recursions):
    """Sample a rule."""
    rules = list(self.nt_rules) + list(self.t_rules)
    weights = [score_fn(parent_rule, nt_idx, rule) for rule in rules]

    if recursions >= self.max_recursion and self.nt_rules:
      weights = [0] * len(self.nt_rules) + weights[len(self.nt_rules):]
    if recursions < self.min_recursion and self.t_rules:
      weights = weights[:len(self.nt_rules)] + [0] * len(self.t_rules)
    return random.choices(rules, weights=weights)[0]

  def _expand(self, score_fn, parent_rule, nt_idx, recursions):
    """Recursively expand `nt_symbol`."""
    rule = self._sample_rule(score_fn, parent_rule, nt_idx, recursions)
    outputs = [rule]
    for nt_idx in range(rule.arity):
      outputs.append(self._expand(score_fn, rule, nt_idx, recursions + 1))
    return outputs

  def sample(self, score_fn=None):
    """Sample a source and target pair.

    Args:
      score_fn: A scoring function that takes parent_rule (QCFGRule), nt_idx,
        child_rule (QCFGRule) and returns s score.

    Returns:
      A tuple of source tokens and target tokens.
    """
    if not score_fn:
      score_fn = _uniform_score_fn
    root_rule = parsing_utils.ROOT_RULE_KEY
    outputs = self._expand(score_fn, root_rule, 0, recursions=0)
    sources, targets = _convert_to_qcfg(outputs)
    return sources, targets

  def save(self, filename):
    raise NotImplementedError("Cannot save QCFGSampler.")

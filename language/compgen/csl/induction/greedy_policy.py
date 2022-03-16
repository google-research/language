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
"""Policy for selecting rules to add or remove."""

import collections

from absl import logging

from language.compgen.csl.induction import action_utils
from language.compgen.csl.induction import derivation_utils
from language.compgen.csl.induction import objective_utils
from language.compgen.csl.induction import rule_utils
from language.compgen.csl.induction import unification_utils
from language.compgen.csl.qcfg import qcfg_target_parser


def _find_relevant_rules(current_rules, candidate_rule):
  # TODO(petershaw): This can potentially be made more effecient by
  # pre-indexing rules in a data structure such as a Trie.
  relevant_rules = []
  for rule in current_rules:
    if (rule_utils.rhs_can_maybe_derive(candidate_rule.source, rule.source) and
        rule_utils.rhs_can_maybe_derive(candidate_rule.target, rule.target)):
      relevant_rules.append(rule)
  return relevant_rules


def _find_possible_candidates(rule_to_split, other_rules, config):
  """Return possible rule candidates."""
  all_candidates = set()
  for other_rule in other_rules:
    if other_rule == rule_to_split:
      continue
    if (rule_utils.rhs_can_maybe_derive(other_rule.source, rule_to_split.source)
        and rule_utils.rhs_can_maybe_derive(other_rule.target,
                                            rule_to_split.target)):
      unifiers = unification_utils.get_rule_unifiers(rule_to_split, other_rule,
                                                     config)
      candidates = {(unifier, other_rule) for unifier in unifiers}
      all_candidates |= candidates
  return all_candidates


def _find_additional_rules_to_remove(seed_rules, current_rules, candidate_rule,
                                     config):
  """Return other rules that `candidate_rule` enables removing."""
  # TODO(petershaw): This is an *extremely* ineffecient way of computing this.
  relevant_rules = _find_relevant_rules(current_rules, candidate_rule)
  # TODO(petershaw): Remove this hack after implementing a more effecient
  # implementation. For now, consider only the 10 shortest relevant rules.
  # TODO(linluqiu): This introduces randomness to grammar induction, which
  # seems to mostly affect SCAN MCD3 split. Might need to re-evaluate this.
  if len(relevant_rules) > 10:
    relevant_rules = sorted(
        relevant_rules, key=lambda x: len(x.source) + len(x.target))[:10]
  additional_rules_to_remove = set()
  for rule in relevant_rules:
    if rule in seed_rules:
      continue
    candidates = unification_utils.get_rule_unifiers(rule, candidate_rule,
                                                     config)
    for existing_rule in candidates:
      if existing_rule in current_rules:
        additional_rules_to_remove.add(rule)
  return additional_rules_to_remove


class GreedyPolicy(object):
  """Policy that greedily selects actions to add."""

  def __init__(self,
               config,
               examples,
               seed_rules,
               target_grammar_rules,
               verbose=False):
    self.config = config
    self.examples = examples
    self.seed_rules = seed_rules
    self.target_checker = None
    if target_grammar_rules:
      self.target_checker = qcfg_target_parser.TargetChecker(
          target_grammar_rules)
    self.objective_calculator = objective_utils.ObjectiveCalculator(
        self.examples, self.config)
    self.verbose = verbose

  def select_action(self, sampled_rule, current_rules):
    """Select action that maximizes improvememt in objective."""
    current_rules = set(current_rules)
    if self.verbose:
      logging.info("Sampled rule: %s, current rules: %d", str(sampled_rule),
                   len(current_rules))

    # Can optionally skip identity rules.
    can_split_seed_rules = self.config.get("can_split_seed_rules", False)
    if (sampled_rule in self.seed_rules and not can_split_seed_rules):
      return None

    # Check if this rule can be derived by the current set of rules.
    derivation_rules = derivation_utils.generate_derivation(
        self.config, sampled_rule, current_rules)
    if derivation_rules:
      action = action_utils.Action(
          rules_to_add=set(), rules_to_remove={sampled_rule})
      return action

    # The "delta" refers to the improvement in grammar induction objective.
    # If no action has a positive delta, then return None.
    max_action = None
    max_delta = 0.0

    # Determine candidate rules to add.
    candidates = _find_possible_candidates(
        sampled_rule,
        current_rules,
        self.config,
    )

    if candidates:
      # Map of rule candidates to set of rules that they enable removing.
      candidates_to_rules_to_remove = collections.defaultdict(set)

      # Dict of rule candidates to the change in objective.
      candidates_to_delta = {}

      for candidate_rule, _ in candidates:
        # Check whether candidate's target can be derived by target grammar.
        if self.target_checker:
          if not self.target_checker.can_parse(candidate_rule.target):
            if self.verbose:
              logging.info("Skipped rule not in target grammar: %s.",
                           str(candidate_rule))
            continue
        candidates_to_rules_to_remove[candidate_rule].add(sampled_rule)
        # For each candidate, attempt to find any additional rules that can be
        # removed.
        additional_rules = _find_additional_rules_to_remove(
            self.seed_rules, current_rules, candidate_rule, self.config)
        candidates_to_rules_to_remove[candidate_rule] |= additional_rules

        # Compute the first term in the objective, the change in grammar size.
        candidates_to_delta[
            candidate_rule] = self.objective_calculator.get_candidate_size_delta(
                candidate_rule, candidates_to_rules_to_remove[candidate_rule])

      # Sort candidates by size reduction prior to computing the
      # delta for rule penalities, which is relatively more
      # expensive. Use lexical ordering to break ties.
      candidates_to_delta_sorted = sorted(
          candidates_to_delta.items(), key=lambda x: (-x[1], x[0]))

      # For debugging, log up to the top 15 candidates.
      if self.verbose:
        logging.info("Candidate rules:")
        for rule, delta in candidates_to_delta_sorted[:15]:
          logging.info("%s (%s)", rule, delta)

      for candidate_rule, delta in candidates_to_delta_sorted:
        if delta <= max_delta:
          # Because the penalty delta is <= 0, we can safely break here as
          # there is no way for delta to increase above max_delta.
          break
        rules_to_remove = candidates_to_rules_to_remove[candidate_rule]
        penalty_delta = 0.0
        if self.config.get("use_rule_penalty", True):
          penalty_delta = self.objective_calculator.get_candidate_penalty_delta(
              candidate_rule, rules_to_remove)
          if self.verbose:
            logging.info("Penalty delta for %s: %s", candidate_rule,
                         penalty_delta)

        # Now consider the full delta in the objective.
        delta += penalty_delta
        # Check if delta is greater than any previous action considered.
        if delta > max_delta:
          max_delta = delta
          max_action = action_utils.Action(
              rules_to_add={candidate_rule}, rules_to_remove=rules_to_remove)
    return max_action

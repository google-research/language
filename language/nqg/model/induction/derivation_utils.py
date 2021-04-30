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
"""Utilities for applying rules to produce derivations.

Note that in this module we will reuse the QCFGRule tuple to represent both
derived string pairs and QCFG rules. Since we only allow a single LHS
non-terminal, both concepts can be represented as a pair of source and target
sequences. Therefore, we abuse terminology and refer to each concept
interchangeably in certain contexts.
"""

from language.nqg.model.induction import rule_utils

from language.nqg.model.qcfg import qcfg_rule


def _substitute(rhs_a, rhs_b, nt=qcfg_rule.NT_1):
  """Replace nt in rhs_a with rhs_b, re-indexing non-terminals if needed."""
  output = []
  for token in rhs_a:
    if token == nt and nt == qcfg_rule.NT_2:
      # Our goal is to replace NT_2 in rhs_a with rhs_b, but we need to
      # do some re-indexing to avoid collisions.
      # First, we re-index NT_1 in rhs_b to NT_2.
      # Based on the logic in `apply`, if rhs_a has arity 2, then rhs_b
      # will have arity < 2, i.e. will not contain NT_2.
      rhs_b = rule_utils.rhs_replace(rhs_b, [qcfg_rule.NT_1], qcfg_rule.NT_2)
      # We can now safely replace NT_2 in rhs_a with rhs_b, which should
      # contain only NT_2.
      output.extend(rhs_b)
    elif token == nt:
      # Replace NT_1 in rhs_a with rhs_b.
      # Based on the logic in `apply`, no collisions on non-terminal indexes
      # should occur, since we should either be in the case:
      # 1. rhs_a only has NT_1, and rhs_b has NT_1 and NT_2
      # 2. rhs_a has NT_1 and NT_2, but rhs_b only has NT_1
      output.extend(rhs_b)
    else:
      output.append(token)
  return output


def _apply(rule_a, rule_b):
  """Applies rule_b to rule_a, returning set of derived rules."""
  outputs = []

  if rule_a.arity == 2:
    new_arity = 1 + rule_b.arity
    if new_arity <= 2:
      # Cannot apply an arity 2 rule to an arity 2 rule because this would lead
      # to a rule with 3 different non-terminal indexes, which is disallowed
      # by our QCFG conventions.
      source_0 = _substitute(rule_a.source, rule_b.source)
      target_0 = _substitute(rule_a.target, rule_b.target)
      outputs.append((source_0, target_0, new_arity))
      # Rule can potentially be applied to either non-terminal in rule_a.
      source_1 = _substitute(rule_a.source, rule_b.source, nt=qcfg_rule.NT_2)
      target_1 = _substitute(rule_a.target, rule_b.target, nt=qcfg_rule.NT_2)
      outputs.append((source_1, target_1, new_arity))

  elif rule_a.arity == 1:
    new_arity = rule_b.arity
    source = _substitute(rule_a.source, rule_b.source)
    target = _substitute(rule_a.target, rule_b.target)
    outputs.append((source, target, new_arity))

  output_rules = set()
  for source, target, arity in outputs:
    source, target = rule_utils.canonicalize_nts(source, target, arity)
    output_rules.add(qcfg_rule.QCFGRule(tuple(source), tuple(target), arity))

  return output_rules


def _can_maybe_derive_from(rule, goal_rule):
  """Return True if rule can potentially be used to derive goal_rule."""
  # Don't allow 'reflexive' derivations.
  if rule == goal_rule:
    return False

  if not rule_utils.rhs_can_maybe_derive(rule.source, goal_rule.source):
    return False

  if not rule_utils.rhs_can_maybe_derive(rule.target, goal_rule.target):
    return False

  return True


def _filter_rules(rules, goal_rule):
  return [rule for rule in rules if _can_maybe_derive_from(rule, goal_rule)]


def _verify_arity(rule):
  """Raise ValueError if rule does not follow valid arity convention."""
  if rule.arity == 0:
    if qcfg_rule.NT_1 in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
    if qcfg_rule.NT_2 in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
  elif rule.arity == 1:
    if qcfg_rule.NT_1 not in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
    if qcfg_rule.NT_2 in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
  elif rule.arity == 2:
    if qcfg_rule.NT_1 not in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
    if qcfg_rule.NT_2 not in rule.source:
      raise ValueError("Invalid rule: %s" % (rule,))
  return True


def can_derive(goal_rule,
               rules,
               derived_rules=None,
               max_iter=15,
               verbose=False):
  """Return True if `goal_rule` can be derived given `rules`.

  We perform a relatively naive breadth first search (BFS), with early pruning
  in cases where it can be quickly determined that an intermediate result
  cannot be used in a derivation of our goal.

  Args:
    goal_rule: A QCFGRule representing a string pair to derive.
    rules: A set of QCFGRules.
    derived_rules: If not None, will add any derived QCFGRules that can
      potentially derive `goal_rule` given some substitution to this set.
    max_iter: Maximum number of iterations (i.e. derivation depth) for
      attempting to derive `goal_rule`.
    verbose: Print debugging logging if True.

  Returns:
    True if `goal_rule` can be derived given `rules`.
  """
  # Filter rules to the set that can potentially be used in a derivation
  # of `goal_rule`.
  filtered_rules = _filter_rules(rules, goal_rule)
  if verbose:
    print("filtered_rules: %s" % filtered_rules)
  # Track seen rules.
  seen_rules = set(filtered_rules)
  # Set of derived rules with derivation depth equal to iteration.
  search_state = set(filtered_rules)
  for _ in range(max_iter):
    if not search_state:
      if verbose:
        print("Cannot derive %s." % str(goal_rule))
      return False
    if verbose:
      print("Starting next iteration with search_state:")
      for rule in search_state:
        print(rule)
    new_search_state = set()
    for rule_a in search_state:
      # Attempt to apply every relevant rule to every rule in search_state.
      for rule_b in filtered_rules:
        new_rules = _apply(rule_a, rule_b)
        if verbose:
          print("Applying %s to %s yields %s" % (rule_b, rule_a, new_rules))

        for new_rule in new_rules:
          # Check that application has not led to a malformed rule.
          _verify_arity(new_rule)
          if new_rule in seen_rules:
            continue
          seen_rules.add(new_rule)
          if goal_rule == new_rule:
            if verbose:
              print("Derived %s." % str(goal_rule))
            return True
          # If the generated rule can be potentially used in a derivation of
          # our goal, then add to the search state for the next iteration.
          if _can_maybe_derive_from(new_rule, goal_rule):
            if derived_rules is not None:
              derived_rules.add(new_rule)
            new_search_state.add(new_rule)
          else:
            if verbose:
              print("Cannot derive goal from: %s" % str(new_rule))
    search_state = new_search_state
  # For the datasets we have studied, this limit should not generally apply.
  print("Reached max iterations for rule `%s` given rules `%s`" %
        (goal_rule, filtered_rules))
  return False

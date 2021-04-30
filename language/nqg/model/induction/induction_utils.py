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
"""Utilities for grammar induction."""

import collections

from language.nqg.model.induction import codelength_utils
from language.nqg.model.induction import derivation_utils
from language.nqg.model.induction import exact_match_utils
from language.nqg.model.induction import rule_utils
from language.nqg.model.induction import split_utils

from language.nqg.model.qcfg import qcfg_parser
from language.nqg.model.qcfg import qcfg_rule

InductionConfig = collections.namedtuple("InductionConfig", [
    "sample_size",
    "max_iterations",
    "min_delta",
    "terminal_codelength",
    "non_terminal_codelength",
    "parse_sample",
    "allow_repeated_target_nts",
    "seed_exact_match",
    "balance_parens",
])

# We track the state of search during rule induction in the following tuple.
# Note that our implementation relies on two important aspects:
# 1. We can quickly determine if any substitution can potentially exist such
#    that a given rule can be used to derive a given string pair, based only
#    on terminal symbol overlap.
# 2. The set of derivable string pairs in our induced grammar is monotonically
#    increasing, based on our criteria for adding and removing rules.
SearchState = collections.namedtuple(
    "SearchState",
    [
        "current_rules",  # Set of rules in induced grammar.
        "rules_to_candidates",  # Dictionary of rules to candidates.
        "derivable_rules",  # Set of derivable rules.
    ])


def _find_affected_rules(rules, new_rule):
  # TODO(petershaw): This can potentially be made more effecient by
  # pre-indexing rules in a data structure such as a Trie.
  found_rules = []
  for rule in rules:
    if (rule_utils.rhs_can_maybe_derive(new_rule.source, rule.source) and
        rule_utils.rhs_can_maybe_derive(new_rule.target, rule.target)):
      found_rules.append(rule)
  return found_rules


def _has_balanced_parens(rhs):
  """Returns True if all '(' precede and are followed by a correspoding ')'."""
  open_parens = 0
  for token in rhs:
    for char in token:
      if char == "(":
        open_parens += 1
      elif char == ")":
        open_parens -= 1
      if open_parens < 0:
        return False
  return open_parens == 0


def _is_balanced_paren_candidate(rule):
  if not _has_balanced_parens(rule.source):
    return False
  if not _has_balanced_parens(rule.target):
    return False
  return True


def _filter_unbalanced_paren_candidates(rules):
  new_rules = set()
  for rule in rules:
    if _is_balanced_paren_candidate(rule):
      new_rules.add(rule)
  return new_rules


def _get_max_rule(search_state, config, examples):
  """Identify a rule to add that maximizes the decrease in codelength."""
  # Dict of rule candidates to the codelenth savings
  # (i.e. negative codelength delta).
  candidates_to_delta = {}
  # Map of rule candidates to the set of rules that they enable removing.
  # (inverse of rules_to_candidates).
  candidates_to_rules = collections.defaultdict(set)

  for rule in search_state.current_rules:
    candidates = search_state.rules_to_candidates[rule]
    for candidate in candidates:
      if candidate not in candidates_to_delta:
        # Subtract cost of new rule if not already accounted for.
        candidates_to_delta[candidate] = -codelength_utils.rule_codelength(
            candidate, config)

      # Add cost of every possible removed rule.
      candidates_to_delta[candidate] += codelength_utils.rule_codelength(
          rule, config)
      candidates_to_rules[candidate].add(rule)

  # Sort candidates by codelength reduction (prior to computing the codelength
  # delta of the dataset encoding, which is relatively more expensive).
  # Use lexical ordering to break ties.
  candidates_to_delta_sorted = sorted(
      candidates_to_delta.items(), key=lambda x: (-x[1], x[0]))

  # For debugging, print up to the top 15 candidates.
  print("Candidate rules:")
  for rule, delta in candidates_to_delta_sorted[:15]:
    print("%s (%s)" % (rule, delta))

  min_delta = config.min_delta
  max_rule_to_add = None
  max_rules_to_remove = None
  for rule, delta in candidates_to_delta_sorted:
    if delta <= min_delta:
      break
    rules_to_remove = candidates_to_rules[rule]
    targets_delta = codelength_utils.get_dataset_encoding_delta(
        sample_size=config.parse_sample,
        examples=examples,
        current_rules=search_state.current_rules,
        candidate_rule_to_add=rule,
        candidate_rules_to_remove=rules_to_remove)

    print("Targets encoding delta for %s: -%s" % (rule, targets_delta))

    # Compute the full deta including both the codelength reduction of encoding
    # the grammar (previously computed) and the codelength delta of encoding
    # the targets with the new grammar.
    delta -= targets_delta
    if delta > min_delta:
      min_delta = delta
      max_rule_to_add = rule
      max_rules_to_remove = rules_to_remove
  return max_rule_to_add, max_rules_to_remove


def _update_state(affected_rules, search_state, config):
  """Sparsely update the state for rules that may be affected."""
  for idx, affected_rule in enumerate(affected_rules):
    # Debug logging every Nth rule.
    if idx % 10 == 0:
      print("Updating rule %s of %s." % (idx + 1, len(affected_rules)))

    # Check if rule can now be generated. Ideally, this should have been
    # determined upstream when determining which rules could be removed,
    # but some cases are not caught until here, such as when source
    # sequences contain repeated substrings and are therefore not considered
    # by `get_candidates`.
    # Regardless, it is still important to run this for the side-effect of
    # updating the set of derivable rules.
    if derivation_utils.can_derive(affected_rule, search_state.current_rules,
                                   search_state.derivable_rules):
      print("Can now generate: %s." % str(affected_rule))
      search_state.current_rules.remove(affected_rule)
    else:
      candidates = split_utils.find_possible_splits(
          affected_rule,
          search_state.derivable_rules,
          allow_repeated_target_nts=config.allow_repeated_target_nts,
      )
      if config.balance_parens:
        candidates = _filter_unbalanced_paren_candidates(candidates)
      for candidate in candidates:
        search_state.rules_to_candidates[affected_rule].add(candidate)
  print("Updates complete.")


def _induce_rules_for_examples(examples, seed_rules, config):
  """Iteratively searches for rules to optimize codelength objective."""
  # Initialize the search state.
  search_state = SearchState(
      current_rules=seed_rules,
      rules_to_candidates=collections.defaultdict(set),
      derivable_rules=seed_rules.copy())

  # Update state for all seed rules.
  _update_state(seed_rules, search_state, config)

  # Iteratively update grammar.
  for iteration_num in range(config.max_iterations):
    print("Iteration %s." % iteration_num)
    rule, rules_to_remove = _get_max_rule(search_state, config, examples)

    # Break if there is no candidate that improves codelength objective.
    if rule is None:
      print("Breaking as no candidate exceeds minimum threshold.")
      break

    # Otherwise, update the set of rules.
    print("Adding rule: %s" % str(rule))
    search_state.current_rules.add(rule)
    search_state.derivable_rules.add(rule)
    for rule_to_remove in rules_to_remove:
      print("Removing rule: %s" % str(rule_to_remove))
      search_state.current_rules.remove(rule_to_remove)
      del search_state.rules_to_candidates[rule_to_remove]
    print("Number of current_rules: %s" % len(search_state.current_rules))

    # Update the search state based on any potentially affected rules.
    # The set of affected rules includes any rule that the added rule
    # may potentially be used in a derivation for.
    affected_rules = _find_affected_rules(search_state.current_rules, rule)
    _update_state(affected_rules, search_state, config)

  # Return the set of induced rules.
  return search_state.current_rules


def _example_to_rule(source_str, target_str):
  """Convert (source, target) example to a QCFGRule."""
  return qcfg_rule.QCFGRule(
      tuple(source_str.split()), tuple(target_str.split()), arity=0)


def _get_rules_for_other_examples(induced_rules, other_examples):
  """Add rules for examples outside of sample that cannot be derived."""
  new_rules = set()
  for source_str, target_str in other_examples:
    goal_rule = qcfg_rule.QCFGRule(
        tuple(source_str.split()), tuple(target_str.split()), arity=0)
    if not derivation_utils.can_derive(goal_rule, induced_rules, None):
      new_rules.add(goal_rule)
  print("Added %s rules for examples outside of initial sample." %
        len(new_rules))
  return new_rules


def _split_examples(examples, config):
  """Split examples into a sampled and a remaining subset based on config."""
  # Only consider unique examples.
  # TODO(petershaw): Consider preserving the number of occurences for each
  # unique example to better weight sampling for computing the dataset encoding
  # codelength.
  examples = list(set([tuple(example) for example in examples]))
  if config.sample_size:
    # Sort by number of input tokens.
    examples_sorted = sorted(examples, key=lambda x: len(x[0].split()))
    examples_sample = examples_sorted[:config.sample_size]
    examples_other = examples_sorted[config.sample_size:]
  else:
    examples_sample = examples
    examples_other = []
  return examples_sample, examples_other


def induce_rules(examples, config):
  """Return set of induced rules for a given set of examples."""
  # For effeciency, we only run grammar induction a subset of examples based
  # on the sample size specified in the config.
  examples_sample, examples_other = _split_examples(examples, config)

  # Initialize with a rule for each example.
  seed_rules = set()
  for source_str, target_str in examples_sample:
    seed_rules.add(_example_to_rule(source_str, target_str))
  print("Added %s seed rules for examples." % len(seed_rules))

  # Optionally add exact match rules.
  if config.seed_exact_match:
    seed_rules |= exact_match_utils.get_exact_match_rules(examples_sample)
  print("Seed rules after adding exact match rules for sampled examples: %s." %
        len(seed_rules))

  # Iteratively induce rules over the sampled set of examples.
  induced_rules = _induce_rules_for_examples(examples_sample, seed_rules,
                                             config)
  print("Induced %s rules from sample of %s examples." %
        (len(induced_rules), len(examples_sample)))

  # Verify that induced grammar can derive all examples in examples_sample.
  # We use the QCFG parser rather than `derivation_utils` as it is typically
  # faster when we do not need to consider non-terminals in the goal strings,
  # and to verify consistency of the algorithms.
  for source_str, target_str in examples_sample:
    if not qcfg_parser.can_parse(source_str, target_str, induced_rules):
      raise ValueError("Induced rules cannot parse example: (%s, %s)" %
                       (source_str, target_str))

  print("Checking %s remaining examples." % len(examples_other))
  # Add rules for any examples that were not in the original sample and cannot
  # be derived by the induced set of rules.
  if examples_other:
    if config.seed_exact_match:
      induced_rules |= exact_match_utils.get_exact_match_rules(examples_other)
    print("Rules after exact match for remaining examples: %s" %
          len(induced_rules))
    for source_str, target_str in examples_other:
      if not qcfg_parser.can_parse(source_str, target_str, induced_rules):
        induced_rules.add(_example_to_rule(source_str, target_str))
    print("Rules after adding rules for unparsable remaining examples: %s" %
          len(induced_rules))

  return induced_rules

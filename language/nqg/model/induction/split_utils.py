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
"""Utilities for identifying candidate rules."""

from language.nqg.model.induction import rule_utils

from language.nqg.model.qcfg import qcfg_rule

# Non-terminal with temporary index that is gauranteed to be unused in the
# current rule. This should be replaced with NT_1 or NT_2 to form a valid
# QCFGRule.
NT_TMP = "NT_?"


def _get_non_terminals(rhs):
  """Return set of non-terminal symbols in `rhs`."""
  non_terminals = set()
  for symbol in rhs:
    if symbol in (qcfg_rule.NT_1, qcfg_rule.NT_2, NT_TMP):
      non_terminals.add(symbol)
  return non_terminals


def _get_tmp_nt_replacement(nts):
  if nts == {NT_TMP}:
    return qcfg_rule.NT_1
  elif nts == {NT_TMP, qcfg_rule.NT_1}:
    return qcfg_rule.NT_2
  elif nts == {NT_TMP, qcfg_rule.NT_2}:
    return qcfg_rule.NT_1
  else:
    raise ValueError("Unexpected NTs: %s" % nts)


def _replace_tmp_nt(source, target, nts):
  new_nt = _get_tmp_nt_replacement(nts)
  source = rule_utils.rhs_replace(source, [NT_TMP], new_nt)
  target = rule_utils.rhs_replace(target, [NT_TMP], new_nt)
  return source, target


def _make_rule(nts, source, target):
  """Canoncalize NT indexes and return QCFGRule."""
  arity = len(nts)
  source, target = rule_utils.canonicalize_nts(source, target, arity)
  return qcfg_rule.QCFGRule(tuple(source), tuple(target), arity)


def _maybe_get_candidate_pair(source_g, source_h, target_g, target_h):
  """Returns candidate rule pair if proposed sources and targets are valid."""
  # Check that proposed sources and targets contain same non-terminal indexes.
  nts_g = _get_non_terminals(source_g)
  if nts_g != _get_non_terminals(target_g):
    return None

  nts_h = _get_non_terminals(source_h)
  if nts_h != _get_non_terminals(target_h):
    return None

  # Canonicalize non-terminal index ordering and return candidate pair.
  source_g, target_g = _replace_tmp_nt(source_g, target_g, nts_g)
  rule_g = _make_rule(nts_g, source_g, target_g)
  rule_h = _make_rule(nts_h, source_h, target_h)
  return (rule_g, rule_h)


def _get_split_candidates(rule, allow_repeated_target_nts=True):
  """Implements `SPLIT` procedure described in paper appendix.

  To explain this function, let us review some notation for SCFGs/QCFGs.
  Let `g` and `h` refer to QCFG rules. Let `=>_g` denote the application of
  rule g, such that <a,b> `=>_g` <c,d> means
  that <c,d> can be generated from <a,b> by applying the rule `g` to replace
  some indexed non-terminal in <a,b>. Let
  `=>_g =>_h` refer to a chain of rule applications of `g` and `h`, ommiting
  the intermediate rule pair.

  We can now define the behavoir of this function. Let `NT -> <a,b>` refer to
  the input argument `rule`. The function returns the following set:

  {(g,h) | <NT,NT> =>_g =>_h <a,b>}

  In other words, we return pairs of rules that can generate the input `rule`.
  We leave it to the caller to also consider the rule pair (h,g).
  Certain restrictions also apply to the rule pairs that will be considered.

  For example, if `rule` is:

  NT -> <foo bar, bar foo>

  Then the return set will include the following rule pair:

  NT -> <NT_0 bar, bar NT_0>
  NT -> <foo, foo>

  Args:
    rule: A QcfgRule.
    allow_repeated_target_nts: Whether to allow repeated substrings to be
      replaced with multiple non-terminals sharing the same index in target
      sequences.

  Returns:
    List of rule pairs.
  """
  candidate_pairs = []

  # Consider all pairs of subspans in source and target to replace with
  # a new non-terminal symbol.
  for source_nt_start in range(len(rule.source)):
    for source_nt_end in range(source_nt_start + 1, len(rule.source) + 1):

      source_h = rule.source[source_nt_start:source_nt_end]

      # Don't allow source_h to occur multiple times in rule.source.
      # Otherwise this leads to an ambiguous selection between the occurences,
      # so take the more conservative approach and disallow this.
      if rule_utils.rhs_count(rule.source, source_h) > 1:
        continue

      # Don't allow source_h to only contain a single non-terminal.
      if source_h == tuple([qcfg_rule.NT_1]) or source_h == tuple(
          [qcfg_rule.NT_2]):
        continue

      source_g = (
          rule.source[:source_nt_start] + tuple([NT_TMP]) +
          rule.source[source_nt_end:])

      # Don't allow source_g to only contain a single non-terminal.
      if source_g == tuple([NT_TMP]):
        continue

      # Don't allow source_g to contain >2 non-terminals.
      if qcfg_rule.NT_1 in source_g and qcfg_rule.NT_2 in source_g:
        continue

      for target_nt_start in range(len(rule.target)):
        for target_nt_end in range(target_nt_start + 1, len(rule.target) + 1):

          target_h = rule.target[target_nt_start:target_nt_end]

          # Optionally allow target_h to occur multiple times in rule.target.
          if rule_utils.rhs_count(rule.target, target_h) > 1:
            if allow_repeated_target_nts:
              target_g = rule_utils.rhs_replace(rule.target, target_h, NT_TMP)
            else:
              continue
          else:
            target_g = (
                rule.target[:target_nt_start] + tuple([NT_TMP]) +
                rule.target[target_nt_end:])

          # Don't allow target_g to contain >2 non-terminals.
          if qcfg_rule.NT_1 in target_g and qcfg_rule.NT_2 in target_g:
            continue

          candidate_pair = _maybe_get_candidate_pair(source_g, source_h,
                                                     target_g, target_h)
          if candidate_pair:
            candidate_pairs.append(candidate_pair)
  return candidate_pairs


def find_possible_splits(rule, derivable_rules, allow_repeated_target_nts=True):
  """Implements `NEW` procedure described in paper appendix."""
  candidates = _get_split_candidates(rule, allow_repeated_target_nts)

  # Set of QCFGRules.
  rule_candidates = set()
  for rule_b, rule_c in candidates:
    if rule_b in derivable_rules and rule_c not in derivable_rules:
      # <NT, NT> =>_a == <NT, NT> =>_b =>_c where b is in derivable_rules.
      rule_candidates.add(rule_c)
    elif rule_c in derivable_rules and rule_b not in derivable_rules:
      # <NT, NT> =>_a == <NT, NT> =>_b =>_c where c is in derivable_rules.
      rule_candidates.add(rule_b)

  return rule_candidates

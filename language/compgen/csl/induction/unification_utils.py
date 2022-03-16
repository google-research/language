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
"""Utilities for higher-order unification.

TODO(petershaw): Clean up the implementation to be more consistent between
`_get_candidates_b_c` and `_get_candidates_c_b` and enable more generic handling
of nonterminal re-indexing.
"""

import collections
import itertools

from language.compgen.csl.induction import rule_utils
from language.compgen.csl.qcfg import qcfg_rule


def _is_single_nt(tokens):
  return len(tokens) == 1 and len(qcfg_rule.get_nts(tokens)) == 1


def _get_non_terminals(rhs):
  """Return set of non-terminal symbols in `rhs`."""
  non_terminals = set()
  for symbol in rhs:
    if qcfg_rule.is_nt(symbol):
      non_terminals.add(symbol)
  return non_terminals


def _get_free_nt_replacement(nts):
  """Get the next free NT index."""
  # For example
  # nts = {NT_1, NT_3} will return NT_2.
  # nts = {NT_1, NT_2} will return NT_3.
  nt_indices = set(qcfg_rule.get_nt_index(nt) for nt in nts)
  # The new NT indices are {NT_1, ..., NT_{n+1}}
  missing_idx = set(range(1, len(nts) + 2)) - nt_indices
  return str(qcfg_rule.NT(missing_idx.pop()))


def _get_free_nt(tokens):
  """Get the free NT index."""
  nts = _get_non_terminals(tokens)
  return _get_free_nt_replacement(nts)


def _rhs_count(rhs_a, rhs_b):
  """Returns count of occurances of rhs_b in rhs_a."""
  if len(rhs_b) > len(rhs_a):
    return 0
  count = 0
  for idx in range(len(rhs_a) - len(rhs_b) + 1):
    if rhs_a[idx:idx + len(rhs_b)] == rhs_b:
      count += 1
  return count


def _make_rule(nts, source, target):
  """Canoncalize NT indexes and return QCFGRule."""
  arity = len(nts)
  source, target = rule_utils.canonicalize_nts(source, target, arity)
  return qcfg_rule.QCFGRule(tuple(source), tuple(target), arity)


def _maybe_make_rule(source, target):
  """Canoncalize NT indexes and return QCFGRule."""
  source_nts = _get_non_terminals(source)
  target_nts = _get_non_terminals(target)
  if source_nts != target_nts:
    return None
  arity = len(source_nts)
  source, target = rule_utils.canonicalize_nts(source, target, arity)
  return qcfg_rule.QCFGRule(tuple(source), tuple(target), arity)


def _maybe_get_candidate_pair(source_c, source_b, target_c, target_b):
  """Returns candidate rule pair if proposed sources and targets are valid."""
  # Check that proposed sources and targets contain same non-terminal indexes.
  nts_c = _get_non_terminals(source_c)
  if nts_c != _get_non_terminals(target_c):
    return None

  nts_b = _get_non_terminals(source_b)
  if nts_b != _get_non_terminals(target_b):
    return None

  # Canonicalize non-terminal index ordering and return candidate pair.
  rule_c = _make_rule(nts_c, source_c, target_c)
  rule_b = _make_rule(nts_b, source_b, target_b)
  return (rule_c, rule_b)


def _equal_modulo_nt_idx(tuple_a, tuple_b):
  for token_a, token_b in zip(tuple_a, tuple_b):
    if not (token_a == token_b or
            _is_nonterminal(token_a) and _is_nonterminal(token_b)):
      return False
  return True


def _get_candidates_b_c(rule_a,
                        goal_rule_b,
                        config):
  """Return candidates where `b` can be applied to `c` to form `a`."""

  # Consider subspans in source to replace with a new non-terminal symbol.
  candidate_sources = set()
  for source_nt_start in range(len(rule_a.source)):
    source_nt_end = source_nt_start + len(goal_rule_b.source)

    source_b = rule_a.source[source_nt_start:source_nt_end]

    if not _equal_modulo_nt_idx(source_b, goal_rule_b.source):
      continue

    # Don't allow source_b to occur multiple times in rule_a.source.
    # Otherwise this leads to an ambiguous selection between the occurences,
    # so take the more conservative approach and disallow this.
    if _rhs_count(rule_a.source, source_b) > 1:
      continue

    free_nt = _get_free_nt(rule_a.source[:source_nt_start] +
                           rule_a.source[source_nt_end:])
    source_c = (
        rule_a.source[:source_nt_start] + tuple([free_nt]) +
        rule_a.source[source_nt_end:])

    # Don't allow source_c to only contain a single non-terminal.
    if _is_single_nt(source_c):
      continue

    # Don't allow source_c to contain >k non-terminals.
    if qcfg_rule.get_num_nts(source_c) > config["max_num_nts"]:
      continue

    candidate_sources.add((source_b, source_c))

  # Consider subspans in target to replace with a new non-terminal symbol.
  candidate_targets = set()
  for target_nt_start in range(len(rule_a.target)):
    target_nt_end = target_nt_start + len(goal_rule_b.target)

    target_b = rule_a.target[target_nt_start:target_nt_end]

    if not _equal_modulo_nt_idx(target_b, goal_rule_b.target):
      continue

    # Optionally don't allow target_b to only contain a single non-terminal.
    if not config["allow_single_nt_target"] and _is_single_nt(target_b):
      continue

    # Optionally allow target_b to occur multiple times in rule.target.
    target_b_count = _rhs_count(rule_a.target, target_b)
    if target_b_count > 1:
      if config["allow_repeated_target_nts"]:
        target_a_remove = rule_utils.rhs_remove(rule_a.target, target_b)
        free_nt = _get_free_nt(target_a_remove)
        target_c = rule_utils.rhs_replace(rule_a.target, target_b, free_nt)
        replaced_count = target_c.count(free_nt)
        if replaced_count != target_b_count:
          # This can happen when a subspan is repeated but overlapping.
          continue
      else:
        continue
    else:
      free_nt = _get_free_nt(rule_a.target[:target_nt_start] +
                             rule_a.target[target_nt_end:])
      target_c = (
          rule_a.target[:target_nt_start] + tuple([free_nt]) +
          rule_a.target[target_nt_end:])

    # Optionally don't allow target_c to only contain a single non-terminal.
    if not config["allow_single_nt_target"] and _is_single_nt(target_c):
      continue

    # Don't allow target_c to contain >k non-terminals.
    if qcfg_rule.get_num_nts(target_c) > config["max_num_nts"]:
      continue

    candidate_targets.add((target_b, target_c))

  candidates = set()
  for ((source_b, source_c),
       (target_b, target_c)) in itertools.product(candidate_sources,
                                                  candidate_targets):
    candidate_pair = _maybe_get_candidate_pair(source_c, source_b, target_c,
                                               target_b)

    if candidate_pair:
      rule_c, rule_b = candidate_pair
      if rule_b == goal_rule_b:
        candidates.add(rule_c)
  return candidates


# TODO(petershaw): Consider making this a mutable object to avoid some copying.
SearchState = collections.namedtuple(
    "SearchState",
    [
        "tuple_a_idx",
        "tuple_b_idx",
        "sub_complete",
        "sub_lhs",
        "sub_rhs",
        "nt_subs",  # Map of nt to nt.
    ])


def _is_nonterminal(token):
  return qcfg_rule.is_nt(token)


def _symbols_eq(state, token_a, token_b):
  # First, handle special cases.
  if _is_nonterminal(token_b):
    if token_b == state.sub_lhs:
      return False
    if token_b in state.nt_subs:
      return False
  return token_a == token_b


def _find_nt_sub(state, token_a, token_b):
  if (_is_nonterminal(token_a) and _is_nonterminal(token_b) and
      token_a != token_b):
    nt_sub = (token_b, token_a)
    if token_b == state.sub_lhs:
      return None
    if token_b in state.nt_subs:
      return None
    return nt_sub
  return None


def _check_symbols_eq_and_maybe_increment(token_a, token_b, state, search_stack,
                                          sub_complete):
  """Returns new states if token_a and token_b are equal."""

  # First, check if symbols are exactly equal.
  if _symbols_eq(state, token_a, token_b):
    new_state = SearchState(
        tuple_a_idx=state.tuple_a_idx + 1,
        tuple_b_idx=state.tuple_b_idx + 1,
        sub_complete=sub_complete,
        sub_lhs=state.sub_lhs,
        sub_rhs=state.sub_rhs,
        nt_subs=state.nt_subs)
    search_stack.append(new_state)

  # Second, check if symbols vary only in non-terminal index.
  nt_sub = _find_nt_sub(state, token_a, token_b)
  if nt_sub:
    new_nt_subs = state.nt_subs.copy()
    new_nt_subs[nt_sub[0]] = nt_sub[1]
    new_state = SearchState(
        tuple_a_idx=state.tuple_a_idx + 1,
        tuple_b_idx=state.tuple_b_idx + 1,
        sub_complete=sub_complete,
        sub_lhs=state.sub_lhs,
        sub_rhs=state.sub_rhs,
        nt_subs=new_nt_subs)
    search_stack.append(new_state)


def _replace_nts(symbols, nt_subs, sub_lhs, sub_rhs):
  """Replace NTs in `symbols`."""
  new_symbols = []
  for symbol in symbols:
    if symbol in nt_subs:
      new_symbols.append(nt_subs[symbol])
    elif symbol == sub_lhs:
      new_symbols.extend(sub_rhs)
    else:
      new_symbols.append(symbol)
  return tuple(new_symbols)


def find_substitutions_for_b(tuple_a, tuple_b, allow_repeated_nts):
  """Return substituions for `tuple_b` to match `tuple_a`."""
  # For example:
  # `a` could be `foo bar`
  # `b` could be `foo NT_1`
  # `c` could be `bar`
  # Or:
  # `a` could be `foo NT_1 and NT_2`
  # `b` could be `foo NT_1`
  # `c` could be `NT_1 and NT_2`.
  # Or:
  # `a` could be `foo NT_1 bar NT_2`
  # `b` could be `foo NT_1 NT_2`
  # `c` could be `NT_1 bar` or `bar NT_2`.
  # Keep a stack of SearchState.
  completed_states = []
  search_stack = [
      SearchState(
          tuple_a_idx=0,
          tuple_b_idx=0,
          sub_complete=False,
          sub_lhs=None,
          sub_rhs=None,
          nt_subs={})
  ]
  while search_stack:
    state = search_stack.pop()

    if (state.sub_lhs and state.tuple_a_idx == len(tuple_a) and
        state.tuple_b_idx == len(tuple_b)):
      completed_states.append(state)
      continue

    if state.tuple_a_idx >= len(tuple_a):
      continue
    token_a = tuple_a[state.tuple_a_idx]

    if state.tuple_b_idx >= len(tuple_b):
      token_b = None
    else:
      token_b = tuple_b[state.tuple_b_idx]

    # First, check for the special case of a repeated non-terminal index.
    if (allow_repeated_nts and state.sub_lhs and token_b == state.sub_lhs and
        len(tuple_a) >= state.tuple_a_idx + len(state.sub_rhs) and
        tuple_a[state.tuple_a_idx:state.tuple_a_idx + len(state.sub_rhs)]
        == tuple(state.sub_rhs)):
      new_state = SearchState(
          tuple_a_idx=state.tuple_a_idx + len(state.sub_rhs),
          tuple_b_idx=state.tuple_b_idx + 1,
          sub_complete=True,
          sub_lhs=state.sub_lhs,
          sub_rhs=state.sub_rhs,
          nt_subs=state.nt_subs)
      search_stack.append(new_state)

    # If we have already determined a substituion, then need to continue
    # matching terminals or non-terminals.
    if state.sub_complete:
      _check_symbols_eq_and_maybe_increment(token_a, token_b, state,
                                            search_stack, True)

    # Handle case where state is currently completing the RHS for a
    # substitution.
    elif not state.sub_complete and state.sub_lhs:
      # Check if substitution can be completed.
      if state.sub_rhs:
        _check_symbols_eq_and_maybe_increment(token_a, token_b, state,
                                              search_stack, True)

      # Continue increasing RHS substituion and increment tuple_a_idx only.
      new_state = SearchState(
          tuple_a_idx=state.tuple_a_idx + 1,
          tuple_b_idx=state.tuple_b_idx,
          sub_complete=False,
          sub_lhs=state.sub_lhs,
          sub_rhs=state.sub_rhs + [token_a],
          nt_subs=state.nt_subs)
      search_stack.append(new_state)

    # Handle case where no substitution has been started yet.
    elif not state.sub_lhs:
      # Consider continuing to match symbols.
      _check_symbols_eq_and_maybe_increment(token_a, token_b, state,
                                            search_stack, False)

      # Consider starting a substituion.
      if _is_nonterminal(token_b) and token_b not in state.nt_subs:
        new_state = SearchState(
            tuple_a_idx=state.tuple_a_idx + 1,
            tuple_b_idx=state.tuple_b_idx + 1,
            sub_complete=False,
            sub_lhs=token_b,
            sub_rhs=[token_a],
            nt_subs=state.nt_subs)
        search_stack.append(new_state)

  # Perform final check that tuple_b with proposed replacements matches tuple_a.
  # Note this can fail for certain cases related to repeated nonterminals.
  # TODO(petershaw): Potentially fix these cases upstream before here.
  ret_values = []
  for state in completed_states:
    tuple_b_replaced = _replace_nts(tuple_b, state.nt_subs, state.sub_lhs,
                                    state.sub_rhs)
    # Note that there is case where `state.sub_rhs` includes nontemrinals that
    # also occur elsewhere in `tuple_b`, which could be problematic, but will
    # be caught by `_maybe_make_rule` below.
    if tuple_a != tuple_b_replaced:
      continue
    ret_values.append((state.sub_lhs, state.sub_rhs, state.nt_subs))
  return ret_values


def _get_candidates_c_b(rule_a, rule_b, config):
  """Return candidates where `c` can be applied to `b` to form `a`."""
  candidates = set()
  if rule_b.arity > 0:
    source_subs = find_substitutions_for_b(rule_a.source, rule_b.source, False)
    target_subs = find_substitutions_for_b(rule_a.target, rule_b.target,
                                           config["allow_repeated_target_nts"])
    for ((source_lhs, source_rhs, source_nt_subs),
         (target_lhs, target_rhs,
          target_nt_subs)) in itertools.product(source_subs, target_subs):
      if source_lhs != target_lhs or source_nt_subs != target_nt_subs:
        continue
      # Don't allow source to only contain a single non-terminal.
      if _is_single_nt(source_rhs):
        continue
      # Optionally don't allow target to only contain a single non-terminal.
      if not config["allow_single_nt_target"] and _is_single_nt(target_rhs):
        continue
      new_rule = _maybe_make_rule(tuple(source_rhs), tuple(target_rhs))
      if new_rule:
        candidates.add(new_rule)
  return candidates


def get_rule_unifiers(rule_a, rule_b, config):
  """Returns `rule_c` that can be combined with `rule_b` to equal `rule_a`.

  Here we interpret rules as functions over strings, with arity equal to the
  number of uniquely indexed non-terminals.
  Our task can be viewed as a restricted form of higher-order unification.
  We are looking for a function `c` that can be applied to an argument of `b`
  that `unifies` `b` and `a`, or vice versa.

  For example, if `a` has arity 2 and `b` has arity 2, we are looking for `c`
  such that one of the following holds:
    - a(x, y) = b(c(x), y)
    - a(x, y) = b(x, c(y))
    - a(x, y) = c(b(x, y))

  More concretely, consider the following rules:
    - `rule_a` is `foo bar ### FOO BAR`
    - `rule_b` is `foo NT_1 ### FOO NT_1`
    - The function should return {`bar ### BAR`}.

  Args:
    rule_a: QCFGRule.
    rule_b: QCFGRule.
    config: Config dict.

  Returns:
    Set of QCFGRule candidates for `rule_c`.
  """
  candidates = set()

  # First, consider applying `c` to an argument of `b`.
  # For example:
  # `a` could be `foo bar ### FOO BAR`
  # `b` could be `foo NT_1 ### FOO NT_1`
  # `c` could be `bar ### BAR`
  # Therefore, our strategy is to find some substitution for the non-terminals
  # in `b` that matches `a`.
  candidates_c_b = _get_candidates_c_b(rule_a, rule_b, config)
  candidates |= candidates_c_b

  # Second, consider applying `b` to an argument of `c`.
  # For example:
  # `a` could be `foo bar ### FOO BAR`
  # `b` could be `bar ### BAR`
  # `c` could be `foo NT_1 ### FOO NT_1`
  candidates_b_c = _get_candidates_b_c(rule_a, rule_b, config)
  candidates |= candidates_b_c

  return candidates

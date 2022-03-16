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
"""Various common functions related to QCFGRules."""

from language.compgen.csl.qcfg import qcfg_rule


def _swap_nt_order(rhs, old_to_new):
  new_rhs = []
  for symbol in rhs:
    if qcfg_rule.is_nt_fast(symbol):
      old_idx = qcfg_rule.get_nt_index(symbol)
      new_rhs.append(str(qcfg_rule.NT(old_to_new[old_idx])))
    else:
      new_rhs.append(symbol)
  return tuple(new_rhs)


def canonicalize_nts(source, target, arity):
  """Follows convention of source indexes being in order."""
  source_nts = []
  for token in source:
    if qcfg_rule.is_nt(token) and token not in source_nts:
      source_nts.append(token)
  if len(set(source_nts)) != arity:
    raise ValueError("Bad arity 2 source: %s" % (source,))
  old_to_new = {
      qcfg_rule.get_nt_index(nt): idx + 1 for idx, nt in enumerate(source_nts)
  }
  source = _swap_nt_order(source, old_to_new)
  target = _swap_nt_order(target, old_to_new)
  return source, target


def rhs_can_maybe_derive(rhs_a, rhs_b):
  """Return False if rhs_a cannot be used in a derivation of rhs_b.

  This function uses a fast approximation based on terminal sequence overlap
  to identify cases where `rhs_a` could never be used in a derivation of
  `rhs_b`.

  For example, given `rhs_a`:

  "foo NT foo NT"

  There is a derivation that includes `rhs_a` that derives:

  "foo bar bar foo NT"

  But there is no derivation that includes `rhs_a` and derives:

  "foo NT dax NT"

  Args:
    rhs_a: Tuple of strings for source or target of QCFGRule.
    rhs_b: Same type as rhs_a.

  Returns:
    False if rhs_a cannot be used in a derivation of rhs_b.
  """
  len_rhs_a = len(rhs_a)
  len_rhs_b = len(rhs_b)
  if len_rhs_a > len_rhs_b:
    return False
  if not rhs_a or not rhs_b:
    return False

  # Cache for each symbol in `rhs_a` whether it is nonterminal.
  is_nt = [qcfg_rule.is_nt_fast(symbol) for symbol in rhs_a]

  # Quick check that all terminals in `rhs_a` are in `rhs_b`.
  rhs_b_symbols = set(rhs_b)
  for symbol_is_nt, symbol in zip(is_nt, rhs_a):
    if not symbol_is_nt:
      if symbol not in rhs_b_symbols:
        return False

  # Track the positions in `rhs_a` and `rhs_b` of the search process.
  rhs_a_idx = 0
  rhs_b_idx = 0
  # Lookahead to check whether the next sequence of terminals in `rhs_a`
  # starting at `a_idx` matches the sequence of terminals in `rhs_b` starting
  # at `b_idx`.
  rhs_a_idx_lookahead = 0
  rhs_b_idx_lookahead = 0

  while True:
    if rhs_a_idx_lookahead >= len_rhs_a:
      # Completed matching all terminals.
      return True

    if rhs_b_idx_lookahead >= len_rhs_b:
      # Failed to match all terminal sequences.
      return False

    # Fail early if match cannot be made based on remaining length.
    if (len_rhs_a - rhs_a_idx_lookahead) > (len_rhs_b - rhs_b_idx_lookahead):
      return False

    a_symbol = rhs_a[rhs_a_idx_lookahead]
    b_symbol = rhs_b[rhs_b_idx_lookahead]

    if is_nt[rhs_a_idx_lookahead]:
      # Completed matching terminal sequence.
      # Increment lookahead indexes past this sequence.
      rhs_a_idx_lookahead += 1
      rhs_b_idx_lookahead += 1
      rhs_a_idx = rhs_a_idx_lookahead
      rhs_b_idx = rhs_b_idx_lookahead
    elif a_symbol == b_symbol:
      # Matched next terminal symbol, increment lookahead indexes.
      rhs_a_idx_lookahead += 1
      rhs_b_idx_lookahead += 1
    else:
      # Terminal symbols do not match, so increment b_idx and try again.
      # We can safely increment b_idx because we either: (1) have not matched
      # any symbols in `rhs_a` yet, or (2) previously matched a nonterminal
      # in `rhs_a`.
      rhs_a_idx_lookahead = rhs_a_idx
      rhs_b_idx += 1
      rhs_b_idx_lookahead = rhs_b_idx


def rhs_replace(rhs, sublist, replacement):
  """Replace occurrences of sublist in rhs with replacement."""
  sublist = tuple(sublist)
  rhs = tuple(rhs)
  if len(sublist) > len(rhs):
    raise ValueError
  if not sublist:
    raise ValueError
  new_list = []
  idx = 0
  while idx < len(rhs):
    if rhs[idx:idx + len(sublist)] == sublist:
      new_list.append(replacement)
      idx += len(sublist)
    else:
      new_list.append(rhs[idx])
      idx += 1
  return tuple(new_list)


def rhs_remove(rhs, sublist):
  """Remove occurrences of sublist in rhs."""
  new_list = []
  idx = 0
  if len(sublist) > len(rhs):
    return new_list
  while idx < len(rhs):
    if rhs[idx:idx + len(sublist)] == sublist:
      idx += len(sublist)
    else:
      new_list.append(rhs[idx])
      idx += 1
  return new_list

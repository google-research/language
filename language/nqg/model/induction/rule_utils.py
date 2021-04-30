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

from language.nqg.model.qcfg import qcfg_rule


def _swap_nt_order(rhs):
  new_rhs = []
  for symbol in rhs:
    if symbol == qcfg_rule.NT_1:
      new_rhs.append(qcfg_rule.NT_2)
    elif symbol == qcfg_rule.NT_2:
      new_rhs.append(qcfg_rule.NT_1)
    else:
      new_rhs.append(symbol)
  return tuple(new_rhs)


def canonicalize_nts(source, target, arity):
  """Follows convention of source indexes being in order."""
  if arity == 1:
    if qcfg_rule.NT_2 in source:
      source = rhs_replace(source, [qcfg_rule.NT_2], qcfg_rule.NT_1)
      target = rhs_replace(target, [qcfg_rule.NT_2], qcfg_rule.NT_1)
  elif arity == 2:
    if qcfg_rule.NT_1 not in source or qcfg_rule.NT_2 not in source:
      raise ValueError("Bad arity 2 source: %s" % (source,))
    if source.index(qcfg_rule.NT_1) > source.index(qcfg_rule.NT_2):
      source = _swap_nt_order(source)
      target = _swap_nt_order(target)
  return source, target


def rhs_count(list_to_search, sublist):
  """Returns count of occurances of sublist in list_to_search."""
  if len(sublist) > len(list_to_search):
    return 0
  count = 0
  for idx in range(len(list_to_search) - len(sublist) + 1):
    if list_to_search[idx:idx + len(sublist)] == sublist:
      count += 1
  return count


def rhs_contains(list_to_search, sublist):
  """Returns True if sublist is contained in list_to_search."""
  if len(sublist) > len(list_to_search):
    return False
  for idx in range(len(list_to_search) - len(sublist) + 1):
    if list_to_search[idx:idx + len(sublist)] == sublist:
      return True
  return False


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

  # Represent search state with backtracking.
  rhs_a_idx_backtrack = 0
  rhs_a_idx = 0
  rhs_b_idx_backtrack = 0
  rhs_b_idx = 0

  while True:
    if rhs_a_idx >= len_rhs_a:
      # Completed matching all terminals.
      return True

    if rhs_b_idx >= len_rhs_b:
      # Failed to match all terminal sequences.
      return False

    # Fail early if match cannot be made based on remaining length.
    if (len_rhs_a - rhs_a_idx) > (len_rhs_b - rhs_b_idx):
      return False

    a_symbol = rhs_a[rhs_a_idx]
    b_symbol = rhs_b[rhs_b_idx]

    if a_symbol == b_symbol:
      # Matched next terminal symbol, increment indexes.
      rhs_a_idx += 1
      rhs_b_idx += 1
    elif a_symbol == qcfg_rule.NT_2 or a_symbol == qcfg_rule.NT_1:
      # Completed matching terminal sequence.
      # Increment backtrack indexes past this sequence.
      rhs_a_idx += 1
      rhs_a_idx_backtrack = rhs_a_idx
      rhs_b_idx_backtrack = rhs_b_idx
    else:
      # Symbols do not match, backtrack.
      rhs_a_idx = rhs_a_idx_backtrack
      rhs_b_idx_backtrack += 1
      rhs_b_idx = rhs_b_idx_backtrack


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

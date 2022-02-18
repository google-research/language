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
"""Utilities for defining atoms and compounds for SQL TMCD splits."""

import collections

from language.compgen.nqg.tasks.spider import sql_parser
from language.compgen.nqg.tasks.spider import sql_tokenizer

# Placeholder symbol for non-terminals in compounds.
_PLACEHOLDER = "___"


def get_atoms(sql_expr):
  """Simply return tokens in the SQL query as atoms."""
  return set(sql_tokenizer.tokenize_sql(sql_expr))


def _get_node_string(node):
  """Return string for RHS of node with non-terminals anonymized."""
  rhs = node.rhs
  tokens = rhs.split()
  new_tokens = []
  # Anonymize non-terminals.
  for token in tokens:
    if token.startswith("##"):
      new_tokens.append(_PLACEHOLDER)
    else:
      new_tokens.append(token)
  return " ".join(new_tokens)


def _replace_nonterminal(node_1_string, node_1_idx, node_2_string):
  """Replace corresponding non-terminal in node_2_string with node_1_string."""
  non_terminals = 0
  new_tokens = []
  for token in node_2_string.split():
    if token == _PLACEHOLDER:
      if non_terminals == node_1_idx:
        new_tokens.append(node_1_string)
      else:
        new_tokens.append(token)
      non_terminals += 1
    else:
      new_tokens.append(token)
  return " ".join(new_tokens)


def _get_2_compound_string(node_1, node_1_idx, node_2):
  node_1_string = _get_node_string(node_1)
  node_2_string = _get_node_string(node_2)
  return _replace_nonterminal(node_1_string, node_1_idx, node_2_string)


def _get_3_compound_string(node_1, node_1_idx, node_2, node_2_idx, node_3):
  node_1_string = _get_node_string(node_1)
  node_2_string = _get_node_string(node_2)
  node_3_string = _get_node_string(node_3)
  return _replace_nonterminal(
      _replace_nonterminal(node_1_string, node_1_idx, node_2_string),
      node_2_idx, node_3_string)


def _get_compounds_inner(node_1, node_1_idx, node_2, node_2_idx, node_3):
  """Recursively generates compounds.

  Each iteration adds the compounds for these paths if inputs are not None:
  - node_2 -> node_1
  - node_3 -> node_2 -> node_1

  Args:
    node_1: Node.
    node_1_idx: Index of node_1 in node_2's children list.
    node_2: Node.
    node_2_idx: Index of node_2 in node_3's children list.
    node_3: Node

  Returns:
    Counter of compounds.
  """
  compounds = collections.Counter()

  if node_2:
    compound_2 = _get_2_compound_string(node_1, node_1_idx, node_2)
    compounds[compound_2] += 1

  if node_3:
    compound_3 = _get_3_compound_string(node_1, node_1_idx, node_2, node_2_idx,
                                        node_3)
    compounds[compound_3] += 1

  for child_idx, child in enumerate(node_1.children):
    compounds.update(
        _get_compounds_inner(child, child_idx, node_1, node_1_idx, node_2))
  return compounds


def get_compounds(sql_expr):
  root_node = sql_parser.parse_sql(sql_expr)
  compounds = _get_compounds_inner(root_node, None, None, None, None)
  return compounds


def get_example_compounds(example):
  return get_compounds(example[1])


def get_example_atoms(example):
  return get_atoms(example[1])

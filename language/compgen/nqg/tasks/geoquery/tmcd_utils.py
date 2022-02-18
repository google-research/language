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
"""Utilities for defining atoms and compounds for FunQL."""

import collections

# Placeholder symbol for compounds.
_PLACEHOLDER = "__"


def _split_arguments(args_string):
  """Splits comma-joined argument list.

  For example, an input of "foo, bar(xyz, abc), bar" will be split
  into: ["foo", "bar(xyz, abc)", "bar"].

  Args:
    args_string: String input for comma-separated argument list.

  Returns:
    List of Strings for each argument.
  """
  argument_buffer = []
  arguments = []
  open_parens = 0
  for char in args_string:
    if char == "," and open_parens == 0:
      arguments.append("".join(argument_buffer))
      argument_buffer = []
    elif char == " " and not argument_buffer:
      continue
    else:
      if char == "(":
        open_parens += 1
      elif char == ")":
        open_parens -= 1
      argument_buffer.append(char)
  arguments.append("".join(argument_buffer))
  return arguments


def _get_name_and_arguments(funql):
  """Returns function name and argument sub-expressions."""
  funql = funql.strip()
  paren_index = funql.find("(")
  if paren_index == -1:
    return funql, None
  name = funql[:paren_index].strip()
  arguments = funql[paren_index + 1:].strip()
  if arguments[-1] != ")":
    raise ValueError("Invalid arguments string ends with %s: %s" %
                     (arguments[-1], arguments))
  arguments = _split_arguments(arguments[:-1])
  return name, arguments


def _get_compound_string(outer, outer_arity, inner, inner_idx):
  arguments = [_PLACEHOLDER] * outer_arity
  arguments[inner_idx] = inner
  return "%s( %s )" % (outer, " , ".join(arguments))


def _get_compounds_inner(funql, compounds_to_counts):
  """Recursively add compound counts to compounds_to_counts."""
  name, arguments = _get_name_and_arguments(funql)
  if not arguments:
    return

  for argument_idx, argument in enumerate(arguments):
    argument_name, _ = _get_name_and_arguments(argument)
    compound = _get_compound_string(name, len(arguments), argument_name,
                                    argument_idx)
    compounds_to_counts[compound] += 1
    _get_compounds_inner(argument, compounds_to_counts)


def get_compounds(target):
  """Use combinations of 2 atoms as compounds."""
  compounds_to_count = collections.Counter()
  _get_compounds_inner(target, compounds_to_count)
  return compounds_to_count


def get_atoms(target):
  """Use individual tokens as atoms."""
  atoms = set()
  for token in target.split():
    if token not in ("(", ")", ","):
      atoms.add(token)
  return atoms


def get_atoms_with_num_arguments(target):
  """Consider symbols and their number of arguments."""
  name, arguments = _get_name_and_arguments(target)
  if arguments:
    atoms = set()
    atoms.add("%s_(%s)" % (name, len(arguments)))
    for argument in arguments:
      atoms |= get_atoms_with_num_arguments(argument)
    return atoms
  else:
    return {name}


def get_example_compounds(example):
  return get_compounds(example[1])


def get_example_atoms(example):
  return get_atoms(example[1])


def get_example_atoms_with_num_arguments(example):
  return get_atoms_with_num_arguments(example[1])

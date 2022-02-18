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
# Lint as: python3
"""Utilities for (reversible) normalization of FunQL.

FunQL is defined here:
https://www.cs.utexas.edu/~ml/wasp/geo-funql.html

We use the corresponding lambda term definitions to expand various functions
to a more intuitive form that better reflects the arity of the underlying
operations.
"""

RELATION_CONSTANTS = [
    "area_1", "capital_1", "capital_2", "density_1", "elevation_1",
    "elevation_2", "high_point_1", "high_point_2", "higher_2", "loc_1", "loc_2",
    "low_point_1", "low_point_2", "lower_2", "next_to_1", "next_to_2",
    "population_1", "traverse_1", "traverse_2", "longer", "len", "size"
]

# Can occur with `all` as argumnent.
UNARY_CONSTANTS = [
    "capital", "city", "lake", "major", "mountain", "place", "river", "state"
]

ENTITY_FUNCTIONS = ["cityid", "stateid", "riverid", "placeid", "countryid"]

ARITY_1 = [
    "largest", "smallest", "highest", "lowest", "longest", "shortest", "count",
    "sum"
]

ARITY_2 = ["largest_one", "smallest_one"]

ARITY_3 = ["most", "fewest"]


def _split_arguments(span):
  """Splits span into list of spans based on commas."""
  argument_buffer = []
  arguments = []
  open_parens = 0
  for char in span:
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


def _get_name_and_arguments(span):
  """Returns function name and argument sub-expressions."""
  span = span.strip()
  paren_index = span.find("(")
  if paren_index == -1:
    raise ValueError("Funql contains no `(`: %s" % span)
  name = span[:paren_index]
  arguments = span[paren_index + 1:]
  if arguments[-1] != ")":
    raise ValueError("Invalid arguments string ends with %s: %s" %
                     (arguments[-1], arguments))
  arguments = _split_arguments(arguments[:-1])
  return name, arguments


def _convert_function(name, argument_0, arity):
  """Converts a function that contains nested arguments."""
  output_arguments = []
  inner_funql = argument_0
  for _ in range(arity - 1):
    nested_argument, arguments = _get_name_and_arguments(inner_funql)
    if len(arguments) > 1:
      raise ValueError
    inner_funql = arguments[0]
    output_arguments.append(nested_argument)
  output_arguments.append(normalize_funql(inner_funql))
  output = "%s(%s)" % (name, ",".join(output_arguments))
  return output


def normalize_funql(funql):
  """Recursively parse FunQL string to re-formatted string."""
  # Special constant used for "sea level".
  if funql == "0":
    return "0"

  name, arguments = _get_name_and_arguments(funql)
  if name == "answer":
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    return "%s(%s)" % (name, normalize_funql(argument_0))
  elif name in ENTITY_FUNCTIONS:
    return funql
  elif name in RELATION_CONSTANTS:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    reformatted_argument_0 = normalize_funql(argument_0)
    if not reformatted_argument_0:
      raise ValueError("Failed to reformat: %s" % argument_0)
    return "%s(%s)" % (name, reformatted_argument_0)
  elif name in UNARY_CONSTANTS:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    if argument_0 == "all":
      return name
    else:
      recursive_term = normalize_funql(argument_0)
      return "intersection(%s,%s)" % (name, recursive_term)
  elif name == "intersection" or name == "exclude":
    if len(arguments) != 2:
      raise ValueError
    argument_0 = arguments[0]
    argument_1 = arguments[1]
    term_a = normalize_funql(argument_0)
    term_b = normalize_funql(argument_1)
    return "%s(%s,%s)" % (name, term_a, term_b)
  elif name in ARITY_1:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    return _convert_function(name, argument_0, 1)
  elif name in ARITY_2:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    return _convert_function(name, argument_0, 2)
  elif name in ARITY_3:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    return _convert_function(name, argument_0, 3)
  else:
    raise ValueError("No match for name: %s" % name)


def restore_funql(funql):
  """Recursively parse FunQL string back to original string."""
  # Special constant used for "sea level".
  if funql == "0":
    return "0"

  if funql in UNARY_CONSTANTS:
    return "%s(all)" % funql

  name, arguments = _get_name_and_arguments(funql)
  if name == "answer" or name in RELATION_CONSTANTS or name in ARITY_1:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    return "%s(%s)" % (name, restore_funql(argument_0))
  elif name in RELATION_CONSTANTS:
    if len(arguments) != 1:
      raise ValueError
    argument_0 = arguments[0]
    restored_argument_0 = restore_funql(argument_0)
    if not restored_argument_0:
      raise ValueError("Failed to restore: %s" % argument_0)
    return "%s(%s)" % (name, restored_argument_0)
  elif name in ENTITY_FUNCTIONS:
    return funql
  elif name == "intersection":
    if len(arguments) != 2:
      raise ValueError
    argument_0 = arguments[0]
    argument_1 = arguments[1]
    term_a = restore_funql(argument_0)
    term_b = restore_funql(argument_1)
    if argument_0 in UNARY_CONSTANTS:
      return "%s(%s)" % (argument_0, restore_funql(argument_1))
    if argument_1 in UNARY_CONSTANTS:
      raise ValueError
    return "%s(%s,%s)" % (name, term_a, term_b)
  elif name == "exclude":
    if len(arguments) != 2:
      raise ValueError
    argument_0 = arguments[0]
    argument_1 = arguments[1]
    term_a = restore_funql(argument_0)
    term_b = restore_funql(argument_1)
    return "%s(%s,%s)" % (name, term_a, term_b)
  elif name in ARITY_2:
    if len(arguments) != 2:
      raise ValueError("Unexpected number of arguments `%s` for `%s`" %
                       (arguments, name))
    argument_0 = arguments[0]
    argument_1 = arguments[1]
    return "%s(%s(%s))" % (name, argument_0, restore_funql(argument_1))
  elif name in ARITY_3:
    if len(arguments) != 3:
      raise ValueError
    argument_0 = arguments[0]
    argument_1 = arguments[1]
    argument_2 = arguments[2]
    return "%s(%s(%s(%s)))" % (name, argument_0, argument_1,
                               restore_funql(argument_2))
  else:
    raise ValueError("No match for name: %s" % name)


def add_space_separation(funql):
  """Split funql and join with space separator."""
  separators = "(),"
  buffer = ""
  symbols = []
  for char in funql:
    if char in separators:
      if buffer:
        symbols.append(buffer)
        buffer = ""
      symbols.append(char)
    else:
      buffer += char
  if buffer:
    symbols.append(buffer)
  return " ".join(symbols)

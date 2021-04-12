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
"""A parser that applies all transformations for text-to-SQL datasets."""

import re

from language.compir.dataset_parsers import dataset_parser


class SqlParser(dataset_parser.DatasetParserInterface):
  """A parser that applies all transformations for text-to-SQL datasets."""

  def preprocess_program(self, program):
    """Switches OOV T5 tokens and performs lowercasing."""
    program_processed = str(program)
    program_processed = program_processed.lower()
    program_processed = program_processed.replace("<", "lt")
    return program_processed

  def f_reversible(self, example):
    """Transforms a single program to its reversible IR."""
    return example.program.replace("alias", "")

  def _remove_table_joins(self, lir_partial, regex):
    table_join_conditions = re.findall(regex, lir_partial)
    for condition in table_join_conditions:
      condition_parts = condition.split(" = ")
      # Remove if it is indeed a table join condition.
      if "." in condition_parts[0] and "." in condition_parts[1]:
        lir_partial = lir_partial.replace(condition, "")
    return lir_partial

  def f_lossy(self, program, is_rir):
    """Transforms a single program to its lossy IR."""

    lir = str(program)

    # replace FROM clause with an "alias" token
    as_command_regex = r" ([^ ]+? as [^ ]+?) "
    as_commands = re.findall(as_command_regex, program)
    for as_command in as_commands:
      lir = lir.replace(as_command, "alias")
    lir = lir.replace(" alias ,", "")

    # anonymize table names
    table_name_regex = r" ([^ ]+?)\.[^ ]+? "
    table_names = re.findall(table_name_regex, lir)
    for table in table_names:
      lir = lir.replace(table, "table")

    # remove table join conditions
    regex_1 = r"and [^ ]+? = [^ ]+? "
    lir = self._remove_table_joins(lir, regex_1)
    regex_2 = r"[^ ]+? = [^ ]+? and "
    lir = self._remove_table_joins(lir, regex_2)
    return lir

  def postprocess_program(self, program):
    """Replaces back T5 OOV tokens."""
    program_processed = str(program)
    program_processed = program_processed.replace("lt", "<")
    return program_processed

  def f_reversible_inverse(self, program):
    """Adds 'alias' back to the names of tables."""
    table_name_regex = r" [^ ]+? as ([^ ]+?) "
    table_names = re.findall(table_name_regex, program)
    program_processed = str(program)
    for table_name in table_names:
      # Insert 'alias' before the numerical suffix.
      table_name_original = re.sub(r"(.+)(\d+)", r"\1alias\2", table_name)
      program_processed = program_processed.replace(table_name,
                                                    table_name_original)
    return program_processed

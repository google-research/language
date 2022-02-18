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
"""Define structures to represent CFG symbols and rules.

For efficiency, all symbols are referenced by integers rather than strings.
This typically requires some pre-processing to define terminal
and non-terminal vocabularies and map symbols to corresponding integers.
"""

import collections

# CFGSymbol type constants.
TERMINAL = 0
NON_TERMINAL = 1

# Represents a TERMINAL or NON_TERMINAL symbol.
CFGSymbol = collections.namedtuple(
    "CFGSymbol",
    [
        "idx",  # Integer (considered as separate id spaces for different type).
        "type",  # Integer (TERMINAL or NON_TERMINAL).
    ])

# Represents a CFG rule.
CFGRule = collections.namedtuple(
    "CFGRule",
    [
        "idx",  # Integer to optionally reference additional rule information.
        "lhs",  # Integer non-terminal index.
        "rhs",  # Tuple of >= 1 CFGSymbols.
    ])

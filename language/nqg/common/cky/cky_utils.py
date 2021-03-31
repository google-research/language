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
"""This module implements a CFG parser based on a variant of the CKY algorithm.

The parser is naively extended to consider non-binarized rules containing up
to 2 RHS non-terminals and any number of terminals. The runtime for this
naive implementation is therefore O(n^6), which can be too slow for longer
inputs.
"""

import collections

from language.nqg.common.cky import cfg_rule


def parse(input_ids,
          rules,
          nonterminals,
          start_idx,
          populate_fn,
          postprocess_fn,
          verbose=False):
  """Run bottom up parser using variant of CKY algorithm."""
  input_len = len(input_ids)
  input_symbols = tuple(
      [cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL) for idx in input_ids])

  # Initialize the empty chart.
  # Keys are a 3-tuple of Integers: (span_begin, span_end, nonterminal_idx)
  # Values are a list of T.
  chart = collections.defaultdict(list)

  # Index rules by RHS.
  rhs_to_rules = collections.defaultdict(list)
  for rule in rules:
    rhs_to_rules[rule.rhs].append(rule)

  # Populate the chart.
  for span_end in range(1, input_len + 1):
    for span_begin in range(span_end - 1, -1, -1):

      # Find matching rules with 0 NTs.
      rhs_key_0_nt = input_symbols[span_begin:span_end]

      if rhs_key_0_nt in rhs_to_rules:
        for rule in rhs_to_rules[rhs_key_0_nt]:
          chart[span_begin, span_end,
                rule.lhs].append(populate_fn(span_begin, span_end, rule, []))

      # Find matching rules with 1 NTs.
      for nt_0_start in range(span_begin, span_end):
        for nt_0_end in range(nt_0_start + 1, span_end + 1):
          for nt_0 in nonterminals:

            rhs_key_1_nt = (
                input_symbols[span_begin:nt_0_start] +
                (cfg_rule.CFGSymbol(nt_0, cfg_rule.NON_TERMINAL),) +
                input_symbols[nt_0_end:span_end])

            if rhs_key_1_nt in rhs_to_rules:
              for node_0 in chart[nt_0_start, nt_0_end, nt_0]:
                for rule in rhs_to_rules[rhs_key_1_nt]:
                  chart[span_begin, span_end, rule.lhs].append(
                      populate_fn(span_begin, span_end, rule, [node_0]))

      # Find matching rules with 2 NTs.
      for nt_0_start in range(span_begin, span_end - 1):
        for nt_0_end in range(nt_0_start + 1, span_end):
          for nt_1_start in range(nt_0_end, span_end):
            for nt_1_end in range(nt_1_start + 1, span_end + 1):
              for nt_0 in nonterminals:
                for nt_1 in nonterminals:

                  rhs_key_2_nt = (
                      input_symbols[span_begin:nt_0_start] +
                      (cfg_rule.CFGSymbol(nt_0, cfg_rule.NON_TERMINAL),) +
                      input_symbols[nt_0_end:nt_1_start] +
                      (cfg_rule.CFGSymbol(nt_1, cfg_rule.NON_TERMINAL),) +
                      input_symbols[nt_1_end:span_end])

                  if rhs_key_2_nt in rhs_to_rules:
                    nt_0_index = (nt_0_start, nt_0_end, nt_0)
                    nt_1_index = (nt_1_start, nt_1_end, nt_1)
                    for node_0 in chart[nt_0_index]:
                      for node_1 in chart[nt_1_index]:
                        for rule in rhs_to_rules[rhs_key_2_nt]:
                          chart[span_begin, span_end, rule.lhs].append(
                              populate_fn(span_begin, span_end, rule,
                                          [node_0, node_1]))

      if postprocess_fn:
        for nt in nonterminals:
          chart[span_begin, span_end, nt] = postprocess_fn(chart[span_begin,
                                                                 span_end, nt])

      if verbose:
        for nt in nonterminals:
          cell = chart[span_begin, span_end, nt]
          if cell:
            print("Populated (%s,%s): %s - %s" %
                  (span_begin, span_end, nt, cell))

  # Return completed parses.
  return chart[(0, input_len, start_idx)]

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
"""CFG sampler for generating data from grammars."""

import collections
import random

from language.compgen.csl.cky import cfg_rule


def sample_rule(rules, nonterminal_coef=1):
  """Sample a rule. Rules with nonterminals are scaled with nonterminal_coef."""
  if not rules:
    raise ValueError

  if len(rules) == 1:
    return rules[0]

  weights = [
      1 if cfg_rule.get_arity(rule) == 0 else nonterminal_coef for rule in rules
  ]
  sampled_idx = random.choices(range(len(rules)), weights=weights, k=1)[0]
  sampled_rule = rules[sampled_idx]
  return sampled_rule


def sample(parser_rules,
           start_idx,
           rule_values=None,
           max_recursion=1,
           nonterminal_coef=1,
           verbose=False):
  """Sample data from CFG.

  Args:
    parser_rules: A list of CFGRule instances.
    start_idx: Index of non-terminal that is start symbol.
    rule_values: A optional list of rules with 1-1 mapping to parser rules. The
      rule values can be target grammar, QCFG rules, strings, etc. Only used
      when verbose is True for debugging purpose.
    max_recursion: The maximum number of recursion depth of applying CFG rules.
    nonterminal_coef: The scaling coefficient for rules with nonterminals.
    verbose: Print debug logging if True.

  Returns:
    A nested list of CFGRule instances.
  """
  nonterminals_to_rules = collections.defaultdict(list)
  for rule in parser_rules:
    nonterminals_to_rules[rule.lhs].append(rule)

  def expand_nonterminal(nonterminal, recursion=0):
    rules_to_sample = nonterminals_to_rules[nonterminal.idx]

    if recursion == max_recursion:
      # Filter out the rules that have NTs on RHS.
      rules_to_sample_no_nts = [
          rule for rule in nonterminals_to_rules[nonterminal.idx]
          if cfg_rule.get_arity(rule) == 0
      ]
      # If there are no rules for this NT that contain no NTs, then keep
      # recursing. In this case, we may exceed `max_recursion`.
      if rules_to_sample_no_nts:
        rules_to_sample = rules_to_sample_no_nts
    sampled_rule = sample_rule(rules_to_sample,
                               nonterminal_coef=nonterminal_coef)
    if verbose and rule_values is not None:
      print("Recursion %d, Sampled rule: %s" %
            (recursion, rule_values[sampled_rule.idx]))

    output = [sampled_rule]
    for symbol in sampled_rule.rhs:
      if symbol.type == cfg_rule.NON_TERMINAL:
        output.append(expand_nonterminal(symbol, recursion=recursion + 1))
    return output

  start_symbol = cfg_rule.CFGSymbol(idx=start_idx, type=cfg_rule.NON_TERMINAL)
  output = expand_nonterminal(start_symbol, recursion=0)
  return output

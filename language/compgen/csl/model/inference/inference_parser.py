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
"""Utilities for generating k-best targets given scoring model."""

import collections
import heapq
import random

from language.compgen.csl.model.data import parsing_utils
from language.compgen.csl.qcfg import qcfg_parser
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.targets import target_grammar


def get_score(wrapper, rule, children):
  """Equivalent of rule scoring in `forest_utils.forest_score_function`."""
  score = 0.0
  for nonterminal_idx, child in enumerate(children):
    lhs_emb_idx = wrapper.lhs_emb_idx_map[child.rule]
    rhs_emb_idx = wrapper.rhs_emb_idx_map[(rule, nonterminal_idx)]
    application_score = wrapper.application_scores[lhs_emb_idx, rhs_emb_idx]
    score += application_score
    score += child.score
  return score


def get_root_score(wrapper, child):
  return get_score(wrapper, parsing_utils.ROOT_RULE_KEY, [child])


def _print_node(node, indent=0):
  """Print out a ScoredChartNode recursively for debugging."""
  print("%s %s" % ("-" * indent, node))
  for child in node.children:
    _print_node(child, indent=indent+1)


class ScoredChartNode(object):
  """Represents node in chart."""

  def __init__(self, wrapper, rule, children):
    self.score = get_score(wrapper, rule, children)
    self.rule = rule
    self.children = children

  def __str__(self):
    return "%s (%s)" % (self.rule, self.score)

  def __repr__(self):
    return self.__str__()

  def target_string(self):
    """Construct target string recursively."""
    return qcfg_rule.apply_target(
        self.rule, [node.target_string() for node in self.children])


def get_node_fn(wrapper):
  """Return node_fn."""

  def node_fn(unused_span_begin, unused_span_end, rule, children):
    return ScoredChartNode(wrapper, rule, children)

  return node_fn


def get_postprocess_cell_fn(nodes_per_rule_per_cell, max_cell_size=None):
  """Return postprocess_cell_fn."""

  def postprocess_cell_fn(nodes):
    if not nodes:
      return []

    # Dictionary of rules to nodes.
    # Currently, we use a very conservative pruning approach.
    # TODO(petershaw): Can we prune more aggressively to speed up parsing
    # without sacrificing exact inference?
    rule_to_nodes = collections.defaultdict(list)
    for node in nodes:
      rule_to_nodes[node.rule].append(node)
    for rule in rule_to_nodes.keys():
      if len(rule_to_nodes[rule]) > nodes_per_rule_per_cell:
        rule_to_nodes[rule] = heapq.nlargest(
            nodes_per_rule_per_cell, rule_to_nodes[rule], key=lambda x: x.score)
    pruned_nodes = []
    for rule_nodes in rule_to_nodes.values():
      pruned_nodes.extend(rule_nodes)
    if max_cell_size:
      pruned_nodes = heapq.nlargest(
          max_cell_size, pruned_nodes, key=lambda x: x.score)
    return pruned_nodes

  return postprocess_cell_fn


def run_inference(source, wrapper, nodes_per_rule_per_cell=1):
  """Determine set of parses given model.

  The set of output parses will be pruned by score, such that each cell
  of the chart contains only at most `nodes_per_rule_per_cell` nodes
  per last applied rule.

  Args:
    source: Input string.
    wrapper: Instance of InferenceWrapper.
    nodes_per_rule_per_cell: Will prune any nodes per cell per rule above this
      number.

  Returns:
    List of (target string, score) for highest scoring derivations, or None
    if there is no parse.
  """
  tokens = source.split(" ")
  node_fn = get_node_fn(wrapper)
  max_cell_size = wrapper.config.get("max_cell_size", None)
  postprocess_cell_fn = get_postprocess_cell_fn(nodes_per_rule_per_cell,
                                                max_cell_size)
  nodes = qcfg_parser.parse(
      tokens,
      wrapper.rules,
      node_fn=node_fn,
      postprocess_cell_fn=postprocess_cell_fn,
      max_single_nt_applications=wrapper.config["max_single_nt_applications"],
      verbose=False)

  if not nodes:
    return None

  # TODO(petershaw): Consider merging nodes with same target.
  targets_and_scores = [
      (node.target_string(), get_root_score(wrapper, node)) for node in nodes
  ]

  random.shuffle(targets_and_scores)
  return sorted(targets_and_scores, key=lambda x: -x[1])


def get_top_output(source, wrapper, nodes_per_rule_per_cell=1, verbose=False):
  """Return highest scoring target string if can be parsed by target grammar."""
  outputs = get_top_outputs(
      source,
      wrapper,
      topk=1,
      nodes_per_rule_per_cell=nodes_per_rule_per_cell,
      verbose=verbose)
  return outputs[0]


def get_top_outputs(source,
                    wrapper,
                    topk=1,
                    nodes_per_rule_per_cell=1,
                    verbose=False):
  """Return highest-k scoring target strings if can be parsed by target grammar."""
  outputs = run_inference(source, wrapper, nodes_per_rule_per_cell)

  if verbose:
    print("outputs:")
    for target, score in outputs:
      print("%s (%s)" % (target, score))

  if not outputs:
    return [None] * topk

  targets = [output[0] for output in outputs][:topk]

  # Validate target if target CFG provided.
  # TODO(petershaw): Consider returning the highest scoring output that *is*
  # parsable if one is available.
  if wrapper.target_grammar_rules:
    for i, target in enumerate(targets):
      if (target and
          not target_grammar.can_parse(target, wrapper.target_grammar_rules)):
        targets[i] = None
  return targets

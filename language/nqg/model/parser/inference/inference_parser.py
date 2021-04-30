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
"""Utilities for generating one-best targets given neural scoring model."""

import collections

from language.nqg.model.qcfg import qcfg_parser
from language.nqg.model.qcfg import qcfg_rule

ScoredAnchoredRuleApplication = collections.namedtuple(
    "ScoredAnchoredRuleApplication",
    [
        "rule",  # QCFGRule.
        "span_begin",  # Integer.
        "span_end",  # Integer.
        "score",  # Float.
    ])


class ScoredChartNode(object):
  """Represents node in chart."""

  def __init__(self, score_fn, span_begin, span_end, rule, children):
    # Get score.
    application_score = score_fn(rule, span_begin, span_end)
    self.score = application_score
    for node in children:
      self.score += node.score

    # Get target string.
    target_string = qcfg_rule.apply_target(
        rule, [node.target_string for node in children])
    self.target_string = target_string

    application = ScoredAnchoredRuleApplication(rule, span_begin, span_end,
                                                application_score)
    # List of ScoredAnchoredRuleApplication, which can be used to inspect
    # parse tree for a given prediction.
    self.applications = [application]
    for node in children:
      for application in node.applications:
        self.applications.append(application)

  def __str__(self):
    return "%s (%s) [%s]" % (self.target_string, self.score, self.applications)

  def __repr__(self):
    return self.__str__()


def get_node_fn(score_fn):
  """Return node_fn."""

  def node_fn(span_begin, span_end, rule, children):
    return ScoredChartNode(score_fn, span_begin, span_end, rule, children)

  return node_fn


def postprocess_cell_fn(nodes):
  if not nodes:
    return []

  # Prune all nodes except the highest scoring node.
  sorted_nodes = sorted(nodes, key=lambda x: -x.score)
  return [sorted_nodes[0]]


def run_inference(source, rules, score_fn):
  """Determine one-best parse using score_fn.

  Args:
    source: Input string.
    rules: Set of QCFGRules.
    score_fn: Function with inputs (rule, span_begin, span_end) and returns
      float score for a given anchored rule application. Note that `span_begin`
      and `span_end` refer to token indexes, where span_end is exclusive, and
      `rule` is a QCFGRule.

  Returns:
    (target string, score) for highest scoring derivation, or (None, None)
    if there is no derivation for given source.
  """
  tokens = source.split(" ")
  node_fn = get_node_fn(score_fn)
  nodes = qcfg_parser.parse(
      tokens, rules, node_fn=node_fn, postprocess_cell_fn=postprocess_cell_fn)

  if not nodes:
    return None, None

  if len(nodes) > 1:
    raise ValueError("Multiple nodes returned for inference: %s" % nodes)

  return nodes[0].target_string, nodes[0].score

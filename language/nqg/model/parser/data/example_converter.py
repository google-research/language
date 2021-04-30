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
"""Utilities for writing tf.Example files."""

import collections

from language.nqg.model.parser.data import forest_serialization
from language.nqg.model.parser.data import parsing_utils
from language.nqg.model.parser.data import tokenization_utils

import tensorflow as tf


def _pad_values(values, padded_length):
  if len(values) > padded_length:
    raise ValueError("length %s is > %s" % (len(values), padded_length))
  for _ in range(len(values), padded_length):
    values.append(0)
  return values


def _create_int_feature(values, padded_length):
  values = _pad_values(values, padded_length)
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def get_rule_to_idx_map(rules):
  rule_to_idx_map = {}
  for idx, rule in enumerate(rules):
    rule_to_idx_map[rule] = idx + 1  # Reserve 0 for padding.
  return rule_to_idx_map


def _get_applications(root_node, rule_to_idx_map, token_start_wp_idx,
                      token_end_wp_idx):
  """Returns structures for anchored applications."""
  # Traverse all nodes.
  node_stack = [root_node]
  seen_fingerprints = set()
  # Set of (span_begin, span_end, rule).
  applications = set()
  while node_stack:
    node = node_stack.pop()
    fingerprint = id(node)
    if fingerprint in seen_fingerprints:
      continue
    seen_fingerprints.add(fingerprint)

    if isinstance(node, parsing_utils.AggregationNode):
      for child in node.children:
        node_stack.append(child)
    elif isinstance(node, parsing_utils.RuleApplicationNode):
      for child in node.children:
        node_stack.append(child)
      applications.add((node.span_begin, node.span_end, node.rule))
    else:
      raise ValueError("Unexpected node type.")

  # Map of (span_begin, span_end, rule) to integer idx.
  application_key_to_idx_map = {}

  # Lists of integers.
  application_span_begin = []
  application_span_end = []
  application_rule_idx = []

  # Sort applications to avoid non-determinism.
  applications = sorted(applications)
  for idx, (span_begin, span_end, rule) in enumerate(applications):
    application_key_to_idx_map[(span_begin, span_end, rule)] = idx

    application_span_begin.append(token_start_wp_idx[span_begin])
    # token_end_wp_idx is an *inclusive* idx.
    # span_end is an *exclusive* idx.
    # application_span_end is an *inclusive* idx.
    application_span_end.append(token_end_wp_idx[span_end - 1])
    rule_idx = rule_to_idx_map[rule]
    application_rule_idx.append(rule_idx)

  return (application_key_to_idx_map, application_span_begin,
          application_span_end, application_rule_idx)


def _convert_to_tf_example(example, tokenizer, rules, config, max_sizes=None):
  """Return tf.Example generated for input (source, target)."""
  source = example[0]
  target = example[1]
  tokens = source.split(" ")
  num_tokens = len(tokens)

  # Tokenize.
  (wordpiece_ids, num_wordpieces, token_start_wp_idx,
   token_end_wp_idx) = tokenization_utils.get_wordpiece_inputs(
       tokens, tokenizer)

  # Run chart parser.
  target_node = parsing_utils.get_target_node(source, target, rules)
  if not target_node:
    raise ValueError("No parse returned for target for example: (%s, %s)" %
                     (source, target))
  merged_node = parsing_utils.get_merged_node(source, rules)

  # Get anchored applications.
  rule_to_idx_map = get_rule_to_idx_map(rules)
  (application_key_to_idx_map, application_span_begin, application_span_end,
   application_rule_idx) = _get_applications(merged_node, rule_to_idx_map,
                                             token_start_wp_idx,
                                             token_end_wp_idx)
  num_applications = len(application_span_begin)

  def application_idx_fn(span_begin, span_end, rule):
    return application_key_to_idx_map[(span_begin, span_end, rule)]

  # Get numerator forest.
  (nu_node_type, nu_node_1_idx, nu_node_2_idx, nu_application_idx,
   nu_num_nodes) = forest_serialization.get_forest_lists(
       target_node, num_tokens, application_idx_fn)
  # Get denominator forest.
  (de_node_type, de_node_1_idx, de_node_2_idx, de_application_idx,
   de_num_nodes) = forest_serialization.get_forest_lists(
       merged_node, num_tokens, application_idx_fn)

  # Create features dict.
  features = collections.OrderedDict()
  features["wordpiece_ids"] = _create_int_feature(wordpiece_ids,
                                                  config["max_num_wordpieces"])
  features["num_wordpieces"] = _create_int_feature([num_wordpieces], 1)

  features["application_span_begin"] = _create_int_feature(
      application_span_begin, config["max_num_applications"])
  features["application_span_end"] = _create_int_feature(
      application_span_end, config["max_num_applications"])
  features["application_rule_idx"] = _create_int_feature(
      application_rule_idx, config["max_num_applications"])

  features["nu_node_type"] = _create_int_feature(
      nu_node_type, config["max_num_numerator_nodes"])
  features["nu_node_1_idx"] = _create_int_feature(
      nu_node_1_idx, config["max_num_numerator_nodes"])
  features["nu_node_2_idx"] = _create_int_feature(
      nu_node_2_idx, config["max_num_numerator_nodes"])
  features["nu_application_idx"] = _create_int_feature(
      nu_application_idx, config["max_num_numerator_nodes"])
  features["nu_num_nodes"] = _create_int_feature([nu_num_nodes], 1)

  features["de_node_type"] = _create_int_feature(
      de_node_type, config["max_num_denominator_nodes"])
  features["de_node_1_idx"] = _create_int_feature(
      de_node_1_idx, config["max_num_denominator_nodes"])
  features["de_node_2_idx"] = _create_int_feature(
      de_node_2_idx, config["max_num_denominator_nodes"])
  features["de_application_idx"] = _create_int_feature(
      de_application_idx, config["max_num_denominator_nodes"])
  features["de_num_nodes"] = _create_int_feature([de_num_nodes], 1)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))

  # Update max sizes.
  if max_sizes is not None:
    max_sizes["num_wordpieces"] = max(max_sizes["num_wordpieces"],
                                      num_wordpieces)
    max_sizes["num_applications"] = max(max_sizes["num_applications"],
                                        num_applications)
    max_sizes["nu_num_nodes"] = max(max_sizes["nu_num_nodes"], nu_num_nodes)
    max_sizes["de_num_nodes"] = max(max_sizes["de_num_nodes"], de_num_nodes)

  return tf_example


class ExampleConverter(object):
  """Converts inputs to tf.Example protos."""

  def __init__(self, rules, tokenizer, config):
    self.rules = rules
    self.tokenizer = tokenizer
    self.config = config
    self.max_sizes = collections.defaultdict(int)

  def convert(self, example):
    """Return tf.Example or Raise."""
    tf_example = _convert_to_tf_example(example, self.tokenizer, self.rules,
                                        self.config, self.max_sizes)
    return tf_example

  def print_max_sizes(self):
    """Print max sizes which is useful for determining necessary padding."""
    print("max_sizes: %s" % self.max_sizes)

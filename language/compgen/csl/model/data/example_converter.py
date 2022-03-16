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

from language.compgen.csl.model.data import forest_serialization
from language.compgen.csl.model.data import parsing_utils
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


def _print_tree(root_node, prefix=""):
  """Print tree for debugging."""
  print("%s%s" % (prefix, str(root_node)))
  new_prefix = prefix + "-"
  for child in root_node.children:
    _print_tree(child, new_prefix)


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


def _create_int_seq_feature(values, padded_seq_length, padded_length):
  if not isinstance(values, list):
    values = values.tolist()
  padded_list = [0] * padded_length
  for _ in range(len(values), padded_seq_length):
    values.append(padded_list)
  features = [_create_int_feature(value, padded_length) for value in values]
  return tf.train.FeatureList(feature=features)


def _transpose(list_of_lists):
  """Transpose list of lists."""
  return [list(l) for l in zip(*list_of_lists)]


def get_rhs_emb_idx_map(rules):
  """Return map from (rule, nt_idx) to integer rhs_emb_idx."""
  # Key is (rule, nt_idx).
  rhs_emb_idx_map = {}
  # Add special entry for "root rule".
  rhs_emb_idx_map[(parsing_utils.ROOT_RULE_KEY, 0)] = 0
  # Add entries for other rules.
  max_num_nts = 0
  idx = 1
  for rule in sorted(rules):
    num_nts = qcfg_rule.get_num_nts(rule.source)
    max_num_nts = max(max_num_nts, num_nts)
    for nt_idx in range(num_nts):
      rhs_emb_idx_map[(rule, nt_idx)] = idx
      idx += 1
  print("max_num_nts: %s" % max_num_nts)
  print("len(rhs_emb_idx_map): %s" % len(rhs_emb_idx_map))
  return rhs_emb_idx_map


def get_lhs_emb_idx_map(rules):
  # Key is rule.
  lhs_emb_idx_map = {}
  for idx, rule in enumerate(sorted(rules)):
    lhs_emb_idx_map[rule] = idx
  print("len(lhs_emb_idx_map): %s" % len(lhs_emb_idx_map))
  return lhs_emb_idx_map


def get_input_features(source,
                       target,
                       rules,
                       rhs_emb_idx_map,
                       lhs_emb_idx_map,
                       config,
                       verbose=False):
  """Return tuple of input features as lists."""
  # Run chart parser.
  target_node = parsing_utils.get_target_node(
      source,
      target,
      rules,
      max_single_nt_applications=config["max_single_nt_applications"])
  if not target_node:
    raise ValueError("No parse returned for target for example: (%s, %s)" %
                     (source, target))
  if verbose:
    print("target_node\n%s" % _print_tree(target_node))

  # Get parse forest.
  return forest_serialization.get_forest_lists(
      target_node,
      rhs_emb_idx_map,
      lhs_emb_idx_map,
      max_num_nts=config["max_num_nts"])


def _convert_to_tf_example(example,
                           rules,
                           rhs_emb_idx_map,
                           lhs_emb_idx_map,
                           config,
                           max_sizes=None):
  """Return tf.Example generated for input (source, target)."""
  source, target = example
  (node_type_list, node_idx_list, rhs_emb_idx_list, lhs_emb_idx_list,
   num_nodes) = get_input_features(source, target, rules, rhs_emb_idx_map,
                                   lhs_emb_idx_map, config)

  # Create features dict.
  features = collections.OrderedDict()

  features["node_type_list"] = _create_int_seq_feature(
      [node_type_list], 1, config["max_num_numerator_nodes"])
  features["node_idx_list"] = _create_int_seq_feature(
      _transpose(node_idx_list), config["max_num_nts"],
      config["max_num_numerator_nodes"])
  features["rhs_emb_idx_list"] = _create_int_seq_feature(
      _transpose(rhs_emb_idx_list), config["max_num_nts"],
      config["max_num_numerator_nodes"])
  features["lhs_emb_idx_list"] = _create_int_seq_feature(
      _transpose(lhs_emb_idx_list), config["max_num_nts"],
      config["max_num_numerator_nodes"])
  features["num_nodes"] = _create_int_seq_feature([[num_nodes]], 1, 1)

  tf_example = tf.train.SequenceExample(
      feature_lists=tf.train.FeatureLists(feature_list=features))

  # Update max sizes.
  if max_sizes is not None:
    max_sizes["num_nodes"] = max(max_sizes["num_nodes"], num_nodes)

  return tf_example


class ExampleConverter(object):
  """Converts inputs to tf.Example protos."""

  def __init__(self, rules, config):
    self.rules = rules
    self.config = config
    self.max_sizes = collections.defaultdict(int)
    self.rhs_emb_idx_map = get_rhs_emb_idx_map(rules)
    self.lhs_emb_idx_map = get_lhs_emb_idx_map(rules)

  def convert(self, example):
    """Return tf.Example or Raise."""
    tf_example = _convert_to_tf_example(example, self.rules,
                                        self.rhs_emb_idx_map,
                                        self.lhs_emb_idx_map, self.config,
                                        self.max_sizes)
    return tf_example

  def print_max_sizes(self):
    """Print max sizes which is useful for determining necessary padding."""
    print("max_sizes: %s" % self.max_sizes)

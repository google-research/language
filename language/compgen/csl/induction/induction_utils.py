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
"""Utilities for grammar induction."""

from absl import logging
from language.compgen.csl.induction import action_utils
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.qcfg import qcfg_rule


def _example_to_rule(source_str, target_str):
  """Convert (source, target) example to a QCFGRule."""
  return qcfg_rule.QCFGRule(
      tuple(source_str.split()), tuple(target_str.split()), arity=0)


def get_example_rules(examples):
  """Returns the example rules with one rule for each example."""
  current_rules = set()
  # Add a rule for each example.
  for source_str, target_str in examples:
    rule = _example_to_rule(source_str, target_str)
    current_rules.add(rule)
  logging.info("Added example rules: %s.", len(current_rules))
  return current_rules


def get_examples_up_to_partition(partition_to_examples, current_partition):
  """Returns examples up to the current partition."""
  examples = []
  for partition in range(current_partition + 1):
    examples.extend(list(partition_to_examples[partition]))
  return examples


class InductionState(object):
  """Dataclass that tracks induction state."""

  def __init__(self, rule_prefix, config):
    self.current_partition = 0
    self.current_step_in_partition = 0
    self.current_rules = set()
    self._rule_prefix = rule_prefix
    self._config = config

  def get_rule_file(self, partition, step_in_partition):
    return "%s-%s-%s" % (self._rule_prefix, partition, step_in_partition)

  def restore_state(self, partition, step_in_partition):
    """Restore from a induction state and set current partition and step."""
    self.current_rules |= set(
        qcfg_file.read_rules(self.get_rule_file(partition, step_in_partition)))
    if step_in_partition == self._config["max_num_steps"] - 1:
      self.current_partition = partition + 1
      self.current_step_in_partition = 0
    else:
      self.current_partition = partition
      self.current_step_in_partition = step_in_partition + 1
    logging.info("Restored from partition %s step %s.", partition,
                 step_in_partition)
    logging.info("Current partition: %s, current step: %s.",
                 self.current_partition, self.current_step_in_partition)

  def save_rules(self):
    rule_file = self.get_rule_file(self.current_partition,
                                   self.current_step_in_partition)
    qcfg_file.write_rules(self.current_rules, rule_file)


def aggregate_actions(actions):
  """Returns an action that aggregates all actions."""
  rules_to_add = set()
  rules_to_remove = set()
  for action in actions:
    if action:
      rules_to_add |= action.rules_to_add
      rules_to_remove |= action.rules_to_remove
  return action_utils.Action(
      rules_to_add=rules_to_add, rules_to_remove=rules_to_remove)


def execute_action(action, current_rules):
  """Execute an action."""
  new_rules = set(current_rules)
  new_rules |= action.rules_to_add
  new_rules -= action.rules_to_remove
  metrics = {
      "num_rules_to_add": len(action.rules_to_add),
      "num_rules_to_remove": len(action.rules_to_remove),
      "num_rules": len(new_rules),
  }
  return metrics, new_rules

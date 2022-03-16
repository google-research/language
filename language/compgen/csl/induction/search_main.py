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
"""Induce and write QCFG rules."""

from absl import app
from absl import flags
from absl import logging

from language.compgen.csl.common import json_utils
from language.compgen.csl.induction import greedy_policy
from language.compgen.csl.induction import induction_utils
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.targets import target_grammar
from language.compgen.nqg.tasks import tsv_utils

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file of examples.")

flags.DEFINE_string("output", "", "Output rule txt file.")

flags.DEFINE_string("config", "", "Path to config file.")

flags.DEFINE_list("seed_rules_file", "",
                  "Optional list of seed rules txt files.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")

flags.DEFINE_integer("restore_partition", 0, "The partition to restore.")

flags.DEFINE_integer("restore_step", 0, "The step of partition to restore.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")


def execute_step(policy, current_rules):
  """Induce rules for one step."""
  actions = []
  for rule in current_rules:
    action = policy.select_action(rule, current_rules)
    if action:
      actions.append(action)
  action = induction_utils.aggregate_actions(actions)
  return induction_utils.execute_action(action, current_rules)


def induce_rules_for_partition(policy, induction_state):
  """Grammar induction for one partition."""
  while (induction_state.current_step_in_partition <
         policy.config["max_num_steps"]):
    metrics, induction_state.current_rules = execute_step(
        policy, induction_state.current_rules)
    if (policy.config["save_every_step"] and
        induction_state.current_step_in_partition %
        policy.config["save_every_step"] == 0):
      induction_state.save_rules()
    for metric_name, metric_value in metrics.items():
      logging.info("%s at step %s: %s", metric_name,
                   induction_state.current_step_in_partition, metric_value)
    num_rules_to_add = metrics["num_rules_to_add"]
    num_rules_to_remove = metrics["num_rules_to_remove"]
    if num_rules_to_add == 0 and num_rules_to_remove == 0:
      break
    induction_state.current_step_in_partition += 1
  induction_state.save_rules()


def main(unused_argv):
  """Induce and write set of rules."""
  examples = tsv_utils.read_tsv(FLAGS.input)
  config = json_utils.json_file_to_dict(FLAGS.config)
  if not config.get("allow_duplicate_examples", True):
    examples = set([tuple(ex) for ex in examples])
  examples = sorted(examples, key=lambda e: (len(e[0]), e))

  seed_rules = set()
  # Add mannual seed rules.
  if FLAGS.seed_rules_file:
    for seed_rules_file in FLAGS.seed_rules_file:
      seed_rules |= set(qcfg_file.read_rules(seed_rules_file))

  target_grammar_rules = (
      target_grammar.load_rules_from_file(FLAGS.target_grammar)
      if FLAGS.target_grammar else None)

  num_partitions = config.get("num_partitions", 1)
  partition_to_examples = np.array_split(examples, num_partitions)
  induction_state = induction_utils.InductionState(FLAGS.output, config)
  if FLAGS.restore_partition or FLAGS.restore_step:
    # Restore from an existing induction state.
    induction_state.restore_state(FLAGS.restore_partition, FLAGS.restore_step)
  else:
    # Initialize the induction state with manual seed rules.
    induction_state.current_rules = seed_rules.copy()

  while induction_state.current_partition < num_partitions:
    current_examples = induction_utils.get_examples_up_to_partition(
        partition_to_examples, induction_state.current_partition)
    logging.info("Partition: %s, number of examples: %s.",
                 induction_state.current_partition, len(current_examples))
    # At the first step of each partition, we add a rule corresponding to
    # each example in the partition.
    if induction_state.current_step_in_partition == 0:
      induction_state.current_rules |= induction_utils.get_example_rules(
          partition_to_examples[induction_state.current_partition])
    policy = greedy_policy.GreedyPolicy(
        config,
        current_examples,
        seed_rules,
        target_grammar_rules,
        verbose=FLAGS.verbose)
    induce_rules_for_partition(policy, induction_state)
    induction_state.current_partition += 1
    induction_state.current_step_in_partition = 0
  qcfg_file.write_rules(induction_state.current_rules, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

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
"""Induce and write QCFG rules with Beam."""

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
from language.compgen.csl.common import beam_utils
from language.compgen.csl.common import json_utils
from language.compgen.csl.induction import greedy_policy
from language.compgen.csl.induction import induction_utils
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.targets import target_grammar
from language.compgen.nqg.tasks import tsv_utils
import numpy as np
from tensorflow.io import gfile

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

flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")


class InduceRulesForPartition(beam.PTransform):
  """Grammar induction for one partition."""

  def __init__(self, policy, induction_state):
    self.policy = policy
    self.induction_state = induction_state

  def expand(self, root):
    max_num_steps = self.policy.config["max_num_steps"]
    save_every_step = self.policy.config["save_every_step"]

    current_rules = root | "CreateRules" >> beam.Create(
        self.induction_state.current_rules)
    while self.induction_state.current_step_in_partition < max_num_steps:
      step = self.induction_state.current_step_in_partition
      current_rules = (
          current_rules
          | "SelectAction_%s" % step >> beam.Map(
              self.policy.select_action,
              current_rules=beam.pvalue.AsList(current_rules))
          | "AggregateActions_%s" % step >> beam.CombineGlobally(
              induction_utils.aggregate_actions)
          | "ExecuteAction_%s" % step >> beam.FlatMap(
              self.execute_action,
              current_rules=beam.pvalue.AsList(current_rules)))
      if ((save_every_step and step % save_every_step == 0) or
          (step == max_num_steps - 1)):
        rule_file = self.induction_state.get_rule_file(
            self.induction_state.current_partition, step)
        _ = (
            current_rules
            | "Write_%s" % step >> beam.io.WriteToText(
                rule_file, shard_name_template=""))
      self.induction_state.current_step_in_partition += 1

  def execute_action(self, action, current_rules):
    metrics, new_rules = induction_utils.execute_action(action, current_rules)
    beam_utils.dict_to_beam_counts(metrics, "InduceRules")
    return new_rules


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
    # Restore induction state from the last partition, since the induced
    # rules are saved with Beam I/O.
    # TODO(linluqiu): Investigate performance improvements from using a single
    # Beam pipeline for all partitions without requiring rules to be explicitly
    # written and read from disk.
    if (induction_state.current_partition > 0 and
        induction_state.current_partition != FLAGS.restore_partition):
      induction_state.restore_state(induction_state.current_partition - 1,
                                    config["max_num_steps"] - 1)
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

    # pylint: disable=cell-var-from-loop
    def _induce_rules(pipeline):
      _ = (
          pipeline | "InduceRulesForPartition" >> InduceRulesForPartition(
              policy, induction_state))

    # pylint: enable=cell-var-from-loop
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        FLAGS.pipeline_options)
    with beam.Pipeline(pipeline_options) as pipeline:
      _induce_rules(pipeline)

    counters = pipeline.result.metrics().query()["counters"]
    for counter in counters:
      logging.info("%s %s: %s", counter.key.step, counter.key.metric.name,
                   counter.committed)
    rule_file = induction_state.get_rule_file(induction_state.current_partition,
                                              config["max_num_steps"] - 1)
    gfile.copy(rule_file, FLAGS.output, overwrite=True)
    induction_state.current_partition += 1


if __name__ == "__main__":
  app.run(main)

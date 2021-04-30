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

from language.nqg.model.induction import induction_utils

from language.nqg.model.qcfg import qcfg_file

from language.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file of examples.")
flags.DEFINE_string("output", "", "Output rule txt file.")
flags.DEFINE_integer("sample_size", 500,
                     "Number of examples to sample for induction.")
flags.DEFINE_integer("max_iterations", 10000,
                     "Maximum number of grammar induction iterations.")
flags.DEFINE_integer("min_delta", 0,
                     "Minimum codelength delta to add a new rule.")
flags.DEFINE_integer("terminal_codelength", 32,
                     "Codelength coeffecient for terminals.")
flags.DEFINE_integer("non_terminal_codelength", 1,
                     "Codelength coeffecient for non-terminals.")
flags.DEFINE_integer(
    "parse_sample", 10,
    "Number of examples to sample for estimating target encoding codelength.")
flags.DEFINE_bool(
    "allow_repeated_target_nts", True,
    "Whether to allow multiple non-terminals with same index in targets.")
flags.DEFINE_bool("seed_exact_match", True,
                  "Whether to seed induction with exact match rules.")
flags.DEFINE_bool("balance_parens", True,
                  "Whether to require rules to have balanced parentheses.")


def induce_and_write_rules():
  """Induce and write set of rules."""
  examples = tsv_utils.read_tsv(FLAGS.input)
  config = induction_utils.InductionConfig(
      sample_size=FLAGS.sample_size,
      max_iterations=FLAGS.max_iterations,
      min_delta=FLAGS.min_delta,
      terminal_codelength=FLAGS.terminal_codelength,
      non_terminal_codelength=FLAGS.non_terminal_codelength,
      parse_sample=FLAGS.parse_sample,
      allow_repeated_target_nts=FLAGS.allow_repeated_target_nts,
      seed_exact_match=FLAGS.seed_exact_match,
      balance_parens=FLAGS.balance_parens,
  )
  induced_rules = induction_utils.induce_rules(examples, config)
  qcfg_file.write_rules(induced_rules, FLAGS.output)


def main(unused_argv):
  induce_and_write_rules()


if __name__ == "__main__":
  app.run(main)

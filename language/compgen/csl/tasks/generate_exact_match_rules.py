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
"""Generates exact match seed rules."""

from absl import app
from absl import flags
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.tasks import exact_match_utils
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output txt file.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = exact_match_utils.get_exact_match_rules(examples)
  # Sort by target.
  rules = list(rules)
  rules.sort(key=lambda x: x.target)
  qcfg_file.write_rules(rules, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

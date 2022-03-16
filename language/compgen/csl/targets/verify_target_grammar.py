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
"""Verify all targets in input TSV can be parsed by given target CFG."""

from absl import app
from absl import flags
from language.compgen.csl.targets import target_grammar
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input TSV file.")

flags.DEFINE_string("target_grammar_file", "", "Target grammar txt file.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")


def main(unused_argv):
  input_examples = tsv_utils.read_tsv(FLAGS.input)
  rules = target_grammar.load_rules_from_file(FLAGS.target_grammar_file)
  for idx, (_, target) in enumerate(input_examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break
    print("Processing example %s." % idx)
    can_parse = target_grammar.can_parse(target, rules)
    if not can_parse:
      raise ValueError("Cannot parse: %s" % target)


if __name__ == "__main__":
  app.run(main)

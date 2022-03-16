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
"""Compute % of examples in a dataset that can be derived by a given QCFG."""

from absl import app
from absl import flags
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.qcfg import qcfg_parser
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_integer("limit", 100, "End processing at this example index.")

flags.DEFINE_integer("offset", 0, "Start processing at this example index.")

flags.DEFINE_string("rules", "", "Grammar rules txt file.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = qcfg_file.read_rules(FLAGS.rules)
  print("Rules: %s" % rules)

  num_examples = 0
  num_covered = 0

  for idx, example in enumerate(examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break
    print("Processing example %s." % idx)
    print("Source: %s" % example[0])
    print("Target: %s" % example[1])

    source = example[0]
    gold_target = example[1]

    can_parse = qcfg_parser.can_parse(source, gold_target, rules, verbose=False)

    num_examples += 1

    if can_parse:
      num_covered += 1
    else:
      print("Output set does not contain gold target.")

  print("%s covered out of %s" % (num_covered, num_examples))


if __name__ == "__main__":
  app.run(main)

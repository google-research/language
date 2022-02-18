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
"""Read and write QCFG grammars to/from human readable txt files."""

from language.compgen.nqg.model.qcfg import qcfg_rule

from tensorflow.io import gfile


def read_rules(filename):
  """Read rule txt file to list of rules."""
  rules = []
  with gfile.GFile(filename, "r") as txt_file:
    for line in txt_file:
      line = line.rstrip()
      rule = qcfg_rule.rule_from_string(line)
      rules.append(rule)
  print("Loaded %s rules from %s." % (len(rules), filename))
  return rules


def write_rules(rules, filename):
  """Write rules to txt file."""
  with gfile.GFile(filename, "w") as txt_file:
    for rule in rules:
      line = "%s\n" % str(rule)
      txt_file.write(line)
  print("Wrote %s rules to %s." % (len(rules), filename))

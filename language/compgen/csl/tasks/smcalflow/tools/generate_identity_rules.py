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
"""Generates identity seed rules."""

from absl import app
from absl import flags
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.qcfg import qcfg_rule
from language.compgen.csl.tasks.smcalflow.tools import string_utils
from language.compgen.nqg.tasks import tsv_utils
import regex as re

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output txt file.")


def get_string_rules(source, target):
  """Return string seed rules based on exact match."""
  rules = set()
  for prefix in ("PersonName", "String", "LocationKeyphrase", "Month",
                 "DayOfWeek"):
    regex = r'%s " (.+?) "' % prefix
    matches = re.findall(regex, target)
    for match in matches:
      source_rhs = string_utils.format_source(match)
      if source_rhs.lower() not in source.lower():
        print("`%s` is not in `%s`." % (source_rhs, source))
        continue
      target_rhs = '%s " %s "' % (prefix, match)
      rules.add(
          qcfg_rule.QCFGRule(
              tuple(source_rhs.split(" ")), tuple(target_rhs.split(" ")), 0))
  return rules


def get_datetime_exact_match(source, target):
  """Return seed rules based on exact match for dates and times."""
  rules = set()
  for arg in ("date", "time"):
    regex = r":%s \( (.+?) \)" % arg
    matches = re.findall(regex, target)
    for match in matches:
      source_rhs = string_utils.format_source(match)
      if source_rhs.lower() not in source.lower():
        continue
      target_rhs = match
      rules.add(
          qcfg_rule.QCFGRule(
              tuple(source_rhs.split(" ")), tuple(target_rhs.split(" ")), 0))
  return rules


def get_number_rules(source, target):
  """Return number seed rules based on exact match."""
  rules = set()

  # First, match numbers without decimals.
  matches = re.findall(r"Number ([0-9]+?) ", target)
  for match in matches:
    if match not in source:
      print("`%s` is not in `%s`." % (match, source))
      continue
    target_rhs = "Number %s" % match
    source_rhs = match
    rules.add(
        qcfg_rule.QCFGRule(
            tuple(source_rhs.split(" ")), tuple(target_rhs.split(" ")), 0))

  # Second, try to match numbers with 0 as decimal.
  matches = re.findall(r"Number ([0-9]+?).0 ", target)
  for match in matches:
    if match not in source:
      continue
    target_rhs = "Number %s.0" % match
    source_rhs = match
    rules.add(
        qcfg_rule.QCFGRule(
            tuple(source_rhs.split(" ")), tuple(target_rhs.split(" ")), 0))

  # Finally, match numbers with decimals.
  matches = re.findall(r"Number ([0-9]+\.[0-9]+?) ", target)
  for match in matches:
    if match not in source:
      print("`%s` is not in `%s`." % (match, source))
      continue
    target_rhs = "Number %s" % match
    source_rhs = match
    rules.add(
        qcfg_rule.QCFGRule(
            tuple(source_rhs.split(" ")), tuple(target_rhs.split(" ")), 0))

  return rules


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = set()

  for source, target in examples:
    rules |= get_number_rules(source, target)
    rules |= get_string_rules(source, target)
    rules |= get_datetime_exact_match(source, target)

  # Sort by target.
  rules = list(rules)
  rules.sort(key=lambda x: x.target)
  qcfg_file.write_rules(rules, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

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
"""Filter examples where output contains literals that aren't in input."""

from absl import app
from absl import flags

from language.compgen.csl.tasks.smcalflow.tools import string_utils
from language.compgen.nqg.tasks import tsv_utils

import regex as re

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "TSV file.")

flags.DEFINE_string("output", "", "TSV file.")


def check_person_name(source, target):
  """Check if example does not have exact match for PersonName.

  The filtered example could be: Who is her boss ? ### ( Yield :output (
  FindManager :recipient ( Execute :intension ( refer ( extensionConstraint (
  RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # (
  PersonName " Angelina " ) ) ) ) ) ) ).

  Args:
    source: source string.
    target: target string.

  Returns:
    False if contains strings that are not exact match for PersonName, True
    otherwise.
  """
  for prefix in ("PersonName",):
    regex = r'%s " (.+?) "' % prefix
    matches = re.findall(regex, target)
    for match in matches:
      source_rhs = string_utils.format_source(match)
      if source_rhs.lower() not in source.lower():
        print("`%s` is not in `%s`. %s" % (source_rhs, source, target))
        return False
  return True


def check_year(source, target):
  """Check if example does not have exact match for year.

  The filtered example usually requires additional context, e.g. Can you create
  an event at 11 for May 4 th ? ### ( Yield :output ( CreateCommitEventWrapper
  :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :start (
  ?= ( DateAtTimeWithDefaults :date ( MDY :day # ( Number 4 ) :month # ( Month "
  MAY " ) :year # ( Number 2019 ) ) :time ( NumberAM :number # ( Number 11 ) ) )
  ) ) ) ) ).

  Args:
    source: source string.
    target: target string.

  Returns:
    False if contains strings that are not exact match for PersonName, True
    otherwise.
  """
  for arg in ("year",):
    regex = r":%s # \( Number 20(.+?)\.?0? \)" % arg
    matches = re.findall(regex, target)
    for match in matches:
      if match not in source:
        print("`%s` is not in `%s`. %s" % (match, source, target))
        return False
  return True


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  new_examples = []
  for source, target in examples:
    if check_person_name(source, target) and check_year(source, target):
      new_examples.append((source, target))
  tsv_utils.write_tsv(new_examples, FLAGS.output)
  num_examples = len(examples)
  num_new_examples = len(new_examples)
  print("original examples: %d." % num_examples)
  print("new examples: %d." % num_new_examples)
  print("filtered examples: %d." % (num_examples - num_new_examples))


if __name__ == "__main__":
  app.run(main)

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
"""Split examples based on whether they are single or cross domain."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "TSV file.")

flags.DEFINE_string("output_single", "", "TSV file.")

flags.DEFINE_string("output_cross", "", "TSV file.")


def is_cross_domain(target):
  if "CreateCommitEventWrapper" not in target:
    return False
  for func in ("FindReports", "FindManager", "FindTeamOf"):
    if func in target:
      return True
  return False


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  single_domain_examples = []
  cross_domain_examples = []
  for source, target in examples:
    if is_cross_domain(target):
      cross_domain_examples.append((source, target))
    else:
      single_domain_examples.append((source, target))
  print("len(cross_domain_examples): %s" % len(cross_domain_examples))
  print("len(single_domain_examples): %s" % len(single_domain_examples))
  tsv_utils.write_tsv(cross_domain_examples, FLAGS.output_cross)
  tsv_utils.write_tsv(single_domain_examples, FLAGS.output_single)


if __name__ == "__main__":
  app.run(main)

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
"""Retokenize inputs by separating on punctuation."""

from absl import app
from absl import flags

from language.compgen.csl.tasks.smcalflow.tools import string_utils
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "TSV file.")

flags.DEFINE_string("output", "", "TSV file.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  new_examples = []
  for source, target in examples:
    new_source = string_utils.format_source(source)
    new_examples.append((new_source, target))
  tsv_utils.write_tsv(new_examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

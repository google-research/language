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
"""Format targets so they are encoded better by T5's SPM."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "TSV file.")

flags.DEFINE_string("output", "", "TSV file.")


def format_target(target):
  """Reformat targets."""
  # """Switches OOV T5 tokens to in-vocabulary tokens."""
  target = target.replace("<", "lb")
  target = target.replace(">", "rb")
  target = target.replace("~", "sim")

  return target


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  new_examples = []
  for source, target in examples:
    new_target = format_target(target)
    new_examples.append((source, new_target))
  tsv_utils.write_tsv(new_examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

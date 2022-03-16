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
"""Preprocess the COGS dataset."""
from absl import app
from absl import flags

from language.compgen.csl.tasks.cogs.tools import cogs_converter
from language.compgen.nqg.tasks import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "TSV file.")

flags.DEFINE_string("output", "", "TSV file.")


def main(_):
  examples = tsv_utils.read_tsv(FLAGS.input, expected_num_columns=3)
  new_examples = []
  for source, target, category in examples:
    if category == "primitive":
      if len(source.split()) != 1:
        raise ValueError(f"Invalid primitive: {source}")
      new_target = source
    else:
      new_target = cogs_converter.cogs_lf_to_funcall(target)
    new_examples.append((source, new_target))
  tsv_utils.write_tsv(new_examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

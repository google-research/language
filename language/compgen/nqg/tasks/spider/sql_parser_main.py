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
"""Run SQL parser on dataset to verify all targets have exactly 1 parse."""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils
from language.compgen.nqg.tasks.spider import sql_parser

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_integer("offset", 0, "Example index to start at. Ignored if 0.")

flags.DEFINE_integer("limit", 0, "Example index to stop at. Ignored if 0.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  for idx, (_, target) in enumerate(examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break
    print("Processing example %s." % idx)

    try:
      _ = sql_parser.parse_sql(target)
    except ValueError as e:
      print(e)
      # Retry parsing with verbose debugging.
      _ = sql_parser.parse_sql(target, verbose=True)


if __name__ == "__main__":
  app.run(main)

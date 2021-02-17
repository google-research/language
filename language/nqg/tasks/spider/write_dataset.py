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
"""Write Spider-SSP dataset in TSV format."""

import json

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils
from language.nqg.tasks.spider import database_constants

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "", "Path to Spider json examples.")

flags.DEFINE_string("output", "", "Output tsv file.")


def normalize_whitespace(source):
  tokens = source.split()
  return " ".join(tokens)


def load_json(filepath):
  with gfile.GFile(filepath, "r") as reader:
    text = reader.read()
  return json.loads(text)


def main(unused_argv):
  examples_json = load_json(FLAGS.examples)
  examples = []
  for example_json in examples_json:
    database = example_json["db_id"]
    source = example_json["question"]
    target = example_json["query"]

    # Skip if database not in set of databases with >= 50 examples.
    if database not in database_constants.DATABASES:
      continue

    # Prepend database.
    source = "%s: %s" % (database, source)

    target = normalize_whitespace(target)
    examples.append((source.lower(), target.lower()))

  tsv_utils.write_tsv(examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

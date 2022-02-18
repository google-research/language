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
"""Print out counts of examples per database."""

import collections
import json

from absl import app
from absl import flags

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "", "Path to Spider json examples.")

flags.DEFINE_integer("limit", 50, "")


def load_json(filepath):
  with gfile.GFile(filepath, "r") as reader:
    text = reader.read()
  return json.loads(text)


def main(unused_argv):
  examples = load_json(FLAGS.examples)
  db_to_count = collections.defaultdict(int)
  for example in examples:
    db_to_count[example["db_id"]] += 1
  print("db_to_count: %s" % db_to_count)

  print("Databases above example limit:")
  total_examples = 0
  for database, count in db_to_count.items():
    if count >= FLAGS.limit:
      print(database)
      total_examples += count
  print("Total examples: %s" % total_examples)


if __name__ == "__main__":
  app.run(main)

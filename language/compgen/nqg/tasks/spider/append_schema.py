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
"""Serialize and append database schema to inputs."""

import collections
import json

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output tsv file.")

flags.DEFINE_string("tables", "", "Spider tables JSON file.")


def load_json(filepath):
  with gfile.GFile(filepath, "r") as reader:
    text = reader.read()
  return json.loads(text)


def _get_schema_string(table_json):
  """Returns the schema serialized as a string."""
  table_id_to_column_names = collections.defaultdict(list)
  for table_id, name in table_json["column_names_original"]:
    table_id_to_column_names[table_id].append(name.lower())
  tables = table_json["table_names_original"]

  table_strings = []
  for table_id, table_name in enumerate(tables):
    column_names = table_id_to_column_names[table_id]
    table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
    table_strings.append(table_string)

  return "".join(table_strings)


def main(unused_argv):
  tables_json = load_json(FLAGS.tables)
  db_id_to_schema_string = {}
  for table_json in tables_json:
    db_id = table_json["db_id"].lower()
    db_id_to_schema_string[db_id] = _get_schema_string(table_json)

  examples = tsv_utils.read_tsv(FLAGS.input)
  new_examples = []
  for source, target in examples:
    db_id = source.split()[0].rstrip(":")
    schema_string = db_id_to_schema_string[db_id]
    new_source = "%s%s" % (source, schema_string)
    new_examples.append((new_source.lower(), target.lower()))
  tsv_utils.write_tsv(new_examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

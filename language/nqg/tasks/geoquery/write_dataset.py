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
"""Tool for generating geoquery data."""

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils

from language.nqg.tasks.geoquery import entity_utils
from language.nqg.tasks.geoquery import funql_normalization
from language.nqg.tasks.geoquery import geobase_utils
from language.nqg.tasks.geoquery import xml_file_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("corpus", "", "Path to geoquery xml file.")

flags.DEFINE_string("geobase", "", "Path to geobase file.")

flags.DEFINE_string("output", "", "Output dataset file.")


def get_examples():
  """Return list of example tuples."""
  xml_examples = xml_file_utils.read_examples(FLAGS.corpus)

  examples = []
  geobase_entities = geobase_utils.load_entities(FLAGS.geobase)
  for utterance, funql in xml_examples:
    funql = funql_normalization.normalize_funql(funql)
    funql, utterance, _ = entity_utils.replace_entities(funql, utterance,
                                                        geobase_entities)
    funql = funql_normalization.add_space_separation(funql)
    examples.append((utterance, funql))
  return examples


def main(unused_argv):
  examples = get_examples()
  tsv_utils.write_tsv(examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)

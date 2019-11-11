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
"""Creates output vocabulary for NLToSQLExamples."""
import json
import os

from absl import app
from absl import flags

from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample
import tensorflow.gfile as gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '',
                    'The directory containing the input JSON files.')

flags.DEFINE_list('input_filenames', None,
                  'Which files to extract vocabulary from.')

flags.DEFINE_string('output_path', '',
                    'Location to save the output vocabulary.')


def main(unused_argv):
  # Load the examples
  vocabulary = set()
  for filename in FLAGS.input_filenames:
    if filename:
      with gfile.Open(os.path.join(FLAGS.data_dir, filename)) as infile:
        for line in infile:
          if line:
            gold_query = NLToSQLExample().from_json(
                json.loads(line)).gold_sql_query

            for token in gold_query.actions:
              if token.symbol:
                vocabulary.add(token.symbol)
  print('Writing vocabulary of size %d to %s' %
        (len(vocabulary), FLAGS.output_path))
  with gfile.Open(FLAGS.output_path, 'w') as ofile:
    ofile.write('\n'.join(list(vocabulary)))


if __name__ == '__main__':
  app.run(main)

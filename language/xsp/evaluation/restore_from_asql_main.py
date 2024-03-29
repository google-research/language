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
r"""Binary for restoring under-specified FROM clause.

This is currently only supported for Spider.
Predictions should be generated using the --fully_qualify_columns flag.

The input and output format is both of the predictions files generated by
`run_inference.py`. They are newline separated serialiazed json dictionaries
containing the predicted SQL strings and their respective scores for each
example.

Example usage:

${PATH_TO_BINARY} \
  --spider_examples_json=${SPIDER_DIR}/spider/dev.json \
  --spider_tables_json=${SPIDER_DIR}/spider/tables.json \
  --input_path=${INPUT} \
  --output_path=${OUTPUT} \
  --alsologtostderr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from language.xsp.evaluation import restore_from_asql

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', '',
    'Path to under-specified predictions. Use trailing * to expand for sharded inputs.'
)
flags.DEFINE_string('output_path', '', 'Path to write restored predictions.')
flags.DEFINE_string('spider_examples_json', '', 'Path to Spider json examples')
flags.DEFINE_string('spider_tables_json', '', 'Path to Spider json tables.')


def main(unused_argv):
  restore_from_asql.restore_from_clauses(FLAGS.input_path, FLAGS.output_path,
                                         FLAGS.spider_examples_json,
                                         FLAGS.spider_tables_json)


if __name__ == '__main__':
  app.run(main)

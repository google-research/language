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
r"""Builds a table of Wikipedia redirects used to improve entity precision/recall."""

from absl import app
from absl import flags
import apache_beam as beam
from language.fruit import beam_pipelines

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_jsonl',
    default=None,
    help='Input Wikipedia JSONL.',
    required=True,
)

flags.DEFINE_string(
    'output_tsv',
    default=None,
    help='Output tsv path.',
    required=True,
)

flags.DEFINE_list(
    'pipeline_options',
    default=['--runner=DirectRunner'],
    help=(
        'A comma-separated list of command line arguments to be used as options '
        'for the beam pipeline.'))


def main(_):
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  pipeline = beam_pipelines.redirect_table_pipeline(
      input_jsonl=FLAGS.input_jsonl,
      output_tsv=FLAGS.output_tsv,
  )
  with beam.Pipeline(options=pipeline_options) as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)

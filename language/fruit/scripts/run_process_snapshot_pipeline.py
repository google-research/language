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
r"""Processes a pair of Wikipedia snapshots to obtain article-update pairs."""

import apache_beam as beam
from absl import app
from absl import flags
from language.fruit import beam_pipelines

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'source_jsonl',
  default=None,
  help='Source Wikipedia JSONL.',
  required=True,
)

flags.DEFINE_string(
  'target_jsonl',
  default=None,
  help='Target Wikipedia JSONL.',
  required=True,
)

flags.DEFINE_string(
  'output_dir',
  default=None,
  help='Output path.',
  required=True,
)


flags.DEFINE_string(
  'source_redirects',
  default=None,
  help='Source Wikipedia redirect tsv.',
  required=True,
)

flags.DEFINE_string(
  'target_redirects',
  default=None,
  help='Target Wikipedia redirect tsv.',
  required=True,
)

flags.DEFINE_boolean(
  'keep_tables',
  default=True,
  help='Whether to keep or remove tables from articles.',
)

flags.DEFINE_boolean(
  'third_party',
  default=True,
  help=(
    'Enabling relaxes the assumption of what constitutes supporting evidence '
    'for an update more strict. "Third party" mentions are updated sentences '
    'from any article that mention the subject of the article-pair and one '
    'of the added entities.'
  ),
)

flags.DEFINE_boolean(
  'truncate',
  default=False,
  help=(
    'Enabling strips out any text after the intro paragraph from '
    'consideration.'
  ),
)

flags.DEFINE_boolean(
  'use_source_mentions',
  default=False,
  help=(
    'Enabling allows mentions from both the source and target article to be '
    'considered as supporting evidence.'
  )
)

flags.DEFINE_list(
  'pipeline_options',
  default=['--runner=DirectRunner'],
  help=(
    'A comma-separated list of command line arguments to be used as options '
    'for the beam pipeline.'
  )
)


def main(_):
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
    FLAGS.pipeline_options
  )
  pipeline = beam_pipelines.process_snapshot_pipeline(
    source_jsonl=FLAGS.source_jsonl,
    target_jsonl=FLAGS.target_jsonl,
    output_dir=FLAGS.output_dir,
    source_redirects=FLAGS.source_redirects,
    target_redirects=FLAGS.target_redirects,
    keep_tables=FLAGS.keep_tables,
    third_party=FLAGS.third_party,
    truncate=FLAGS.truncate,
    use_source_mentions=FLAGS.use_source_mentions,
  )
  with beam.Pipeline(options=pipeline_options) as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)

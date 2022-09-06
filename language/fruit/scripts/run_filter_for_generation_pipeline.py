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
r"""Applies post-processing to article-evidence pairs.

This pipeline filters out undesirable article-evidence pairs, e.g., ones with
no evidence, as well as applies truncation strategies to handle long texts and
updated articles that excessive amounts of supporting evidence.
"""
from absl import app
from absl import flags
import apache_beam as beam
from language.fruit import beam_pipelines

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_pattern',
    default=None,
    help='Input JSONL files.',
    required=True,
)

flags.DEFINE_string(
    'output_pattern',
    default=None,
    help='Ouput tfrecords files.',
    required=True,
)

flags.DEFINE_string(
    'vocab_model_file',
    default=None,
    help='Vocabulary for mapping text to ids.',
    required=True,
)

flags.DEFINE_integer(
    'max_article_length', default=512, help='Maximum article length.')

flags.DEFINE_string(
    'excessive_length_strategy',
    default='truncate',
    help='What to do with excessively long text. Options: truncate, discard',
)

flags.DEFINE_integer(
    'max_mention_length', default=256, help='Maximum mention length.')

flags.DEFINE_integer(
    'max_mentions',
    default=256,
    help='Maximum number of mentions.',
)

flags.DEFINE_string(
    'excessive_mention_strategy',
    default='truncate',
    help='What to do with excessive mentions. Options: truncate, discard',
)

flags.DEFINE_boolean(
    'use_source_mentions',
    default=True,
    help='Whether to filter source mentions from evidence.',
)

flags.DEFINE_boolean(
    'include_new_articles',
    default=False,
    help='Whether to include new articles (e.g., empty sources).',
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
  pipeline = beam_pipelines.filter_for_generation_pipeline(
      input_pattern=FLAGS.input_pattern,
      output_pattern=FLAGS.output_pattern,
      vocab_model_file=FLAGS.vocab_model_file,
      max_article_length=FLAGS.max_article_length,
      excessive_length_strategy=FLAGS.excessive_length_strategy,
      max_mention_length=FLAGS.max_mention_length,
      max_mentions=FLAGS.max_mentions,
      excessive_mention_strategy=FLAGS.excessive_mention_strategy,
      use_source_mentions=FLAGS.use_source_mentions,
      include_new_articles=FLAGS.include_new_articles,
      dry_run=False,
  )
  with beam.Pipeline(options=pipeline_options) as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)

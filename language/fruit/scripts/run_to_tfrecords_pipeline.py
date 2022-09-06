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
r"""Converts JSONL to tfrecords format for model training/evaluation.

This pipeline also takes care of adding the special sentinel tokens that are
used by the EdiT5 model, and control codes in the controllable setting.
"""
from absl import app
from absl import flags
import apache_beam as beam
from language.fruit import beam_pipelines
from language.fruit import rendering_utils

Task = rendering_utils.Task
DelimiterType = rendering_utils.DelimiterType
EvidenceMarkerType = rendering_utils.EvidenceMarkerType

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

flags.DEFINE_enum_class(
    'task',
    default=Task.diff,
    enum_class=Task,
    help=(
        'Specifies the format of the input and output sequences. Options:\n'
        'diff - the output sequence is the diff between the source and target '
        'article.\n'
        'fullgen - the output sequence is the full target article.\n'
        'controllable - the input sequence includes control tokens indicating '
        'which sentences should be edited, added, or deleted.'),
)

flags.DEFINE_enum_class(
    'delimiter_type',
    default=DelimiterType.text,
    enum_class=DelimiterType,
    help=(
        'Specifies the format of delimiters in the input and output sequences. '
        'Options:\n'
        'text - Delimiters are represented using text, e.g., (0)/[0]. '
        'This is the setting used in the original FRUIT experiments.\n'
        'extra_id - Delimiters are represented using the special extra id tokens '
        'in T5\'s vocabulary.\n'
        'NOTE: We did not find substantial variations in performance across these '
        'settings.'))

flags.DEFINE_boolean(
    'include_source',
    default=True,
    help='Whether the input includes the source article.',
)

flags.DEFINE_boolean(
    'include_evidence',
    default=True,
    help='Whether the input includes the evidence.',
)

flags.DEFINE_boolean(
    'include_distractors',
    default=True,
    help=(
        'Whether the input includes evidence that is not used to produce updates, '
        'e.g., whether the instances require content selection.'))

flags.DEFINE_enum_class(
    'evidence_marker_type',
    default=EvidenceMarkerType.reference,
    enum_class=EvidenceMarkerType,
    help=(
        'Specifies whether or not to include delimiters indicating which pieces '
        'of evidence support a given edit. Options:\n'
        'empty - No delimiters are added.\n'
        'reference - Delimiters are added.'))

flags.DEFINE_integer(
    'max_input_length',
    default=1024,
    help='Maximum sequence length. Longer sequences will be truncated.')

flags.DEFINE_boolean(
    'filter_no_diff',
    default=True,
    help='Whether to filter out articles that were not updated.')

flags.DEFINE_boolean(
    'plot_lengths',
    default=False,
    help='Whether to plot the lengths of the input and output sequences.')

flags.DEFINE_list(
    'pipeline_options',
    default=['--runner=DirectRunner'],
    help=(
        'A comma-separated list of command line arguments to be used as options '
        'for the beam pipeline.'))


def main(_):
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  pipeline = beam_pipelines.to_tfrecords_pipeline(
      input_pattern=FLAGS.input_pattern,
      output_pattern=FLAGS.output_pattern,
      vocab_model_file=FLAGS.vocab_model_file,
      task=FLAGS.task,
      delimiter_type=FLAGS.delimiter_type,
      include_source=FLAGS.include_source,
      include_evidence=FLAGS.include_evidence,
      include_distractors=FLAGS.include_distractors,
      evidence_marker_type=FLAGS.evidence_marker_type,
      max_input_length=FLAGS.max_input_length,
      filter_no_diff=FLAGS.filter_no_diff,
      plot_lengths=FLAGS.plot_lengths,
  )
  with beam.Pipeline(options=pipeline_options) as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)

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
"""Converts the serialized examples to TFRecords for putting into a model."""
# TODO(alanesuhr): Factor out what should be in a lib and what should be in a
# binary.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import random

from absl import app
from absl import flags
import apache_beam as beam
from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample
from language.xsp.model.model_config import load_config
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('examples_dir', '',
                    'The directory containing the examples.')

flags.DEFINE_list('filenames', None,
                  'The list of files to process containing NLToSQLExamples.')

flags.DEFINE_string('config', '', 'The path to a model config file.')

flags.DEFINE_string('tf_examples_dir', '',
                    'The location to put the Tensorflow examples.')

flags.DEFINE_string('output_vocab', '',
                    'The location of the output vocabulary.')

flags.DEFINE_bool('permute', False, 'Whether to permute the train schemas.')

flags.DEFINE_bool('generate_output', False,
                  'Whether to generate output sequences.')

flags.DEFINE_integer(
    'num_spider_repeats', 7,
    'The number of times to permute the Spider data tables (for train only).')

BEG_TOK = '[CLS]'
SEP_TOK = '[SEP]'
TAB_TOK = '[TAB]'
UNK_TOK = '[UNK]'

GENERATE_TYPE = 1
COPY_TYPE = 2

COL_TYPE_TO_TOK = {
    'text': '[STR_COL]',
    'number': '[NUM_COL]',
    'others': '[OTH_COL]',
    'time': '[TIME_COL]',
    'boolean': '[BOOL_COL]',
}


class InputToken(
    collections.namedtuple('InputToken', [
        'wordpiece', 'index', 'copy_mask', 'segment_id',
        'indicates_foreign_key', 'aligned'
    ])):
  pass


class OutputAction(
    collections.namedtuple('OutputAction', ['wordpiece', 'action_id', 'type'])):
  pass


def add_context(key):
  """Adds context features required by the model."""
  features = dict()
  features['language'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=['en']))
  features['region'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=['US_eng']))
  features['type'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
  features['weight'] = tf.train.Feature(
      float_list=tf.train.FloatList(value=[1.0]))
  features['tag'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=['all']))
  features['key'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[key.encode('utf8')]))
  return features


class ConvertToSequenceExampleDoFn(beam.DoFn):
  """DoFn for converting from NLToSQLExample to a TFRecord."""

  def __init__(self, model_config, generate_output, permute, num_repeats,
               *unused_args, **unused_kwargs):
    self.model_config = model_config

    self.input_vocabulary = None
    self.output_vocabulary = None
    self.permute = permute
    self.num_repeats = num_repeats

    if not self.permute and self.num_repeats > 1:
      raise ValueError('Not permuting but num_repeats = ' +
                       str(self.num_repeats))

    # This cache maps from a proto representing a schema to its string
    # equivalent
    # (NOTE: this assumes there's no randomness in the order of the tables,
    # cols, etc.)
    self.table_cache = dict()

    self.generate_output = generate_output

  def non_parallel_process(self, example):
    # Load cache
    if not self.input_vocabulary:
      with tf.gfile.Open(
          self.model_config.data_options.bert_vocab_path) as infile:
        self.input_vocabulary = [
            line.rstrip('\n') for line in infile.readlines()
        ]

    if not self.output_vocabulary:
      with tf.gfile.Open(FLAGS.output_vocab) as infile:
        self.output_vocabulary = [
            line.replace('\n', '', 1) for line in infile.readlines()
        ]

    results = list()
    for _ in range(self.num_repeats):
      # Convert the input to an indexed sequence
      input_conversion = self._convert_input_to_indexed_sequence(
          example.model_input, random_permutation=self.permute)
      if input_conversion is None:
        return None

      # input_tokens stores the raw wordpieces, its index in the vocabulary, and
      # whether it is copiable

      # The maps store tuples of table or column entities paired with their head
      # index in input_tokens
      input_tokens, table_index_map, column_index_map, base_idx = input_conversion

      # Convert the output to an indexed sequence
      output_actions = list()
      if self.generate_output:
        output_actions = self._convert_output_to_indexed_sequence(
            example, table_index_map, column_index_map, base_idx)

        if output_actions is None:
          return None

        raw_input_wordpieces = [
            input_token.wordpiece for input_token in input_tokens
        ]
        for action in output_actions:
          if action.type == COPY_TYPE:

            # Copy actions should only either
            #  1. Copy from the input (i.e., before SEP)
            #  2. Copy TAB or COL tokens
            assert input_tokens[
                action.action_id].index == self.input_vocabulary.index(
                    TAB_TOK) or input_tokens[action.action_id].index in [
                        self.input_vocabulary.index(col_tok)
                        for col_tok in COL_TYPE_TO_TOK.values()
                    ] or action.action_id < raw_input_wordpieces.index(
                        SEP_TOK
                    ), 'Unexpected copying action: %r with proto:\n%r' % (
                        input_tokens[action.action_id], example)

            assert input_tokens[action.action_id].copy_mask == 1, (
                'Copied, but copy mask is 0: %s at '
                'index %d; copied action was %s') % (
                    input_tokens[action.action_id], action.action_id, action)

      # Actually create the TF Example
      results.append(
          self._convert_to_sequence_example(
              input_tokens, output_actions,
              example.model_input.original_utterance).SerializeToString())
    return results

  def process(self, example):
    results = self.non_parallel_process(example)
    if results is not None:
      for result in results:
        yield result

  def _convert_input_to_sequence_example(self, input_tokens, features):
    features['source_wordpieces'] = tf.train.FeatureList(feature=[
        tf.train.Feature(
            int64_list=tf.train.Int64List(value=[input_token.index]))
        for input_token in input_tokens
    ])

    features['copiable_input'] = tf.train.FeatureList(feature=[
        tf.train.Feature(
            int64_list=tf.train.Int64List(value=[input_token.copy_mask]))
        for input_token in input_tokens
    ])

    copy_features = list()
    foreign_key_features = list()
    for input_token in input_tokens:
      copy_features.append(
          tf.train.Feature(
              bytes_list=tf.train.BytesList(
                  value=[input_token.wordpiece.encode('utf8')])))
      foreign_key_features.append(
          tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=[input_token.indicates_foreign_key])))
    features['copy_strings'] = tf.train.FeatureList(feature=copy_features)

    features['segment_ids'] = tf.train.FeatureList(feature=[
        tf.train.Feature(
            int64_list=tf.train.Int64List(value=[input_token.segment_id]))
        for input_token in input_tokens
    ])

    features['indicates_foreign_key'] = tf.train.FeatureList(
        feature=foreign_key_features)

    features['utterance_schema_alignment'] = tf.train.FeatureList(feature=[
        tf.train.Feature(
            int64_list=tf.train.Int64List(value=[input_token.aligned]))
        for input_token in input_tokens
    ])

  def _convert_output_to_sequence_example(self, output_actions, features):
    features['target_action_ids'] = tf.train.FeatureList(feature=[
        tf.train.Feature(
            int64_list=tf.train.Int64List(value=[action.action_id]))
        for action in output_actions
    ])

    features['target_action_types'] = tf.train.FeatureList(feature=[
        tf.train.Feature(int64_list=tf.train.Int64List(value=[action.type]))
        for action in output_actions
    ])

  def _convert_to_sequence_example(self, input_tokens, output_actions,
                                   utterance):
    features = collections.OrderedDict()
    self._convert_input_to_sequence_example(input_tokens, features)

    self._convert_output_to_sequence_example(output_actions, features)

    context_features = add_context(utterance)
    return tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list=features))

  def _get_vocab_index_or_unk(self, token, is_input=True):
    # Note that this will return a 'Unicode equals warning' if the token is a
    # unicode-only token
    if is_input:
      if token in self.input_vocabulary:
        return self.input_vocabulary.index(token)
      return self.input_vocabulary.index(UNK_TOK)
    if token in self.output_vocabulary:
      # Add 3 to this because there are 3 placeholder tokens in the output
      # vocabulary that will be used during train (PAD, BEG, and END).
      return self.output_vocabulary.index(token) + 3
    print('Could not find token ' + token + ' in output vocabulary.')

  def _convert_input_to_indexed_sequence(self, model_input, random_permutation):
    # Everything is tokenized, but need to combine the utterance with the
    # schema.
    converted_wordpiece_tokens = list()
    for wordpiece in model_input.utterance_wordpieces:
      converted_wordpiece_tokens.append(
          InputToken(('##' if '##' in wordpiece.wordpiece else '') +
                     model_input.original_utterance[
                         wordpiece.span_start_index:wordpiece.span_end_index],
                     self._get_vocab_index_or_unk(wordpiece.wordpiece), 1, 0, 0,
                     int(wordpiece.matches_to_schema)))

    tokens = [
        InputToken(BEG_TOK, self.input_vocabulary.index(BEG_TOK), 0, 0, 0, 0)
    ] + converted_wordpiece_tokens + [
        InputToken(SEP_TOK, self.input_vocabulary.index(SEP_TOK), 0, 0, 0, 0)
    ]

    table_index_map = list()
    column_index_map = list()

    # Add the table tokens
    # Look it up in the cache
    string_serial = ','.join([str(table) for table in model_input.tables])
    if string_serial in self.table_cache and not random_permutation:
      tokens_suffix, table_index_map, column_index_map = self.table_cache[
          string_serial]
    else:
      # The input tokens contain the string to copy, rather than the wordpiece
      # that's being embedded.
      tokens_suffix = list()

      order = list(range(len(model_input.tables)))
      if random_permutation:
        random.shuffle(order)

      for table_segment_idx, table_idx in enumerate(order):
        table = model_input.tables[table_idx]
        table_index_map.append((len(tokens_suffix), table))
        table_wordpieces_tokens = list()
        for wordpiece in table.table_name_wordpieces:
          table_wordpieces_tokens.append(
              InputToken('', self._get_vocab_index_or_unk(wordpiece.wordpiece),
                         0, table_segment_idx + 1, 0,
                         int(table.matches_to_utterance)))

        tokens_suffix.extend([
            InputToken(
                table.original_table_name, self.input_vocabulary.index(TAB_TOK),
                1, table_segment_idx + 1, 0, int(table.matches_to_utterance))
        ] + table_wordpieces_tokens)

        col_order = list(range(len(table.table_columns)))
        if random_permutation:
          random.shuffle(col_order)

        # Add the column tokens for this table
        for col_idx in col_order:
          column = table.table_columns[col_idx]
          column_index_map.append((len(tokens_suffix), column))
          column_wordpiece_tokens = list()
          for wordpiece in column.column_name_wordpieces:
            column_wordpiece_tokens.append(
                InputToken('',
                           self._get_vocab_index_or_unk(wordpiece.wordpiece), 0,
                           table_segment_idx + 1, int(column.is_foreign_key),
                           int(column.matches_to_utterance)))

          tokens_suffix.extend([
              InputToken(
                  column.original_column_name,
                  self.input_vocabulary.index(COL_TYPE_TO_TOK[
                      column.column_type]), 1, table_segment_idx + 1,
                  int(column.is_foreign_key), int(column.matches_to_utterance))
          ] + column_wordpiece_tokens)
      # Update cache
      if not random_permutation:
        self.table_cache[string_serial] = (tokens_suffix, table_index_map,
                                           column_index_map)

    base_idx = len(tokens)
    tokens.extend(tokens_suffix)

    # If there are too many tokens, return None.
    if len(tokens) > self.model_config.data_options.max_num_tokens:
      return None

    return tokens, table_index_map, column_index_map, base_idx

  def _convert_output_to_indexed_sequence(self, example, table_index_map,
                                          column_index_map, base_idx):
    action_sequence = list()

    gold_query = example.gold_sql_query

    if len(
        gold_query.actions) > self.model_config.data_options.max_decode_length:
      return None

    for action in gold_query.actions:
      if action.symbol:
        action_sequence.append(
            OutputAction(action.symbol,
                         self._get_vocab_index_or_unk(action.symbol, False),
                         GENERATE_TYPE))
      elif action.entity_copy:
        found = False
        if action.entity_copy.copied_table:
          # Copied a table.
          table = action.entity_copy.copied_table
          for index, entity in table_index_map:
            if entity.original_table_name == table.original_table_name:
              action_sequence.append(
                  OutputAction(table.original_table_name, index + base_idx,
                               COPY_TYPE))
              found = True
              break
        else:
          # Copied a column.
          column = action.entity_copy.copied_column
          for index, entity in column_index_map:
            if entity.original_column_name == column.original_column_name and entity.table_name == column.table_name:
              action_sequence.append(
                  OutputAction(column.original_column_name, index + base_idx,
                               COPY_TYPE))
              found = True
              break
        if not found:
          return None
      elif action.utterance_copy:
        copy_wordpiece = action.utterance_copy
        action_sequence.append(
            OutputAction(copy_wordpiece.wordpiece,
                         copy_wordpiece.tokenized_index + 1, COPY_TYPE))

    if None in [action.action_id for action in action_sequence]:
      return None

    return action_sequence


def creation_wrapper(process_dataset_fn):
  """Wrapper for creating the TFRecords files."""
  # Create the tf examples directory.
  if not tf.gfile.IsDirectory(FLAGS.tf_examples_dir):
    print('Creating TFExamples directory at ' + FLAGS.tf_examples_dir)
    tf.gfile.MkDir(FLAGS.tf_examples_dir)

  # Get the model config.
  model_config = load_config(FLAGS.config)

  for filename in FLAGS.filenames:
    if not filename:
      continue

    input_path = os.path.join(FLAGS.examples_dir, filename)
    output_path = os.path.join(
        FLAGS.tf_examples_dir,
        filename.split('/')[-1].split('.')[0] + '.tfrecords')

    permute = 'spider_train' in output_path and FLAGS.permute
    num_repeats = FLAGS.num_spider_repeats if permute else 1

    print('Processing %s. Permute: %r with %d repetitions' %
          (filename, permute, num_repeats))
    print('Writing to ' + output_path)

    process_dataset_fn(input_path, model_config, permute, num_repeats,
                       output_path)


def process_dataset(input_path, model_config, permute, num_repeats,
                    output_path):
  """Function that processes a dataset without multiprocessing."""
  fn = ConvertToSequenceExampleDoFn(
      model_config,
      FLAGS.generate_output,
      permute=permute,
      num_repeats=num_repeats)

  with tf.gfile.Open(input_path) as infile:
    examples = [NLToSQLExample().from_json(json.loads(line)) for line in infile]

  with tf.python_io.TFRecordWriter(output_path) as writer:
    num_examples_written = 0
    total_examples = 0
    for example in examples:
      total_examples += 1
      converteds = fn.non_parallel_process(example)
      if converteds:
        num_examples_written += 1
        for converted in converteds:
          writer.write(converted)
  print('Wrote to %d / %d to %s' %
        (num_examples_written, total_examples, output_path))


def main(unused_argv):
  creation_wrapper(process_dataset)


if __name__ == '__main__':
  app.run(main)

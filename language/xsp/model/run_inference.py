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
"""Runs inference on an XSP model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import apache_beam as beam

from language.xsp.data_preprocessing.michigan_preprocessing import get_nl_sql_pairs
from language.xsp.data_preprocessing.michigan_preprocessing import read_schema
from language.xsp.data_preprocessing.spider_preprocessing import load_spider_examples
from language.xsp.data_preprocessing.spider_preprocessing import load_spider_tables
from language.xsp.data_preprocessing.spider_preprocessing import preprocess_sql
from language.xsp.evaluation import restore_from_asql
from language.xsp.model import input_pipeline
from language.xsp.model import model_builder
from language.xsp.model.model_config import load_config

import sqlparse
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('config_filepath', '', 'The location of the model config.')

flags.DEFINE_string('output_vocab_filepath', '',
                    'The location of the output vocabulary.')

flags.DEFINE_string('clean_output_vocab_filepath', None,
                    'Path to the clean output vocabfile.')

flags.DEFINE_string('checkpoint_filepath', '',
                    'The location of the checkpoint.')

flags.DEFINE_string('input', '',
                    'TFRecords file containing TFExamples to evaluate.')

flags.DEFINE_string('data_filepath', '',
                    'Directory containing the original data.')
flags.DEFINE_string('database_filepath', '',
                    'Local directory containing the databases.')
flags.DEFINE_string('empty_database_filepath', '',
                    'Local directory containing the empty databases.')
flags.DEFINE_string('predictions_path', '', 'Path for jsonl of predictions.')
flags.DEFINE_string('output', '', 'Path for the output text file.')
flags.DEFINE_list('splits', None, 'The splits to run with.')
flags.DEFINE_string('dataset_name', '',
                    'The name of the dataset being processed.')

flags.DEFINE_integer('beam_size', 1, 'The size of the beam to predict.')

flags.DEFINE_bool('match_and_save', True,
                  'Whether to join and save predictions for evaluation.')

flags.DEFINE_bool(
    'restore_preds_from_asql', False,
    'Whether model was trained with under-specified from clauses.')
flags.DEFINE_bool(
    'use_oracle_foriegn_keys', True,
    'Whether to use oracle foreign keys when restoring from asql.')
# These flags are only required if restore_preds_from_asql is True.
# TODO(petershaw): Better method for handling other datasets.
flags.DEFINE_string('spider_examples_json', '', 'Path to Spider json examples')
flags.DEFINE_string('spider_tables_json', '', 'Path to Spider json tables.')
flags.DEFINE_string(
    'restored_predictions_path', '',
    'Path for jsonl of predictions that have been restored from Abstract SQL format.'
)

QUOTES = {'\'', '"'}


def _action_id_to_table_name_map(segment_ids, copy_strings):
  """Returns a map of action_ids to table names for columns."""
  current_segment_id = 0
  table_name = None
  # The first copy_string for a new segment_id is the table name.
  # Following copy_string are for columns belonging to that table.
  # TODO(petershaw): This is really hacky! We should provide a better channel
  # for passing this information to the output during inference.
  action_id_to_table_name_map = {}
  for action_id, (segment_id,
                  copy_string) in enumerate(zip(segment_ids, copy_strings)):
    if segment_id > current_segment_id:
      current_segment_id = segment_id
      table_name = copy_string
    elif table_name:
      action_id_to_table_name_map[action_id] = table_name
  return action_id_to_table_name_map


def clean_predicted_sequence(action_ids,
                             action_types,
                             scores,
                             vocab,
                             copy_strings,
                             segment_ids,
                             restore_preds_from_asql,
                             clean_vocab=None):
  """Cleans a set of predicted SQL queries."""
  action_id_to_table_name_map = None
  if restore_preds_from_asql:
    action_id_to_table_name_map = _action_id_to_table_name_map(
        segment_ids, copy_strings)
  string_seq = list()
  for action_type, action_id in zip(action_types, action_ids):
    if action_type == 1:
      # Generate symbol from output vocabulary.
      pred_idx = action_id
      if pred_idx == 1:
        # END symbol.
        break
      # Indices into vocab are offset by 3.
      symbol = vocab[pred_idx - 3]
      if clean_vocab:
        if symbol in clean_vocab:
          string_seq.append(symbol)
      else:
        string_seq.append(symbol)
    else:
      # Copy symbol from input.
      symbol = copy_strings[action_id]
      if restore_preds_from_asql and action_id in action_id_to_table_name_map:
        # For abstract SQL, need to fully qualify column names by prepending
        # the table name.
        table_name = action_id_to_table_name_map[action_id]
        symbol = '%s.%s' % (table_name, symbol)
      string_seq.append(symbol)

  sql = ''
  in_quote = False
  for i, token in enumerate(string_seq):
    if not in_quote and token not in QUOTES:
      sql += ' '
    if token in QUOTES:
      in_quote = not in_quote
    sql += token
    if not in_quote or token not in QUOTES and i < len(
        string_seq) - 1 and string_seq[i + 1] not in QUOTES:
      sql += ' '

  sql = sql.replace('  ', ' ')
  sql = sql.replace(' ##', '')
  sql = sqlparse.format(sql, reident=True, keyword_case='upper')
  sql = sql.replace('\n', ' ')
  sql = sql.replace('( ', '(')
  sql = sql.replace(' )', ')')
  sql = sql.replace(' . ', '.')
  sql = sql.replace(' %', '%')
  sql = sql.replace('% ', '%')

  for func in ('count', 'min', 'max', 'avg', 'sum'):
    sql = sql.replace('%s (' % func.upper(), '%s(' % func)

  for i in range(1, 11):
    sql = sql.replace('t%d' % i, 'T%d' % i)

  sql = sql.strip()

  return sql, float(scores)


def setup_graph():
  """Sets up the Tenorflow graph for inference."""
  # Set up the model for inference
  model_config = load_config(os.path.join(FLAGS.config_filepath))
  placeholder, features, labels = input_pipeline.create_placeholder_inputs(
      model_config.model_parameters.use_segment_ids,
      model_config.model_parameters.use_foreign_key_features,
      model_config.model_parameters.use_alignment_features)

  model_fn = model_builder.build_model_fn(
      model_config,
      FLAGS.output_vocab_filepath,
      FLAGS.clean_output_vocab_filepath,
      beam_size=FLAGS.beam_size)
  mode = tf.estimator.ModeKeys.PREDICT
  predictions = model_fn(features, labels, mode).predictions
  saver = tf.train.Saver()

  return saver, placeholder, predictions


def _get_copy_strings(tf_example):
  copy_strings = []
  for token in tf_example.feature_lists.feature_list['copy_strings'].feature:
    if len(token.bytes_list.value) != 1:
      raise ValueError('Invalid copy_strings in example: %s' % tf_example)
    copy_strings.append(token.bytes_list.value[0])
  if not copy_strings:
    raise ValueError('Missing copy_strings in example: %s' % tf_example)
  return copy_strings


def _get_segment_ids(tf_example):
  segment_ids = []
  for token in tf_example.feature_lists.feature_list['segment_ids'].feature:
    if len(token.int64_list.value) != 1:
      raise ValueError('Invalid segment_ids in example: %s' % tf_example)
    segment_ids.append(token.int64_list.value[0])
  if not segment_ids:
    raise ValueError('Missing segment_ids in example: %s' % tf_example)
  return segment_ids


def get_prediction(placeholder,
                   tf_example,
                   sess,
                   outputs,
                   vocab,
                   beam_size,
                   restore_preds_from_asql=False,
                   clean_vocab=None):
  """Gets predicted outputs for a specific input to the model."""
  copy_strings = _get_copy_strings(tf_example)
  segment_ids = _get_segment_ids(tf_example)

  feed_dict = {placeholder: tf_example.SerializeToString()}
  output_vals = sess.run(outputs, feed_dict=feed_dict)

  predictions = list()
  scores = list()
  for index in range(beam_size):
    prediction, score = clean_predicted_sequence(
        output_vals['predicted_action_ids'][index],
        output_vals['predicted_action_types'][index],
        output_vals['scores'][index],
        vocab,
        copy_strings,
        segment_ids,
        restore_preds_from_asql=restore_preds_from_asql,
        clean_vocab=clean_vocab)
    predictions.append(prediction)
    scores.append(score)
  return predictions, scores


class RunInferenceDoFn(beam.DoFn):
  """DoFn for running inference on an example given model parameters."""

  class _GraphState(object):
    """This class caches the tf session/graph across process instances."""

    def __init__(self, checkpoint):
      # Set up the graph and load the checkpoint
      saver, placeholder, outputs = setup_graph()

      self.placeholder = placeholder
      self.sess = tf.Session()
      self.saver = saver
      self.outputs = outputs

      self.saver.restore(self.sess, checkpoint)
      print('Restoring checkpoint: {}'.format(checkpoint))

  def __init__(self, checkpoint, *unused_args, **unused_kwargs):
    with tf.gfile.Open(FLAGS.output_vocab_filepath) as infile:
      self._vocab = [line.strip() for line in infile]

    if FLAGS.clean_output_vocab_filepath:
      with tf.gfile.Open(FLAGS.clean_output_vocab_filepath) as infile:
        self._clean_vocab = [line.strip() for line in infile]
    else:
      self._clean_vocab = None

    self._checkpoint = checkpoint
    self._graph_state = None

  def non_parallel_process(self, example):
    # Runs inference for the example.
    if self._graph_state is None:
      self._graph_state = self._GraphState(self._checkpoint)

    predicted_sequences, scores = get_prediction(
        self._graph_state.placeholder, example, self._graph_state.sess,
        self._graph_state.outputs, self._vocab, FLAGS.beam_size,
        FLAGS.restore_preds_from_asql, self._clean_vocab)

    return {
        'utterance': dict(example.context.feature)['key'].bytes_list.value[0],
        'predictions': predicted_sequences,
        'scores': scores
    }

  def process(self, example):
    if isinstance(example, str):
      raise ValueError('Example is a str! %r' % example)
    yield self.non_parallel_process(example)


def inference_wrapper(inference_fn, sharded=False):
  """Wrapper for running inference."""
  dataset_name = FLAGS.dataset_name

  predictions = FLAGS.predictions_path + '*'
  # Don't run inference if predictions have already been generated.
  if not tf.gfile.Glob(FLAGS.predictions_path + '*'):
    inference_fn(FLAGS.input, FLAGS.predictions_path, FLAGS.checkpoint_filepath,
                 dataset_name)

  # If using Abstract SQL, need to restore under-specified FROM clauses
  # output above.
  if FLAGS.restore_preds_from_asql:
    spider = dataset_name.lower() == 'spider'

    if not tf.io.gfile.exists(FLAGS.restored_predictions_path):
      restore_from_asql.restore_from_clauses(
          predictions,
          FLAGS.restored_predictions_path,
          spider_examples_json=FLAGS.spider_examples_json if spider else '',
          spider_tables_json=FLAGS.spider_tables_json if spider else '',
          michigan_schema=None if spider else read_schema(
              os.path.join(FLAGS.data_filepath, FLAGS.dataset_name +
                           '_schema.csv')),
          dataset_name=FLAGS.dataset_name,
          use_oracle_foriegn_keys=FLAGS.use_oracle_foriegn_keys)
    predictions = FLAGS.restored_predictions_path

  if FLAGS.match_and_save:
    # Load the database tables.
    schema_obj = None
    if dataset_name.lower() == 'spider':
      schema_obj = load_spider_tables(
          os.path.join(FLAGS.data_filepath, 'tables.json'))
    elif dataset_name.lower() == 'wikisql':
      raise ValueError('WikiSQL inference is not supported yet')
    else:
      schema_csv = os.path.join(FLAGS.data_filepath,
                                FLAGS.dataset_name + '_schema.csv')
      schema_obj = read_schema(schema_csv)

    # Now match with the original data and save
    match_and_save(predictions, FLAGS.output, dataset_name.lower(),
                   FLAGS.splits, FLAGS.data_filepath, schema_obj, sharded)


def match_and_save(predictions_path, output_path, dataset_name, splits,
                   data_filepath, schema_obj, sharded):
  """Loads an original dataset and matches with a predictions file."""
  # Load the predictions file
  prediction_dict = dict()
  if sharded:
    for data_file in tf.gfile.Glob(predictions_path + '*'):
      with tf.gfile.Open(data_file) as infile:
        for line in infile:
          if line:
            obj = json.loads(line)
            prediction_dict[obj['utterance']] = obj

  else:
    with tf.gfile.Open(predictions_path) as infile:
      for line in infile:
        if line:
          obj = json.loads(line)
          prediction_dict[obj['utterance']] = obj

  # Load the data for this particular dataset (for look up)
  # `examples` is a list of dictionaries for each example, containing a TFRecord
  #  object, nl, sql, and a db_id (if running inference on Spider).
  matched_examples = list()
  if dataset_name.lower() == 'spider':
    assert len(splits) == 1
    split = splits[0]

    for example in load_spider_examples(
        os.path.join(data_filepath, split + '.json')):
      # Looks up the example's schema.
      schema = schema_obj[example['db_id']]

      # Returns a dictionary containing relevant prediction information.
      database_filepath = os.path.join('spider_databases',
                                       example['db_id'] + '.sqlite')

      key = ' '.join(example['question_toks'])
      prediction = prediction_dict[key]
      matched_examples.append({
          'utterance':
              key,
          'predictions':
              prediction['predictions'],
          'scores':
              prediction['scores'],
          'gold':
              example['query'],
          'database_path':
              os.path.join(FLAGS.database_filepath, database_filepath),
          'empty_database_path':
              os.path.join(FLAGS.empty_database_filepath, database_filepath),
          'schema':
              schema
      })

  elif dataset_name.lower() == 'wikisql':
    raise ValueError('Inference on WikiSQL not supported.')
  else:
    for nl, sql in get_nl_sql_pairs(
        os.path.join(data_filepath, dataset_name + '.json'), set(splits)):
      key = nl.encode('utf8')

      # Returns a dictionary containing relevant prediction information.
      database_filepath = dataset_name + '.db'

      prediction = prediction_dict[key]
      matched_examples.append({
          'utterance':
              key,
          'predictions':
              prediction['predictions'],
          'scores':
              prediction['scores'],
          'gold':
              preprocess_sql(sql),
          'database_path':
              os.path.join(FLAGS.database_filepath, database_filepath),
          'empty_database_path':
              os.path.join(FLAGS.empty_database_filepath, database_filepath),
          'schema':
              schema_obj
      })

  with tf.gfile.Open(output_path, 'w') as ofile:
    ofile.write(json.dumps(matched_examples))


# dataset_name is unused because it's unnecessary in the local inference
# function
# pylint: disable=unused-argument
def inference(input_path, predictions_path, checkpoint, dataset_name):
  """Runs inference locally.

  Args:
    input_path: The TFRecord file with the input examples.
    predictions_path: The filepath to save example predictions (as a .jsonl).
    checkpoint: Filepath to the model save checkpoint.
    dataset_name: Name of the dataset.
  """
  fn = RunInferenceDoFn(checkpoint)

  # Load and process the TFRecords. First, inference is ran on these records
  # without looking at the gold query.
  examples = list()
  record_iterator = tf.python_io.tf_record_iterator(path=input_path)
  for record in record_iterator:
    example = tf.train.SequenceExample()
    example.ParseFromString(record)
    examples.append(example)

  # The predictions are written to the path that only contains predictions
  with tf.gfile.Open(predictions_path, 'w') as ofile:
    for example in examples:
      ofile.write(json.dumps(fn.non_parallel_process(example)) + '\n')


def main(unused_argv):
  inference_wrapper(inference)


if __name__ == '__main__':
  app.run(main)

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
"""Official evaluation script for natural language to SQL datasets.

Arguments:
  predictions_filepath (str): Path to a predictions file (in JSON format).
  output_filepath (str): Path to the file where the result of execution is
    saved.
  cache_filepath (str): Path to a JSON file containing a mapping from gold SQL
    queries to cached resulting tables.  Should be ran locally. All filepaths
    above should refer to the local filesystem.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import numpy as np
import sqlite3
import timeout_decorator
from tqdm import tqdm

# Maximum allowable timeout for executing predicted and gold queries.
TIMEOUT = 60

# Maximum number of candidates we should consider
MAX_CANDIDATE = 20


# These are substrings of exceptions from sqlite3 that indicate certain classes
# of schema and syntax errors.
SCHEMA_INCOHERENCE_STRINGS = {
    'no such table', 'no such column', 'ambiguous column name'
}
SYNTAX_INCORRECTNESS_STRINGS = {
    'bad syntax', 'unrecognized token', 'incomplete input',
    'misuse of aggregate', 'left and right', 'wrong number of arguments',
    'sub-select returns', '1st order by term does not match any column',
    'no such function', 'clause is required before',
    'incorrect number of bindings', 'datatype mismatch', 'syntax error'
}


def normalize_sql_str(string):
  """Normalizes the format of a SQL string for string comparison."""
  string = string.lower()
  while '  ' in string:
    string = string.replace('  ', ' ')
  string = string.strip()
  string = string.replace('( ', '(').replace(' )', ')')
  string = string.replace(' ;', ';')
  string = string.replace('"', '\'')

  if ';' not in string:
    string += ';'
  return string


def string_acc(s1, s2):
  """Computes string accuracy between two SQL queries."""
  return normalize_sql_str(s1) == normalize_sql_str(s2)


def result_table_to_string(table):
  """Converts a resulting SQL table to a human-readable string."""
  string_val = '\t' + '\n\t'.join(
      [str(row) for row in table[:min(len(table), 5)]]) + '\n'
  if len(table) > 5:
    string_val += '... and %d more rows.\n' % (len(table) - 5)
  return string_val


def try_executing_query(prediction, cursor, case_sensitive=True, verbose=False):
  """Attempts to execute a SQL query against a database given a cursor."""
  exception_str = None

  prediction_str = prediction[:]
  prediction_str = prediction_str.replace(';', '').strip()
  print('Current prediction:' + prediction_str)

  st = time.time()
  try:
    if not case_sensitive:
      new_prediction = ''
      last_quote = ''
      for char in prediction:
        new_prediction += char
        if char in {'"', '\''} and not last_quote:
          last_quote = char
        elif char == last_quote:
          last_quote = ''
          new_prediction += ' COLLATE NOCASE'
      prediction = new_prediction

      if verbose:
        print('Executing case-insensitive query:')
        print(new_prediction)
    pred_results = timeout_execute(cursor, prediction)
  except timeout_decorator.timeout_decorator.TimeoutError:
    print('!time out!')
    pred_results = []
    exception_str = 'timeout'
  except (sqlite3.Warning, sqlite3.Error, sqlite3.DatabaseError,
          sqlite3.IntegrityError, sqlite3.ProgrammingError,
          sqlite3.OperationalError, sqlite3.NotSupportedError) as e:
    exception_str = str(e).lower()
    pred_results = []
  execution_time = time.time() - st

  return pred_results, exception_str, execution_time


@timeout_decorator.timeout(seconds=TIMEOUT, use_signals=False)
def timeout_execute(cursor, prediction):
  cursor.execute(prediction)
  pred_results = cursor.fetchall()
  pred_results = [list(result) for result in pred_results]
  return pred_results


def find_used_entities_in_string(query, columns, tables):
  """Heuristically finds schema entities included in a SQL query."""
  used_columns = set()
  used_tables = set()

  nopunct_query = query.replace('.', ' ').replace('(', ' ').replace(')', ' ')

  for token in nopunct_query.split(' '):
    if token.lower() in columns:
      used_columns.add(token.lower())
    if token.lower() in tables:
      used_tables.add(token.lower())
  return used_columns, used_tables


def compute_f1(precision, recall):
  if precision + recall > 0.:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.


def compute_set_f1(pred_set, gold_set):
  """Computes F1 of items given two sets of items."""
  prec = 1.
  if pred_set:
    prec = float(len(pred_set & gold_set)) / len(pred_set)

  rec = 1.
  if gold_set:
    rec = float(len(pred_set & gold_set)) / len(gold_set)
  return compute_f1(prec, rec)


def col_tab_f1(schema, gold_query, predicted_query):
  """Computes the F1 of tables and columns mentioned in the two queries."""

  # Get the schema entities.
  db_columns = set()
  db_tables = set()
  for name, cols in schema.items():
    for col in cols:
      db_columns.add(col['field name'].lower())
    db_tables.add(name.lower())

  # Heuristically find the entities used in the gold and predicted queries.
  pred_columns, pred_tables = find_used_entities_in_string(
      predicted_query, db_columns, db_tables)
  gold_columns, gold_tables = find_used_entities_in_string(
      gold_query, db_columns, db_tables)

  # Compute and return column and table F1.
  return (compute_set_f1(pred_columns,
                         gold_columns), compute_set_f1(pred_tables,
                                                       gold_tables))


def execute_prediction(prediction, empty_table_cursor, cursor, case_sensitive,
                       verbose):
  """Executes a single example's prediction(s).

  If more than one prediction is available, the most likely executable
  prediction is used as the "official" prediction.

  Args:
    prediction: A dictionary containing information for a single example's
      prediction.
    empty_table_cursor: The cursor to a database containing no records, to be
      used only to determine whether a query is executable in the database.
    cursor: The sqlite3 database cursor to execute queries on.
    case_sensitive: Boolean indicating whether the execution should be case
      sensitive with respect to string values.
    verbose: Whether to print details about what queries are being executed.

  Returns:
    Tuple containing the highest-ranked executable query, the resulting table,
    and any exception string associated with executing this query.
  """

  # Go through predictions in order of probability and test their executability
  # until you get an executable prediction. If you don't find one, just
  # "predict" the most probable one.
  paired_preds_and_scores = zip(prediction['predictions'], prediction['scores'])
  sorted_by_scores = sorted(
      paired_preds_and_scores, key=lambda x: x[1], reverse=True)

  best_prediction = None
  pred_results = None
  exception_str = None
  execution_time = 0

  if len(sorted_by_scores) > MAX_CANDIDATE:
    sorted_by_scores = sorted_by_scores[:MAX_CANDIDATE]

  for i, (pred, _) in enumerate(sorted_by_scores):
    # Try predicting
    if verbose:
      print('Trying to execute query:\n\t' + pred)
      print('... on empty database')
    temp_exception_str = try_executing_query(pred, empty_table_cursor,
                                             case_sensitive, verbose)[1]

    if temp_exception_str:
      if i == 0:
        # By default, set the prediction to the first (highest-scoring)
        # one.
        best_prediction = pred

        # Get the actual results
        if verbose:
          print('... on actual database')
        pred_results, exception_str, execution_time = try_executing_query(
            pred, cursor, case_sensitive, verbose)
      if exception_str == 'timeout':
        # Technically, this query didn't have a syntax problem, so
        # continue and set this as the best prediction.
        best_prediction = pred

        if verbose:
          print('... on actual database')
        pred_results, exception_str, execution_time = try_executing_query(
            pred, cursor, case_sensitive, verbose)
        break
    else:
      best_prediction = pred
      exception_str = None

      if verbose:
        print('No exception... on actual database')
      pred_results, _, execution_time = try_executing_query(pred, cursor, case_sensitive,
                                         verbose)
      break

  return best_prediction, pred_results, exception_str, execution_time


def _convert_to_unicode_string(value):
  if isinstance(value, int) or isinstance(value, float):
    return str(value).decode('utf-8', 'ignore')
  elif isinstance(value, unicode):
    return value
  elif isinstance(value, str):
    return value.decode('utf-8', 'ignore')
  else:
    return str(value).decode('utf-8', 'ignore')


def execute_predictions(predictions, cache_dict, ofile, case_sensitive,
                        verbose, update_cache):
  """Executes predicted/gold queries and computes performance.

  Writes results to ofile.

  Args:
    predictions: A list of dictionaries defining the predictions made by a
      model.
    cache_dict: A dictionary mapping from gold queries to the resulting tables.
    ofile: A file pointer to be written to.
    case_sensitive: A Boolean indicating whether execution of queries should be
      case sensitive with respect to strings.
    verbose: Whether to print detailed information about evaluation (e.g., for
      debugging).
  """
  # Keeps tracks of metrics throughout all of the evaluation.
  exec_results_same = list()
  string_same = list()

  precision = list()
  recall = list()

  column_f1s = list()
  table_f1s = list()

  conversion_errors = 0

  schema_errors = 0
  syntax_errors = 0
  timeouts = 0

  gold_error = 0

  i = 0

  predictions_iterator = tqdm
  if verbose:
    # Don't use TQDM if verbose: it might mess up the verbose messages
    predictions_iterator = lambda x: x

  for prediction in predictions_iterator(predictions):
    # Attempt to connect to the database for executing.
    try:
      conn = sqlite3.connect(prediction['database_path'])
      conn.text_factory = str
    except sqlite3.OperationalError as e:
      print(e)
      print(prediction['database_path'])
      exit()

    empty_path = prediction['empty_database_path']
    try:
      empty_conn = sqlite3.connect(empty_path)
      empty_conn.text_factory = str
    except sqlite3.OperationalError as e:
      print(e)
      print(empty_path)
      exit()

    empty_cursor = empty_conn.cursor()
    cursor = conn.cursor()

    ofile.write('Example #' + str(i) + '\n')
    printable_utterance = u''.join(
        prediction['utterance']).encode('utf-8').strip()
    ofile.write(printable_utterance + '\n')

    if verbose:
      print('Finding the highest-rated prediction for utterance:\n\t' +
            printable_utterance)

    best_prediction, pred_results, exception_str, execution_time = execute_prediction(
        prediction, empty_cursor, cursor, case_sensitive, verbose)

    ofile.write('Predicted query:\n')
    if best_prediction:
      ofile.write('\t' + u''.join(best_prediction).encode('utf-8').strip() +
                  '\n')
    else:
      ofile.write('ERROR: Cannot write prediction %r\n' % best_prediction)
    ofile.write('Took %s s to execute' % execution_time)

    # If it didn't execute correctly, check why.
    if exception_str:
      ofile.write(exception_str + '\n')

      found_error = False
      for substring in SCHEMA_INCOHERENCE_STRINGS:
        if substring in exception_str.lower():
          schema_errors += 1
          found_error = True
          break

      if not found_error:
        for substring in SYNTAX_INCORRECTNESS_STRINGS:
          if substring in exception_str.lower():
            syntax_errors += 1
            found_error = True
            break

      if not found_error and 'timeout' in exception_str:
        ofile.write('Execution (predicted) took too long.\n')
        found_error = True
        timeouts += 1

      # If the error type hasn't been identified, exit and report it.
      if not found_error:
        print(best_prediction)
        print(exception_str)
        exit(1)

      # Predicted table should be empty for all of these cases.
      pred_results = []

    # Compare to gold and update metrics
    gold_query = prediction['gold']

    ofile.write('Gold query:\n')
    ofile.write('\t' + u''.join(gold_query).encode('utf-8').strip() + '\n')

    # Get the gold results
    if cache_dict is None or gold_query not in cache_dict:
      if printable_utterance not in cache_dict:
        if update_cache:
            if verbose:
              print('Trying to execute the gold query:\n\t' + gold_query)
            gold_results, gold_exception_str = try_executing_query(
                gold_query, cursor, case_sensitive, verbose)
      
            if gold_exception_str:
              gold_error += 1
              gold_results = []
            elif cache_dict is not None:
              cache_dict[u''.join(gold_query).decode('utf-8')] = gold_results
        else:
          print(gold_query)
          print(printable_utterance)
          raise ValueError('Cache miss!')

      else:
        gold_results = cache_dict[cache_dict[printable_utterance]]
    else:
      gold_results = cache_dict[gold_query]

    if best_prediction:
      string_same.append(string_acc(gold_query, best_prediction))
      col_f1, tab_f1 = col_tab_f1(prediction['schema'], gold_query,
                                  best_prediction)
      column_f1s.append(col_f1)
      table_f1s.append(tab_f1)
      ofile.write('Column F1: %f\n' % col_f1)
      ofile.write('Table F1: %f\n' % tab_f1)

      if 'order by' in gold_query:
        results_equivalent = pred_results == gold_results
      else:
        pred_set = set()
        gold_set = set()
        for pred in pred_results:
          if isinstance(pred, list):
            pred_set.add(u' '.join(
                [_convert_to_unicode_string(item) for item in pred]))
          else:
            pred_set.add(pred)
        for gold in gold_results:
          if isinstance(gold, list):
            gold_set.add(u' '.join(
                [_convert_to_unicode_string(item) for item in gold]))
          else:
            gold_set.add(gold)

        results_equivalent = pred_set == gold_set

    else:
      string_same.append(0.)
      ofile.write('Column F1: 0.')
      ofile.write('Table F1: 0.')
      column_f1s.append(0.)
      table_f1s.append(0.)

      conversion_errors += 1

      # Only consider correct if the gold table was empty.
      results_equivalent = gold_results == list()

    exec_results_same.append(int(results_equivalent))
    ofile.write('Execution was correct? ' + str(results_equivalent) + '\n')

    # Add some debugging information about the tables, and compute the
    # precisions.
    if pred_results:
      if not results_equivalent:
        ofile.write('Predicted table:\n')
        ofile.write(result_table_to_string(pred_results))

      precision.append(int(results_equivalent))
    elif best_prediction is None or not results_equivalent:
      ofile.write('Predicted table was EMPTY!\n')

    if gold_results:
      ofile.write('Gold table:\n')
      ofile.write(result_table_to_string(gold_results))

      recall.append(int(results_equivalent))
    else:
      ofile.write('Gold table was EMPTY!\n')

    ofile.write('\n')
    ofile.flush()

    conn.close()
    empty_conn.close()

    i += 1

  # Write the overall metrics to the file.
  num_empty_pred = len(precision)
  num_empty_gold = len(recall)

  precision = np.mean(np.array(precision))
  recall = np.mean(np.array(recall))

  execution_f1 = compute_f1(precision, recall)

  ofile.write('String accuracy: ' +
              '{0:.2f}'.format(100. * np.mean(np.array(string_same))) + '\n')
  ofile.write('Accuracy: ' +
              '{0:.2f}'.format(100. * np.mean(np.array(exec_results_same))) +
              '\n')
  ofile.write('Precision: ' + '{0:.2f}'.format(100. * precision) + ' ; ' +
              str(num_empty_pred) + ' nonempty predicted tables' + '\n')
  ofile.write('Recall: ' + '{0:.2f}'.format(100. * recall) + ' ; ' +
              str(num_empty_gold) + ' nonempty gold tables' + '\n')
  ofile.write('Execution F1: ' + '{0:.2f}'.format(100. * execution_f1) + '\n')
  ofile.write('Timeout: ' +
              '{0:.2f}'.format(timeouts * 100. / len(predictions)) + '\n')
  ofile.write('Gold did not execute: ' +
              '{0:.2f}'.format(gold_error * 100. / len(predictions)) + '\n')
  ofile.write('Average column F1: ' +
              '{0:.2f}'.format(100. * np.mean(np.array(column_f1s))) + '\n')
  ofile.write('Average table F1: ' +
              '{0:.2f}'.format(100. * np.mean(np.array(table_f1s))) + '\n')
  ofile.write('Schema errors: ' +
              '{0:.2f}'.format((schema_errors) * 100. / len(predictions)) +
              '\n')
  ofile.write('Syntax errors:  ' +
              '{0:.2f}'.format((syntax_errors) * 100. / len(predictions)) +
              '\n')
  ofile.write('Conversion errors: ' +
              '{0:.2f}'.format((conversion_errors * 100.) / len(predictions)) +
              '\n')


def main(predictions_filepath, output_filepath, cache_filepath, verbose):
  # Load the predictions filepath.
  with open(predictions_filepath) as infile:
    predictions = json.load(infile)
  print('Loaded %d predictions.' % len(predictions))

  # Load or create the cache dictionary mapping from gold queries to resulting
  # tables.
  cache_dict = None

  # Only instantiate the cache dict if using Spider.
  print('cache path: ' + cache_filepath)

  basefilename = os.path.basename(predictions_filepath).lower()

  if 'spider' not in basefilename:
    cache_dict = dict()
    if os.path.exists(cache_filepath):
      print('Loading cache from %s' % cache_filepath)
      with open(cache_filepath) as infile:
        cache_dict = json.load(infile)
      print('Loaded %d cached queries' % len(cache_dict))

  # Create the text file that results will be written to.
  with open(output_filepath, 'w') as ofile:
    execute_predictions(predictions, cache_dict, ofile,
                        'scholar' not in basefilename, verbose)

  if 'spider' not in basefilename:
    try:
      cache_str = json.dumps(cache_dict)
      with open(cache_filepath, 'w') as ofile:
        ofile.write(cache_str)
    except UnicodeDecodeError as e:
      print('Could not save the cache dict. Exception:')
      print(e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--predictions_filepath',
      type=str,
      help='Where the predictions JSON file is located.')
  parser.add_argument(
      '--output_filepath', type=str, help='Where to write the results.')
  parser.add_argument(
      '--cache_filepath', type=str, help='A cache of the gold tables.')
  parser.add_argument(
      '--verbose',
      type=bool,
      help='If set to True, evaluation will be verbose.')
  parser.add_argument(
      '--update_cache',
      type=bool,
      help='If set to True, will execute and cache gold queries.')
  args = parser.parse_args()

  main(args.predictions_filepath, args.output_filepath, args.cache_filepath,
       args.verbose, args.update_cache)

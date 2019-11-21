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
"""Utilties for converting to/from SqlSpan tuples to/from NLToSQLExample."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.xsp.data_preprocessing import abstract_sql
from language.xsp.data_preprocessing import schema_utils

from language.xsp.data_preprocessing.sql_parsing import VALID_GENERATED_TOKENS
from language.xsp.data_preprocessing.sql_utils import SchemaEntityCopy
from language.xsp.data_preprocessing.sql_utils import SQLAction


class ParseError(Exception):
  pass


def _find_column(table_name, column_name, example):
  """Find schema entity for column."""
  for table in example.model_input.tables:
    if table.original_table_name.lower() != table_name:
      continue
    for column in table.table_columns:
      if column.original_column_name.lower() == column_name:
        return column
  raise ParseError('No matching column for %s %s' % (table_name, column_name))


def _find_table(table_name, example):
  """Find schema entity for table."""
  for table in example.model_input.tables:
    if table.original_table_name.lower() == table_name:
      return table
  raise ParseError('No matching table for %s' % table_name)


def _add_table_copy(table_name, example):
  table = _find_table(table_name, example)
  example.gold_sql_query.actions.append(
      SQLAction(entity_copy=SchemaEntityCopy(copied_table=table)))


def _add_column_copy(table_name, column_name, example):
  column = _find_column(table_name, column_name, example)
  example.gold_sql_query.actions.append(
      SQLAction(entity_copy=SchemaEntityCopy(copied_column=column)))


def _add_generate_action(token, example):
  example.gold_sql_query.actions.append(SQLAction(symbol=token))


# TODO(petershaw): De-duplicate with function from `sql_parsing.py`.
def _add_value_literal(item_str, example):
  """Adds a value action to the output."""
  # Add quotes if [1] there aren't quotes and [2] it's not numeric
  if not item_str.replace(
      '.', '',
      1).isdigit() and item_str.count('"') < 2 and item_str.count('\'') < 2:
    item_str = "'" + item_str + "'"

  def find_and_add_copy_from_text(substr):
    """Finds a substring in the utterance and adds a copying action."""
    # It's a substring of the original utterance, but not so sure it could be
    # composed of wordpieces.
    found = False
    start_wordpiece = -1
    end_wordpiece = -1
    for i in range(len(example.model_input.utterance_wordpieces)):
      for j in range(i + 1, len(example.model_input.utterance_wordpieces) + 1):
        # Compose a string. If it has ##, that means it's a wordpiece, so should
        # not have a space in front.
        composed_pieces = ' '.join([
            wordpiece.wordpiece
            for wordpiece in example.model_input.utterance_wordpieces[i:j]
        ]).replace(' ##', '')
        if substr.lower() == composed_pieces:
          start_wordpiece = i
          end_wordpiece = j

          found = True
          break
      if found:
        break

    if start_wordpiece >= 0 and end_wordpiece >= 0:
      # Found wordpiece(s, when put together) comprising this item
      for i in range(start_wordpiece, end_wordpiece):
        action = SQLAction(
            utterance_copy=example.model_input.utterance_wordpieces[i])
        example.gold_sql_query.actions.append(action)
      return True
    return False

  # First, check if this string could be copied from the wordpiece-tokenized
  # inputs.
  quote_type = ''
  if item_str.lower() in example.model_input.original_utterance.lower():
    success = find_and_add_copy_from_text(item_str)

    if not success or item_str in VALID_GENERATED_TOKENS:
      example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

    return success or item_str in VALID_GENERATED_TOKENS

  elif item_str.startswith('\'') and item_str.endswith('\''):
    quote_type = '\''
  elif item_str.startswith('"') and item_str.endswith('"'):
    quote_type = '"'

  if quote_type:
    if item_str[1:-1].lower() in example.model_input.original_utterance.lower():
      example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

      success = find_and_add_copy_from_text(item_str[1:-1])
      if not success or item_str in VALID_GENERATED_TOKENS:
        example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

      example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

      return success or item_str in VALID_GENERATED_TOKENS
    elif item_str[1] == '%' and item_str[-2] == '%' and item_str[2:-2].lower(
    ) in example.model_input.original_utterance.lower():
      example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))
      example.gold_sql_query.actions.append(SQLAction(symbol='%'))

      success = find_and_add_copy_from_text(item_str[2:-2])
      if not success or item_str in VALID_GENERATED_TOKENS:
        example.gold_sql_query.actions.append(SQLAction(symbol=item_str[2:-2]))

      example.gold_sql_query.actions.append(SQLAction(symbol='%'))
      example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

      return success or item_str in VALID_GENERATED_TOKENS

  # Just add it as choice from the output vocabulary
  if u'u s a' in item_str:
    raise ValueError('WHAT????????')
  example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

  return item_str in VALID_GENERATED_TOKENS


def populate_example_from_sql_spans(sql_spans, example):
  """Creates a sequence of output / decoder actions from sql_spans.

  Args:
    sql_spans: List of SqlSpan tuples.
    example: The NLToSQLExample object to add output actions.

  Raises:
    ParseError: if the SQL query can't be parsed.

  Returns:
    Successful copy.
  """
  successful_copy = True
  for sql_span in sql_spans:
    if sql_span.sql_token:
      _add_generate_action(sql_span.sql_token, example)
    elif sql_span.value_literal:
      successful_copy = _add_value_literal(sql_span.value_literal,
                                           example) and successful_copy
    elif sql_span.column:
      _add_column_copy(sql_span.column.table_name, sql_span.column.column_name,
                       example)
    elif sql_span.table_name:
      _add_table_copy(sql_span.table_name, example)
    elif sql_span.nested_statement:
      successful_copy = populate_example_from_sql_spans(
          sql_span.nested_statement, example) and successful_copy
    else:
      raise ParseError('Invalid SqlSpan: %s' % sql_span)
  return successful_copy


def _nested_statement_end_idx(sql_query, action_idx):
  """If acion_idx is start of nested statement, return end_idx, else None."""
  if action_idx + 2 >= len(sql_query.actions):
    return None
  current_action = sql_query.actions[action_idx]
  next_action = sql_query.actions[action_idx + 1]
  if current_action.symbol == '(' and next_action.symbol.lower() == 'select':
    open_parens = 0
    for end_idx in range(action_idx + 1, len(sql_query.actions)):
      nested_action = sql_query.actions[end_idx]
      if nested_action.symbol == '(':
        open_parens += 1
      if nested_action.symbol == ')':
        if open_parens > 0:
          open_parens -= 1
        else:
          return end_idx
    raise ParseError('Could not find end index for nested statement: %s' %
                     sql_query)
  return None


def _get_sql_spans_from_actions(actions):
  """Get list of SqlSpan tuples from list of SQLAction objects."""
  sql_spans = []
  action_idx = 0
  while action_idx < len(actions):
    action = actions[action_idx]

    # First, check if we need to recurse on a nested statement.
    nested_statement_end_idx = _nested_statement_end_idx(actions, action_idx)
    if nested_statement_end_idx:
      nested_sql_spans = _get_sql_spans_from_actions(
          actions[action_idx:nested_statement_end_idx])
      sql_spans.append(
          abstract_sql.make_sql_span(nested_statement=nested_sql_spans))
      # Set action index to end of nested statement.
      action_idx = nested_statement_end_idx + 1
      continue

    if action.symbol:
      sql_spans.append(abstract_sql.make_sql_span(sql_token=action.symbol))
    elif action.entity_copy:
      copy_action = action.entity_copy
      if copy_action.copied_table:
        sql_spans.append(
            abstract_sql.make_sql_span(
                table_name=copy_action.copied_table.original_table_name))
      else:
        sql_spans.append(
            abstract_sql.make_sql_span(
                table_name=copy_action.copied_column.table_name,
                column_name=copy_action.copied_column.original_column_name))
    else:
      sql_spans.append(
          abstract_sql.make_sql_span(
              value_literal=action.utterance_copy.wordpiece))
    # Increment action index.
    action_idx += 1

  return sql_spans


def get_sql_spans_from_query(example):
  """Get list of SqlSpan tuples from NLToSQLExample object."""
  return _get_sql_spans_from_actions(example.gold_sql_query.actions)


def wikisql_db_to_table_tuples(db_name, db):
  """Return list of abstract_sql.TableSchema from a dict describing a DB."""
  del db_name
  table_schemas = list()
  for table_name, columns in db.items():
    column_names = [column['field name'].lower() for column in columns]
    table_schemas.append(
        abstract_sql.TableSchema(
            table_name=table_name.lower(), column_names=column_names))
  return table_schemas


def spider_db_to_table_tuples(db):
  """Return list of abstract_sql.TableSchema from spider json."""
  # The format of the db json object is documented here:
  # https://github.com/taoyds/spider#tables
  # List of string table names.
  table_names = db['table_names_original']
  # List of lists with [table_idx, column_name].
  column_list = db['column_names_original']
  column_names_indexed_by_table = [[] for _ in table_names]
  for column in column_list:
    column_names_indexed_by_table[column[0]].append(column[1].lower())
  table_schemas = []
  for table_name, column_names in zip(table_names,
                                      column_names_indexed_by_table):
    table_schemas.append(
        abstract_sql.TableSchema(
            table_name=table_name.lower(),
            column_names=column_names,
        ))
  return table_schemas


def michigan_db_to_table_tuples(db):
  """Returns list of abstract_sql.TableSchema from Michigan schema object."""
  table_schemas = list()
  for table_name, columns in db.items():
    table_schemas.append(
        abstract_sql.TableSchema(
            table_name=table_name.lower(),
            column_names=[column['field name'].lower() for column in columns]))
  return table_schemas


def spider_db_to_foreign_key_tuples(db):
  """Return list of abstract_sql.ForiegnKeyRelation."""
  # The format of the db json object is documented here:
  # https://github.com/taoyds/spider#tables
  # List of string table names.
  table_names = db['table_names_original']
  # List of lists with [table_idx, column_name].
  column_list = db['column_names_original']
  foreign_keys = []
  for foreign_key_idx_pair in db['foreign_keys']:
    child_column_id = foreign_key_idx_pair[0]
    parent_column_id = foreign_key_idx_pair[1]
    child_table_id = column_list[child_column_id][0]
    child_column_name = column_list[child_column_id][1].lower()
    parent_table_id = column_list[parent_column_id][0]
    parent_column_name = column_list[parent_column_id][1].lower()
    child_table_name = table_names[child_table_id].lower()
    parent_table_name = table_names[parent_table_id].lower()
    foreign_keys.append(
        abstract_sql.ForeignKeyRelation(
            child_table=child_table_name,
            parent_table=parent_table_name,
            child_column=child_column_name,
            parent_column=parent_column_name))
  return foreign_keys


def michigan_db_to_foreign_key_tuples_orcale(dataset_name):
  """Returns a list of abstract_sql.ForeignKeyRelation."""
  # Uses hand curated oracle foreign key annotations.
  if dataset_name == 'academic':
    return [
        abstract_sql.ForeignKeyRelation('publication', 'writes', 'pid', 'pid'),
        abstract_sql.ForeignKeyRelation('author', 'writes', 'aid', 'aid'),
        abstract_sql.ForeignKeyRelation('journal', 'publication', 'jid', 'jid'),
        abstract_sql.ForeignKeyRelation('conference', 'publication', 'cid',
                                        'cid'),
        abstract_sql.ForeignKeyRelation('publication', 'publication_keyword',
                                        'pid', 'pid'),
        abstract_sql.ForeignKeyRelation('keyword', 'publication_keyword', 'kid',
                                        'kid'),
        abstract_sql.ForeignKeyRelation('author', 'organization', 'oid', 'oid'),
        abstract_sql.ForeignKeyRelation('author', 'domain_author', 'aid',
                                        'aid'),
        abstract_sql.ForeignKeyRelation('domain', 'domain_author', 'did',
                                        'did'),
        abstract_sql.ForeignKeyRelation('domain', 'domain_publication', 'did',
                                        'did'),
        abstract_sql.ForeignKeyRelation('domain_publication', 'publication',
                                        'pid', 'pid'),
        abstract_sql.ForeignKeyRelation('cite', 'publication', 'cited', 'pid'),
        abstract_sql.ForeignKeyRelation('cite', 'publication', 'citing', 'pid'),
        abstract_sql.ForeignKeyRelation('domain', 'domain_keyword', 'did',
                                        'did'),
        abstract_sql.ForeignKeyRelation('domain_keyword', 'keyword', 'kid',
                                        'kid'),
        abstract_sql.ForeignKeyRelation('domain', 'domain_journal', 'did',
                                        'did'),
        abstract_sql.ForeignKeyRelation('domain_journal', 'journal', 'jid',
                                        'jid'),
        abstract_sql.ForeignKeyRelation('conference', 'domain_conference',
                                        'cid', 'cid'),
        abstract_sql.ForeignKeyRelation('domain', 'domain_conference', 'did',
                                        'did'),
    ]
  elif dataset_name == 'atis':
    return [
        abstract_sql.ForeignKeyRelation('airport_service', 'city', 'city_code',
                                        'city_code'),
        abstract_sql.ForeignKeyRelation('airport_service', 'flight',
                                        'airport_code', 'from_airport'),
        abstract_sql.ForeignKeyRelation('airport_service', 'flight',
                                        'airport_code', 'to_airport'),
        abstract_sql.ForeignKeyRelation('date_day', 'days', 'day_name',
                                        'day_name'),
        abstract_sql.ForeignKeyRelation('days', 'flight', 'days_code',
                                        'flight_days'),
        abstract_sql.ForeignKeyRelation('fare', 'flight_fare', 'fare_id',
                                        'fare_id'),
        abstract_sql.ForeignKeyRelation('flight', 'flight_fare', 'flight_id',
                                        'flight_id'),
        abstract_sql.ForeignKeyRelation('fare', 'fare_basis', 'fare_basis_code',
                                        'fare_basis_code'),
        abstract_sql.ForeignKeyRelation('flight', 'flight_stop', 'flight_id',
                                        'flight_id'),
        abstract_sql.ForeignKeyRelation('days', 'fare_basis', 'days_code',
                                        'basis_days'),
        abstract_sql.ForeignKeyRelation('airport_service', 'flight_stop',
                                        'airport_code', 'stop_airport'),
        abstract_sql.ForeignKeyRelation('city', 'ground_service', 'city_code',
                                        'city_code'),
        abstract_sql.ForeignKeyRelation('airline', 'flight', 'airline_code',
                                        'airline_code'),
        abstract_sql.ForeignKeyRelation('airport', 'airport_service',
                                        'airport_code', 'airport_code'),
        abstract_sql.ForeignKeyRelation('flight', 'food_service', 'meal_code',
                                        'meal_code'),
        abstract_sql.ForeignKeyRelation('aircraft', 'equipment_sequence',
                                        'aircraft_code', 'aircraft_code'),
        abstract_sql.ForeignKeyRelation('equipment_sequence', 'flight',
                                        'aircraft_code_sequence',
                                        'aircraft_code_sequence'),
        abstract_sql.ForeignKeyRelation('city', 'state', 'state_code',
                                        'state_code'),
        abstract_sql.ForeignKeyRelation('airport', 'flight', 'airport_code',
                                        'to_airport'),
        abstract_sql.ForeignKeyRelation('airport', 'ground_service',
                                        'airport_code', 'airport_code'),
        abstract_sql.ForeignKeyRelation('airport', 'flight', 'airport_code',
                                        'from_airport'),
        abstract_sql.ForeignKeyRelation('airport_service', 'fare',
                                        'airport_code', 'to_airport'),
        abstract_sql.ForeignKeyRelation('airport_service', 'fare',
                                        'airport_code', 'from_airport'),
        abstract_sql.ForeignKeyRelation('flight', 'flight_leg', 'flight_id',
                                        'flight_id'),
        abstract_sql.ForeignKeyRelation('flight', 'flight_leg', 'flight_id',
                                        'leg_flight'),
        abstract_sql.ForeignKeyRelation('class_of_service', 'fare_basis',
                                        'booking_class', 'booking_class'),
        abstract_sql.ForeignKeyRelation('airport', 'state', 'state_code',
                                        'state_code'),
        abstract_sql.ForeignKeyRelation('airport', 'flight_stop',
                                        'airport_code', 'stop_airport'),
        abstract_sql.ForeignKeyRelation('fare', 'restriction',
                                        'restriction_code', 'restriction_code'),
    ]
  elif dataset_name == 'geography':
    return [
        abstract_sql.ForeignKeyRelation('border_info', 'state', 'border',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('river', 'state', 'traverse',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('city', 'state', 'city_name',
                                        'capital'),
        abstract_sql.ForeignKeyRelation('border_info', 'state', 'state_name',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('city', 'state', 'state_name',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('border_info', 'river', 'border',
                                        'traverse'),
        abstract_sql.ForeignKeyRelation('highlow', 'state', 'state_name',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('border_info', 'border_info', 'border',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('highlow', 'river', 'state_name',
                                        'traverse'),
        abstract_sql.ForeignKeyRelation('border_info', 'river', 'state_name',
                                        'traverse'),
        abstract_sql.ForeignKeyRelation('city', 'river', 'state_name',
                                        'traverse'),
        abstract_sql.ForeignKeyRelation('border_info', 'highlow', 'border',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('border_info', 'city', 'border',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('state', 'state', 'state_name',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('border_info', 'border_info', 'border',
                                        'border'),
        abstract_sql.ForeignKeyRelation('city', 'highlow', 'state_name',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('highlow', 'state', 'highest_point',
                                        'capital'),
        abstract_sql.ForeignKeyRelation('border_info', 'lake', 'border',
                                        'state_name'),
        abstract_sql.ForeignKeyRelation('river', 'river', 'river_name',
                                        'river_name'),
    ]
  elif dataset_name == 'restaurants':
    return [
        abstract_sql.ForeignKeyRelation('location', 'restaurant',
                                        'restaurant_id', 'id'),
        abstract_sql.ForeignKeyRelation('geographic', 'restaurant', 'city_name',
                                        'city_name'),
    ]
  else:
    raise ValueError('Unknown dataset: %s' % dataset_name)


def michigan_db_to_foreign_key_tuples(db):
  """Returns a list of abstract_sql.ForeignKeyRelation."""
  # Michigan doesn't come with a gold standard of *which* columns are foreign
  # keys with one another. For now: assume that they're foreign keys if the
  # schema marks them as foreign keys and the column name is the same in both
  # tables.
  # TODO(alanesuhr): This is a bit hacky because there might be exceptions
  #  (e.g., columns that are foreign keys with different names).

  foreign_keys = list()
  for table_name, columns in db.items():
    for data in columns:
      could_have_link = schema_utils.column_is_foreign_key(
          data) or schema_utils.column_is_primary_key(data)
      column_name = data['field name'].lower()

      if could_have_link:
        for other_table_name, other_columns in db.items():
          if table_name != other_table_name:
            for other_data in other_columns:
              other_column = other_data['field name'].lower()
              if other_column == column_name:
                if schema_utils.column_is_foreign_key(
                    other_data) or schema_utils.column_is_primary_key(
                        other_data):
                  foreign_keys.append(
                      abstract_sql.ForeignKeyRelation(
                          child_table=table_name.lower(),
                          parent_table=other_table_name.lower(),
                          child_column=column_name,
                          parent_column=other_column))
  return foreign_keys


def spider_table_schemas_map(schema):
  """Returns map of database id to a list of TableSchema tuples."""
  # The format of the schema json object is documented here:
  # https://github.com/taoyds/spider#tables
  return {db['db_id']: spider_db_to_table_tuples(db) for db in schema}


def wikisql_table_schemas_map(schema):
  """Returns map of database id to a list of TableSchema tuples."""
  # The format of the schema json object is documented here:
  # https://github.com/taoyds/wikisql#tables
  return {
      db_name: wikisql_db_to_table_tuples(db_name, db)
      for db_name, db in schema.items()
  }


def spider_foreign_keys_map(schema):
  """Returns map of database id to a list of ForiegnKeyRelation tuples."""
  # The format of the schema json object is documented here:
  # https://github.com/taoyds/spider#tables
  return {db['db_id']: spider_db_to_foreign_key_tuples(db) for db in schema}


def populate_abstract_sql(example, sql_string, table_schemas):
  """Populate SQL in example.

  Args:
    example: NLToSQLExample instance with utterance populated.
    sql_string: SQL query as string.
    table_schemas: List of TableSchema tuples.

  Returns:
    Successful copy.
  """
  sql_spans = abstract_sql.sql_to_sql_spans(sql_string, table_schemas)
  sql_spans = abstract_sql.replace_from_clause(sql_spans)
  return populate_example_from_sql_spans(sql_spans, example)


def restore_predicted_sql(sql_string, table_schemas, foreign_keys):
  """Restore FROM clause in predicted SQL.

  TODO(petershaw): Add call to this function from run_inference.py.

  Args:
    sql_string: SQL query as string.
    table_schemas: List of TableSchema tuples.
    foreign_keys: List of ForeignKeyRelation tuples.

  Returns:
    SQL query with restored FROM clause as a string.
  """
  sql_spans = abstract_sql.sql_to_sql_spans(
      sql_string, table_schemas, lowercase=False)
  sql_spans = abstract_sql.restore_from_clause(sql_spans, foreign_keys)
  return abstract_sql.sql_spans_to_string(sql_spans)

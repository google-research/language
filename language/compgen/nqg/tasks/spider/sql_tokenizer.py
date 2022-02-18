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
"""Utilities for tokenizing SQL."""

import sqlparse


def _is_whitespace(sqlparse_token):
  return sqlparse_token.ttype == sqlparse.tokens.Whitespace


def tokenize_sql(sql_exp):
  sql_exp = sql_exp.lower()
  sql_exp = sql_exp.rstrip(";")
  parse = sqlparse.parse(sql_exp)
  sql = parse[0]
  flat_tokens = sql.flatten()
  sql_tokens = [
      token.value for token in flat_tokens if not _is_whitespace(token)
  ]
  return sql_tokens

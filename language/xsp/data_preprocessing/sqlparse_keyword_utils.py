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
# Lint as: python3
"""Utilities for removing conflicting keywords in sqlparse package."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import sqlparse


def remove_bad_sqlparse_keywords():
  """Removes keywords from the sqlparse keyword lists.

  This is a relatively hacky solution (removing from the globally-scoped
  internal keyword lists) but sqlparse does not offer an API for controlling
  keywords.

  These keywords have to be removed because they are often used in Spider as
  table or column names, and parsing therefore fails otherwise.

  This fuction is designed to be safe if it is called more than once.
  """
  for keyword in [
      'CAST', 'CATALOG_NAME', 'CHARACTER', 'CHARACTERISTICS', 'CLASS',
      'CONNECTION', 'DATE', 'DOMAIN', 'HOST', 'LANGUAGE', 'LENGTH', 'LEVEL',
      'LOCATION', 'LONG', 'MATCH', 'MONTH', 'OWNER', 'POSITION', 'RESULT',
      'ROLE', 'SHARE', 'SHOW', 'SOURCE', 'START', 'TEXT', 'TYPE', 'UID', 'USER',
      'YEAR', 'DAY', 'PARTIAL', 'BIT', 'PUBLIC', 'QUARTER', 'NATIONAL', 'OUT', 'CURRENT', 'METHOD', 'FREE', 'SECURITY', 'OF', 'FIRST', 'UNKNOWN', 'FORWARD', 'FINAL', 'ENGINE', 'RESET', 'NONE', 'HOUR', 'GENERAL', 'END', 'NO', 'ALL', 'PRIMARY', 'BOTH'
  ]:
    if keyword in sqlparse.keywords.KEYWORDS:
      del sqlparse.keywords.KEYWORDS[keyword]
  for oracle_keyword in [
      'BLOCK', 'EVENTS', 'ROLES', 'SECTION', 'STATEMENT_ID', 'TIME', 'STOP', 'PRIVATE', 'RESTRICTED'
  ]:
    if oracle_keyword in sqlparse.keywords.KEYWORDS_ORACLE:
      del sqlparse.keywords.KEYWORDS_ORACLE[oracle_keyword]
  if 'CHARACTER' in sqlparse.keywords.KEYWORDS_PLPGSQL:
    del sqlparse.keywords.KEYWORDS_PLPGSQL['CHARACTER']
  if 'BOX' in sqlparse.keywords.KEYWORDS_PLPGSQL:
    del sqlparse.keywords.KEYWORDS_PLPGSQL['BOX']
  if 'END' in sqlparse.keywords.KEYWORDS_COMMON:
    del sqlparse.keywords.KEYWORDS_COMMON['END']

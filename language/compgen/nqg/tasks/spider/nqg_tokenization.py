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
"""Functions to preprocess Spider to accomondate tokenization of NQG.

NQG performs simple space-separated tokenization. The tokenization in this
module to accomondate this primarily involves splitting on punctuation, e.g.
`"foo"` becomes `" foo "`.
"""

import string

from language.compgen.nqg.tasks.spider import sql_tokenizer


def _split_punc(source):
  """Split leading or trailing punctuation."""
  tokens = source.split(" ")
  new_tokens = []
  for token in tokens:
    if all(char in string.punctuation for char in token):
      new_tokens.append(token)
      continue

    leading_punc = None
    for punc in string.punctuation:
      if token.startswith(punc):
        leading_punc = punc
        token = token.lstrip(punc)
        break

    trailing_punc = None
    for punc in string.punctuation:
      if token.endswith(punc):
        trailing_punc = punc
        token = token.rstrip(punc)
        break

    if leading_punc:
      new_tokens.append(leading_punc)
    if token:
      new_tokens.append(token)
    if trailing_punc:
      new_tokens.append(trailing_punc)

  return " ".join(new_tokens)


def process_source(source):
  source = _split_punc(source)
  # Remove extra whitespace.
  source = " ".join(source.split())
  return source


def process_target(target):
  """Preprocess target for space-separated tokenization."""
  target_sql_tokens = sql_tokenizer.tokenize_sql(target)
  target = " ".join(target_sql_tokens)
  target = _split_punc(target)
  # Split punc twice, to handle "%foo%" wrapped in two punc chars.
  # TODO(petershaw): Update _split_punc to correctly handle this case with
  # a single invocation.
  target = _split_punc(target)
  # Remove extra whitespace.
  target = " ".join(target.split())
  return target

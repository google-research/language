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
"""Utilities for tokenization.

We use tokens to refer to coarsely tokenized (e.g. split on spaces) tokens
which is implicitly used for tokenization by the QCFG rules and parser.
We use wordpieces to refer to the wordpieces tokenized inputs for BERT.
"""

from official.nlp.bert import tokenization

# Map for special tokens.
SPECIAL_MAP = {
    "m0": "[unused0]",
    "m1": "[unused1]"
}


def get_tokenizer(bert_vocab_file):
  tokenizer = tokenization.FullTokenizer(bert_vocab_file, do_lower_case=True)
  return tokenizer


def get_wordpiece_inputs(tokens, tokenizer, verbose=False):
  """Returns inputs related to tokenization.

  The resulting set of tensors includes alignment information between the
  space-separated token sequence (which the QCFG parser uses) and the resulting
  wordpiece sequence (which the neural encoder uses). There is always a
  one-to-many correspondance between tokens and wordpieces.

  Args:
    tokens: List of string tokens.
    tokenizer: `tokenization.FullTokenizer` instance or equivalent.
    verbose: Print debug logging if True.

  Returns:
    A tuple of (wordpiece_ids, num_wordpieces, token_start_wp_idx,
    token_end_wp_idx):
      wordpiece_ids: List of wordpiece ids for input sequence.
      num_wordpieces: Number of wordpieces.
      token_start_wp_idx: Specifies the index in wordpiece_ids for the first
        wordpiece for each input token (inclusive).
      token_end_wp_idx: Specifies the index in wordpiece_ids for the last
        wordpiece for each input token (inclusive).
  """
  wordpiece_idx = 1
  token_start_wp_idx = []
  token_end_wp_idx = []
  wordpieces = []

  for token in tokens:
    token_start_wp_idx.append(wordpiece_idx)

    if token in SPECIAL_MAP:
      wordpieces.append(SPECIAL_MAP[token])
      wordpiece_idx += 1
    else:
      token_wordpieces = tokenizer.tokenize(token)
      wordpieces.extend(token_wordpieces)
      wordpiece_idx += len(token_wordpieces)

    # Inclusive end idx.
    token_end_wp_idx.append(wordpiece_idx - 1)

  if verbose:
    print("token_start_wp_idx: %s" % token_start_wp_idx)
    print("token_end_wp_idx: %s" % token_end_wp_idx)

  if len(token_start_wp_idx) != len(tokens):
    raise ValueError("Bad token alignment!")
  if len(token_end_wp_idx) != len(tokens):
    raise ValueError("Bad token alignment!")

  wordpieces = ["[CLS]"] + wordpieces + ["[SEP]"]
  wordpiece_ids = tokenizer.convert_tokens_to_ids(wordpieces)
  num_wordpieces = len(wordpiece_ids)

  if verbose:
    print("wordpieces: %s" % wordpieces)
    print("wordpiece_ids: %s" % wordpiece_ids)

  return (wordpiece_ids, num_wordpieces, token_start_wp_idx, token_end_wp_idx)

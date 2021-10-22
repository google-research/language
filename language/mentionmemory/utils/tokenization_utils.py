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
"""Tokenizer utils."""



from absl import logging
from typing_extensions import Protocol


class TokenConverter(Protocol):
  """Protocol definition for TokenConverter.

  This is an object (such as a vocab or tokenizer) which can convert a sequence
  of tokens to a sequence of IDs. Note that every Tokenizer is also a
  TokenConverter.
  """

  def convert_tokens_to_ids(self, tokens):
    """Converts a list of string tokens to a list of vocab identifiers."""
    pass


class Tokenizer(TokenConverter, Protocol):
  """Protocol definition for Tokenizer type/class.

  This encapsulates the core methods needed from BERT's tokenizer.
  """

  def tokenize(self, text):
    """Tokenizes a string (in raw bytes) return a list of string tokens."""
    pass


def convert_tokens_to_ids_and_pad(
    word_pieces, max_length,
    token_converter):
  """Converts the input tokens to padded ids and mask.

  Args:
    word_pieces: list of word-piece tokens.
    max_length: length sequences should be padded/truncated to.
    token_converter: the tokenizer used.

  Returns:

    word_piece_ids: IDs of wordpieces, padded or truncated to max_length.
    word_piece_mask: a mask for the wordpieces.

  """
  word_piece_ids = token_converter.convert_tokens_to_ids(word_pieces)
  word_piece_mask = [1] * len(word_pieces)

  # Retrieve pad token id from tokenizer. The method expects a list, so we pass
  # a list with a single token ("[PAD]") and  retrieve the first id returned.
  pad_id = token_converter.convert_tokens_to_ids(["[PAD]"])[0]
  while len(word_pieces) < max_length:
    word_pieces.append("")
    word_piece_ids.append(pad_id)
    word_piece_mask.append(0)

  return word_piece_ids, word_piece_mask


def tokenize_with_mention_spans(
    tokenizer,
    sentence,
    spans,
    max_length,
    add_bert_tokens = True,
    allow_truncated_spans = False
):
  """Tokenizes and resolves byte-offsets to word-offsets.

  Given a sentence and a set of inclusive byte-offset spans, applies a BERT
  tokenizer to the sentence and returns inclusive word-piece spans.

  Args:
    tokenizer: a BERT tokenizer
    sentence: a string, sentence to be tokenized
    spans: [first token, last token] byte-offset spans. The spans should be
      sorted by start_offset, and should also be non-overlapping.
    max_length: the maximum length to truncate or pad to.
    add_bert_tokens: whether to add special BERT tokens (CLS, SEP) to the input.
    allow_truncated_spans: when it is okay to drop spans due to length. This
      should be false for official evals!

  Returns:
    word_pieces: tokenized word pieces.
    word_piece_ids: IDs of tokenized word pieces.
    word_piece_mask: mask of word pieces (for padding).
    token_spans: inclusive word-piece spans of the input byte-offset spans.
    span_indexes: indexes with respect to the input spans corresponding to
      output token spans. This is necessary because some input spans can be
      truncated or otherwise dropped.

  Raises:
    RuntimeError: if allow_truncated_spans is False but spans are dropped due
      to length.
  """
  sentence = sentence.encode("utf-8")

  last_index = 0
  last_start = 0
  word_pieces = []
  if add_bert_tokens:
    word_pieces.append("[CLS]")

  token_spans = []
  span_indexes = []
  for i, span in enumerate(spans):
    if span[0] < last_start:
      raise ValueError("Spans not in sequential order: %s" % (str(spans)))
    if span[1] < span[0]:
      raise ValueError("Span end before start: %s" % (str(span)))
    if span[0] < last_index:
      continue
    pre_span_string = sentence[last_index:span[0]]
    span_string = sentence[span[0]:span[1] + 1]
    last_index = span[1] + 1
    last_start = span[0]

    pre_span_word_pieces = tokenizer.tokenize(pre_span_string)
    word_pieces += pre_span_word_pieces
    span_token_start = len(word_pieces)
    span_word_pieces = tokenizer.tokenize(span_string)
    # span_word_pieces might be empty if the span consisted only of characters
    # discarded by the tokenizer.
    if span_word_pieces:
      word_pieces += span_word_pieces
      span_token_end = len(word_pieces) - 1
      token_spans.append((span_token_start, span_token_end))
      span_indexes.append(i)

  remaining_word_pieces = tokenizer.tokenize(sentence[last_index:])

  word_pieces += remaining_word_pieces

  last_token_idx = max_length - int(add_bert_tokens)
  if len(word_pieces) > last_token_idx:
    truncated_token_spans = [
        (start, end) for start, end in token_spans if end < last_token_idx
    ]
    num_dropped = len(token_spans) - len(truncated_token_spans)
    if num_dropped:
      logging.warn("Dropped %d spans due to length.", num_dropped)
      logging.warn("%d word pieces.", len(word_pieces))
      if not allow_truncated_spans:
        raise RuntimeError("Dropping spans not allowed when "
                           "allow_truncated_spans is False")
    word_pieces = word_pieces[:last_token_idx]
  else:
    truncated_token_spans = token_spans

  span_indexes = span_indexes[:len(truncated_token_spans)]

  if add_bert_tokens:
    word_pieces.append("[SEP]")
  word_piece_ids, word_piece_mask = convert_tokens_to_ids_and_pad(
      word_pieces, max_length, tokenizer)

  return (word_pieces, word_piece_ids, word_piece_mask, truncated_token_spans,
          span_indexes)

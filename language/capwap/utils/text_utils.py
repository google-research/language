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
"""Utility functions for dealing with text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import string
from bert import tokenization
from language.capwap.utils import nltk_utils
from language.capwap.utils import tensor_utils
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import lookup as contrib_lookup

TextInputs = collections.namedtuple(
    "TextInputs", ["token_ids", "mask", "segment_ids", "positions"])

TextOutputs = collections.namedtuple("TextLabels", ["token_ids", "mask"])

# ------------------------------------------------------------------------------
#
# General purpose text functions for masking/unmasking.
#
# ------------------------------------------------------------------------------


class Vocab(object):
  """Wrapper around the BERT tokenizer and vocabulary."""
  PAD = "[PAD]"
  UNK = "[UNK]"
  SEP = "[SEP]"
  CLS = "[CLS]"
  IMG = "[IMG]"
  ANS = "[A]"
  QUE = "[Q]"

  def __init__(self, vocab_file, do_lower_case):
    # Load BERT tokenizer.
    self._tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    if self.IMG not in self:
      # Override [unused0] to point to IMG.
      idx = self._tokenizer.vocab.pop("[unused0]")
      self._tokenizer.vocab[self.IMG] = idx
      self._tokenizer.inv_vocab[idx] = self.IMG

    if self.ANS not in self:
      # Override [unused1] to point to ANS.
      idx = self._tokenizer.vocab.pop("[unused1]")
      self._tokenizer.vocab[self.ANS] = idx
      self._tokenizer.inv_vocab[idx] = self.ANS

    if self.QUE not in self:
      # Override [unused2] to point to QUE.
      idx = self._tokenizer.vocab.pop("[unused2]")
      self._tokenizer.vocab[self.QUE] = idx
      self._tokenizer.inv_vocab[idx] = self.QUE

    # Validate
    for i in range(len(self)):
      assert i in self._tokenizer.inv_vocab
    for special_token in [self.PAD, self.UNK, self.SEP, self.CLS]:
      assert special_token in self

  def __len__(self):
    return len(self._tokenizer.vocab)

  def __contains__(self, token):
    return token in self._tokenizer.vocab

  def t2i(self, token):
    return self._tokenizer.vocab[token]

  def i2t(self, index):
    return self._tokenizer.inv_vocab[index]

  def tokenize(self, text):
    """Convert text to word pieces."""
    return self._tokenizer.tokenize(text)

  @staticmethod
  def clean(wordpieces):
    """Clean word pieces."""
    if Vocab.CLS in wordpieces:
      idx = wordpieces.index(Vocab.CLS)
      wordpieces = wordpieces[idx + 1:]
    if Vocab.SEP in wordpieces:
      idx = wordpieces.index(Vocab.SEP)
      wordpieces = wordpieces[:idx]
    if Vocab.PAD in wordpieces:
      wordpieces = [w for w in wordpieces if w != Vocab.PAD]

    # Various adhoc hacks.
    adjusted = []
    for w in wordpieces:
      # Remove non-ascii.
      try:
        w.encode(encoding="utf-8").decode("ascii")
      except UnicodeDecodeError:
        continue
      # Remove [unused*]
      if w.startswith("[unused"):
        continue
      # Remove repeated word.
      if not w.startswith("##") and adjusted and adjusted[-1] == w:
        continue
      adjusted.append(w)

    return adjusted

  @staticmethod
  def detokenize(wordpieces):
    """Convert word pieces to text."""
    wordpieces = Vocab.clean(wordpieces)
    tokens = []
    for w in wordpieces:
      if w.startswith("##") and len(tokens):
        tokens[-1] = tokens[-1] + w.lstrip("##")
      else:
        tokens.append(w)
    return " ".join(tokens)

  def get_string_lookup_table(self):
    unk_idx = self._tokenizer.vocab[self.UNK]
    ordered = [self.i2t(i) for i in range(len(self))]
    return contrib_lookup.index_table_from_tensor(
        np.array(ordered), default_value=unk_idx)

  @classmethod
  def load(cls, path):
    do_lower_case = "uncased" in path or "cased" not in path
    return cls(path, do_lower_case)


# ------------------------------------------------------------------------------
#
# General purpose text functions for masking/unmasking.
#
# ------------------------------------------------------------------------------


def get_token_mask(token_ids, stop_id):
  """Create mask for all ids past stop_id (inclusive)."""
  batch_size = tensor_utils.shape(token_ids, 0)
  num_tokens = tensor_utils.shape(token_ids, 1)

  # Create position matrix.
  idx_range = tf.expand_dims(tf.range(num_tokens), 0)
  idx_range = tf.tile(idx_range, [batch_size, 1])

  # Find positions of stop_id.
  stop_positions = tf.where(
      condition=tf.equal(token_ids, stop_id),
      x=idx_range,
      y=tf.fill([batch_size, num_tokens], num_tokens))

  # Find earliest stop position (length).
  stop_positions = tf.reduce_min(stop_positions, -1)

  # Mask out all tokens at positions > stop_id.
  mask = tf.less_equal(idx_range, tf.expand_dims(stop_positions, -1))

  return tf.cast(mask, tf.int32)


def get_random_span(text, p, max_span_len, max_iter=10):
  """Get random subspan from text token sequence, following heuristics.

  Heuristics:
    1) Should not start or end mid-wordpiece.
    2) Must contain at least one non-stopword token.
    3) Length should be drawn from Geo(p) and less than max_span_len.

  Args:
    text: <string> [], space-separated token string.
    p: <float32> Geometric distribution parameter.
    max_span_len: Length to pad or truncate to.
    max_iter: Maximum rejection sampling iterations.

  Returns:
    span_wid: <string>
  """
  # Split text into tokens.
  tokens = tf.string_split([text]).values
  seq_len = tf.size(tokens)

  def reject(start, end):
    """Reject span sample."""
    span = tokens[start:end + 1]
    wordpiece_boundary = tf.logical_or(
        tf.strings.regex_full_match(span[0], r"^##.*"),
        tf.strings.regex_full_match(span[-1], r"^##.*"))
    span = tokens[start:end]
    stopwords = list(nltk_utils.get_stopwords() | set(string.punctuation))
    non_stopword = tf.setdiff1d(span, stopwords)
    all_stopword = tf.equal(tf.size(non_stopword.out), 0)
    length = tf.equal(tf.size(span), 0)
    return tf.reduce_any([wordpiece_boundary, all_stopword, length])

  def sample(start, end):
    """Sample length from truncated Geo(p)."""
    # Sample from truncated geometric distribution.
    geometric = lambda k: (1 - p)**(k - 1) * p
    probs = np.array([geometric(k) for k in range(1, max_span_len + 1)])
    probs /= probs.sum()
    length = tf.distributions.Categorical(probs=probs).sample() + 1

    # Sample start uniformly.
    max_offset = tf.maximum(1, seq_len - length + 1)
    start = tf.random.uniform([], 0, max_offset, dtype=tf.int32)
    end = start + length

    # Return span.
    return [start, end]

  # Rejection sample. Start with dummy span variable.
  start = tf.constant(0)
  end = tf.constant(0)
  start, end = tf.while_loop(
      reject, sample, [start, end], maximum_iterations=max_iter)
  span = tf.strings.reduce_join(tokens[start:end], separator=" ")

  return span


# ------------------------------------------------------------------------------
#
# General purpose text functions for masking/unmasking.
#
# ------------------------------------------------------------------------------


def build_text_inputs(
    text,
    length,
    lookup_table,
    segment_id=0,
    start_token=None,
    end_token=None,
):
  """Convert text to TextInputs.

  Args:
    text: <string>, space-separated token string.
    length: Length to pad or truncate to.
    lookup_table: Instance of contrib.lookup.index_table_from_tensor.
    segment_id: Integer denoting segment type.
    start_token: Optional start token.
    end_token: Optional end token.

  Returns:
    Instance of TextInputs.
  """
  # Tokenize and truncate.
  tokens = tf.string_split([text]).values
  length_offset = sum([0 if i is None else 1 for i in [start_token, end_token]])
  tokens = tokens[:length - length_offset]
  if start_token is not None:
    tokens = tf.concat([[start_token], tokens], axis=0)
  if end_token is not None:
    tokens = tf.concat([tokens, [end_token]], axis=0)

  token_ids = tf.cast(lookup_table.lookup(tokens), tf.int32)
  mask = tf.ones_like(token_ids)
  segment_ids = tf.fill(tf.shape(token_ids), segment_id)

  pad = [[0, length - tf.size(token_ids)]]
  token_ids = tf.pad(token_ids, pad)
  mask = tf.pad(mask, pad)
  segment_ids = tf.pad(segment_ids, pad)
  positions = tf.range(length)
  text_input = TextInputs(
      token_ids=tf.ensure_shape(token_ids, [length]),
      mask=tf.ensure_shape(mask, [length]),
      segment_ids=tf.ensure_shape(segment_ids, [length]),
      positions=tf.ensure_shape(positions, [length]))

  return text_input


def build_planner_inputs(question, answer, length, lookup_table):
  """Convert text to TextInputs for conditional text planner.

  Args:
    question: <string>, space-separated token string.
    answer: <string>, space-separated token string.
    length: Length to pad or truncate to.
    lookup_table: Instance of contrib.lookup.index_table_from_tensor.

  Returns:
    Instance of TextInputs.
  """
  # Build question.
  q_tokens = tf.string_split([question]).values
  q_tokens = tf.concat([["[Q]"], q_tokens], axis=0)
  q_token_ids = tf.cast(lookup_table.lookup(q_tokens), tf.int32)
  q_len = tensor_utils.shape(q_token_ids, 0)
  q_positions = tf.range(q_len)

  # Build answer.
  a_tokens = tf.string_split([answer]).values
  a_tokens = tf.concat([["[A]"], a_tokens], axis=0)
  a_token_ids = tf.cast(lookup_table.lookup(a_tokens), tf.int32)
  a_len = tensor_utils.shape(a_token_ids, 0)
  a_positions = tf.range(a_len)

  # Combine.
  token_ids = tf.concat([q_token_ids, a_token_ids], axis=0)
  segment_ids = tf.concat([tf.fill([q_len], 2), tf.fill([a_len], 1)], axis=0)
  positions = tf.concat([q_positions, a_positions], axis=0)
  q_mask = tf.ones_like(q_token_ids)
  mask = tf.concat([q_mask, tf.ones_like(a_token_ids)], axis=0)

  # Truncate.
  token_ids = token_ids[:length]
  segment_ids = segment_ids[:length]
  mask = mask[:length]
  positions = positions[:length]

  # Pad.
  pad = [[0, length - tf.size(token_ids)]]
  token_ids = tf.pad(token_ids, pad)
  mask = tf.pad(mask, pad)
  segment_ids = tf.pad(segment_ids, pad)
  positions = tf.pad(positions, pad)

  text_input = TextInputs(
      token_ids=tf.ensure_shape(token_ids, [length]),
      mask=tf.ensure_shape(mask, [length]),
      segment_ids=tf.ensure_shape(segment_ids, [length]),
      positions=tf.ensure_shape(positions, [length]))

  return text_input

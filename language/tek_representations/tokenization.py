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
# coding=utf-8
"""Wrapper class for tokenization which maintains information about whole word boundaries.

Currently implements BERT and RoBERTa tokenization interfaces. See
get_tokenizer(.) for usage.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from gpt2 import encoder
from bert import tokenization
import regex as re
import six
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode('utf-8', 'ignore')
    else:
      raise ValueError('Unsupported string type: %s' % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode('utf-8', 'ignore')
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError('Unsupported string type: %s' % (type(text)))
  else:
    raise ValueError('Not running on Python2 or Python 3?')


def printable_text(text, strip_roberta_space=False):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  strip = (lambda x: (x.replace('Ġ', '') if x.startswith('Ġ') else x)
          ) if strip_roberta_space else (lambda x: x)
  if six.PY3:
    if isinstance(text, str):
      return strip(text)
    elif isinstance(text, bytes):
      return strip(text.decode('utf-8', 'ignore'))
    else:
      raise ValueError('Unsupported string type: %s' % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return strip(text)
    elif isinstance(text, unicode):
      return strip(text.encode('utf-8'))
    else:
      raise ValueError('Unsupported string type: %s' % (type(text)))
  else:
    raise ValueError('Not running on Python2 or Python 3?')


class Tokenizer(object):
  """Tokenizer interface which returns information about whole word boundaries."""

  def tokenize(self, text):
    raise NotImplementedError()

  def convert_tokens_to_ids(self, text):
    raise NotImplementedError()

  @property
  def vocab(self):
    raise NotImplementedError()

  @property
  def bos(self):
    raise NotImplementedError()

  @property
  def eos(self):
    raise NotImplementedError()

  @property
  def mask(self):
    raise NotImplementedError()


class RobertaTokenizer(Tokenizer):
  """Extends the RoBERTa tokenizer following the Tokenizer interface."""

  def __init__(self, models_dir):
    with tf.gfile.Open(os.path.join(models_dir, 'encoder.json'), 'r') as f:
      encoder_json = json.loads(f.read())
    with tf.gfile.Open(os.path.join(models_dir, 'vocab.bpe')) as f:
      bpe_data = f.read()
    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
    ]
    self.marker_tokens = [
        'madeupword0000', 'madeupword0001', 'madeupword0002', '*=-', 'ÃĽÃĽ',
        'ĠEntityItem', 'EngineDebug', 'ĠstrutConnector'
    ]
    self.bpe = encoder.Encoder(
        encoder=encoder_json,
        bpe_merges=bpe_merges,
    )

  @property
  def bos(self):
    return '<s>'

  @property
  def eos(self):
    return '</s>'

  @property
  def entity_separator(self):
    return ':'

  @property
  def mask(self):
    return '<mask>'

  def tokenize(self, text):
    bpe_tokens = []
    list_starts, str_starts = [], []
    basic_tokens = text if isinstance(text, list) else [text]
    for i, basic_token in enumerate(basic_tokens):
      num_subtokens = 0
      basic_token = basic_token if (i == 0 or not isinstance(text, list)) else (
          ' ' + basic_token)
      for token in re.findall(self.bpe.pat, basic_token):
        token = ''.join(self.bpe.byte_encoder[b] for b in token.encode('utf-8'))
        sub_tokens = [bpe_token for bpe_token in self.bpe.bpe(token).split(' ')]
        bpe_tokens.extend(sub_tokens)
        str_starts += [True] + [False] * (len(sub_tokens) - 1)
        num_subtokens += len(sub_tokens)
      list_starts += [True] + [False] * (num_subtokens - 1)
    word_starts = list_starts if isinstance(text, list) else str_starts
    assert len(bpe_tokens) == len(word_starts)
    return bpe_tokens, word_starts

  def convert_tokens_to_ids(self, tokens):
    return [self.bpe.encoder[token] for token in tokens]

  def tokenized_to_original(self, tok_tokens):
    text = ''.join(tok_tokens)
    text = bytearray([self.bpe.byte_decoder[c] for c in text]).decode(
        'utf-8', errors=self.bpe.errors)
    return text

  def get_marker_token(self, i):
    return self.marker_tokens[i % len(self.marker_tokens)]

  @property
  def vocab(self):
    return list(self.bpe.encoder.keys())


class BertTokenizer(Tokenizer):
  """Extends the BERT tokenizer following the Tokenizer interface."""

  def __init__(self, vocab_file, do_lower_case):
    self.marker_tokens = [
        '[unused0]', '[unused1]', '[unused2]', '[unused3]', '[unused4]',
        '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]',
        '[unused10]', '[unused11]', '[unused12]', '[unused13]', '[unused14]',
        '[unused15]', '[unused16]', '[unused17]', '[unused18]', '[unused19]'
    ]
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

  @property
  def bos(self):
    return '[CLS]'

  @property
  def eos(self):
    return '[SEP]'

  @property
  def entity_separator(self):
    return ':'

  @property
  def mask(self):
    return '[MASK]'

  def tokenize(self, text):
    basic_tokenizer = self.tokenizer.basic_tokenizer.tokenize if not isinstance(
        text, list) else (lambda x: x)
    split_tokens, word_starts = [], []
    for token in basic_tokenizer(text):
      subtokens = self.tokenizer.tokenize(token)
      if not subtokens:
        subtokens = ['[UNK]']
      word_starts += [True] + [False] * (len(subtokens) - 1)
      split_tokens.extend(subtokens)
    assert len(word_starts) == len(split_tokens)
    return split_tokens, word_starts

  def convert_tokens_to_ids(self, tokens):
    return self.tokenizer.convert_tokens_to_ids(tokens)

  def tokenized_to_original(self, tok_tokens):
    tok_text = ' '.join(tok_tokens)
    tok_text = tok_text.replace(' ##', '')
    tok_text = tok_text.replace('##', '')
    tok_text = tok_text.strip()
    tok_text = ' '.join(tok_text.split())
    return tok_text

  def get_marker_token(self, i):
    return self.marker_tokens[i % len(self.marker_tokens)]

  @property
  def vocab(self):
    return list(self.tokenizer.vocab.keys())


def get_tokenizer(tokenizer_type, file_or_dir, do_lower_case=True):
  tokenizer_type = tokenizer_type.lower()
  if tokenizer_type == 'bert':
    return BertTokenizer(file_or_dir, do_lower_case)
  elif tokenizer_type == 'roberta':
    return RobertaTokenizer(file_or_dir)
  else:
    raise NotImplementedError()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)

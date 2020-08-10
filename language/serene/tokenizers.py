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
"""Tokenizers for fever models."""
import re

from language.serene import util
import tensorflow_datasets as tfds
from tensorflow_models.official.nlp.bert import tokenization


class Tokenizer:
  """Class for all tokenizers to implement so that they share an API."""

  def tokenize(self, text):
    raise NotImplementedError()

  def save_to_file(self, filename_prefix):
    raise NotImplementedError()

  @classmethod
  def load_from_file(cls, filename_prefix):
    raise NotImplementedError()


class SpaceTokenizer(Tokenizer):
  """A simple whitespace tokenizer, useful in testing RegexTokenizer."""

  def __init__(self, lowercase = True):
    self._lowercase = lowercase

  def tokenize(self, text):
    if self._lowercase:
      text = text.lower()
    return text.split()

  def save_to_file(self, filename_prefix):
    util.write_json({'lowercase': self._lowercase},
                    filename_prefix + '.space_tokenizer')

  @classmethod
  def load_from_file(cls, filename_prefix):
    conf = util.read_json(filename_prefix + '.space_tokenizer')
    return cls(**conf)


class PassthroughTokenizer(Tokenizer):
  """Returns the text as a single token."""

  def tokenize(self, text):
    return [text]

  def save_to_file(self, filename_prefix):
    pass

  @classmethod
  def load_from_file(cls, filename_prefix):
    return cls()


class ReservedTokenizer(Tokenizer):
  """Tokenizer that tokenizes based on regex and given tokenizer.

  This is specifically designed to make it easier to tokenize reserved tokens
  or regexes and then tokenize the leftover text with another tokenizer.
  """

  def __init__(
      self,
      *,
      tokenizer,
      # This could be later improved to take a list of regex
      reserved_re):
    self._tokenizer: Tokenizer = tokenizer or PassthroughTokenizer()
    if reserved_re is None:
      self._reserved_re = None
      self._compiled_reserved_re = None
    else:
      self._reserved_re = reserved_re
      self._compiled_reserved_re = re.compile(reserved_re)

  def marked_tokenize(self, text):
    """Return text tokenized with indicators if each token is reserved.

    Args:
      text: Text to tokenize

    Returns:
      List of text tokens and list of true/false, true if it is a reserved token
    """
    if self._reserved_re is None:
      tokens = self._tokenizer.tokenize(text)
      return tokens, len(tokens) * [False]
    else:
      reserved_tokens = list(re.finditer(self._compiled_reserved_re, text))
      if reserved_tokens:
        all_tokens = []
        all_marks = []
        position = 0
        for match in reserved_tokens:
          start = match.start()
          end = match.end()
          token = text[start:end]
          # This assumes a space based tokenizer, may not want to strip if that
          # changes
          pre_text = text[position:start].strip()
          if pre_text:
            tokenized = self._tokenizer.tokenize(pre_text)
            all_marks.extend(len(tokenized) * [False])
            all_tokens.extend(tokenized)
          all_marks.append(True)
          all_tokens.append(token)
          position = end
        if position != len(text):
          # This assumes a space based tokenizer, may not want to strip if that
          # changes
          post_text = text[position:].strip()
          if post_text:
            tokenized = self._tokenizer.tokenize(post_text)
            all_marks.extend(len(tokenized) * [False])
            all_tokens.extend(tokenized)
        return all_tokens, all_marks
      else:
        tokens = self._tokenizer.tokenize(text)
        return tokens, len(tokens) * [False]

  def tokenize(self, text):
    tokens, _ = self.marked_tokenize(text)
    return tokens

  def save_to_file(self, filename_prefix):
    self._tokenizer.save_to_file(filename_prefix)
    util.write_json(
        {
            'reserved_re': self._reserved_re,
            'tokenizer_cls': self._tokenizer.__class__.__name__,
        }, filename_prefix + '.regex_tokenizer')

  @classmethod
  def load_from_file(cls, filename_prefix):
    conf = util.read_json(filename_prefix + '.regex_tokenizer')
    tokenizer_cls = conf['tokenizer_cls']
    tokenizer = tokenizer_registry[tokenizer_cls].load_from_file(
        filename_prefix)
    return cls(tokenizer=tokenizer, reserved_re=conf['reserved_re'])


# Using semi-custom implementation to change serialization behavior
# and make the vocab overridable for unit tests
class BertTokenizer(Tokenizer):
  """Runs end-to-end tokenization."""

  def __init__(self,
               vocab_file,
               do_lower_case = True,
               vocab_override = None):
    super().__init__()
    self.vocab_file = vocab_file
    self.do_lower_case = do_lower_case
    if vocab_override is None:
      self.vocab = tokenization.load_vocab(vocab_file)
    else:
      self.vocab = vocab_override
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = tokenization.BasicTokenizer(
        do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)

  def save_to_file(self, filename_prefix):
    vocab_file = f'{filename_prefix}.vocab'
    util.safe_copy(self.vocab_file, vocab_file)
    util.write_json(
        {
            'vocab_file': vocab_file,
            'do_lower_case': self.do_lower_case,
        }, f'{filename_prefix}.tokenizer')

  @classmethod
  def load_from_file(cls, filename_prefix):
    params = util.read_json(f'{filename_prefix}.tokenizer')
    bert_tokenizer = BertTokenizer(
        vocab_file=params['vocab_file'], do_lower_case=params['do_lower_case'])
    return bert_tokenizer


tokenizer_registry = {
    BertTokenizer.__name__: BertTokenizer,
    tfds.deprecated.text.Tokenizer.__name__: tfds.deprecated.text.Tokenizer,
    ReservedTokenizer.__name__: ReservedTokenizer,
    SpaceTokenizer.__name__: SpaceTokenizer,
    PassthroughTokenizer.__name__: PassthroughTokenizer
}

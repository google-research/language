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
r"""Preprocessor class that extract creates a database of text blocks.

Each input line should have the following JSON format:
```
{
  "title": "Document Tile",
  "text": "This is a full document."
}
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import six
import tensorflow.compat.v1 as tf


def add_int64_feature(key, values, example):
  example.features.feature[key].int64_list.value.extend(values)


class Preprocessor(object):
  """Preprocessor."""

  def __init__(self, sentence_splitter, max_block_length, tokenizer):
    self._tokenizer = tokenizer
    self._sentence_splitter = sentence_splitter
    self._max_block_length = max_block_length
    tf.logging.info("Max block length {}".format(self._max_block_length))

  def generate_sentences(self, title, text):
    """Generate sentences in each block from text."""
    title_length = len(self._tokenizer.tokenize(title))
    current_token_count = 0
    current_block_sentences = []
    for sentence in self._sentence_splitter.tokenize(text):
      num_tokens = len(self._tokenizer.tokenize(sentence))
      # Hypothetical sequence [CLS] <title> [SEP] <current> <next> [SEP].
      hypothetical_length = 3 + title_length + current_token_count + num_tokens
      if hypothetical_length <= self._max_block_length:
        current_token_count += num_tokens
        current_block_sentences.append(sentence)
      else:
        yield current_block_sentences
        current_token_count = num_tokens
        current_block_sentences = []
        current_block_sentences.append(sentence)
    if current_block_sentences:
      yield current_block_sentences

  def create_example(self, title, sentences):
    """Create example."""
    title_tokens = self._tokenizer.tokenize(title)
    title_ids = self._tokenizer.convert_tokens_to_ids(title_tokens)
    token_ids = []
    sentence_starts = []
    for sentence in sentences:
      sentence_starts.append(len(token_ids))
      sentence_tokens = self._tokenizer.tokenize(sentence)
      token_ids.extend(self._tokenizer.convert_tokens_to_ids(sentence_tokens))
    example = tf.train.Example()
    add_int64_feature("title_ids", title_ids, example)
    add_int64_feature("token_ids", token_ids, example)
    add_int64_feature("sentence_starts", sentence_starts, example)
    return example.SerializeToString()

  def generate_block_info(self, title, text):
    for sentences in self.generate_sentences(title, text):
      if sentences:
        block = " ".join(sentences)
        example = self.create_example(title, sentences)
        yield title, block, example


def remove_doc(title):
  return re.match(r"(List of .+)|"
                  r"(Index of .+)|"
                  r"(Outline of .+)|"
                  r"(.*\(disambiguation\).*)", title)


def example_from_json_line(line, html_parser, preprocessor):
  if not isinstance(line, six.text_type):
    line = line.decode("utf-8")
  data = json.loads(line)
  title = data["title"]
  if not remove_doc(title):
    text = html_parser.unescape(data["text"])
    for info in preprocessor.generate_block_info(title, text):
      yield info

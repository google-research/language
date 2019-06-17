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
"""Provides a `Tokenizer` classes that can break strings into tokens."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from language.boolq.utils import py_utils
import nltk

# In case there is an issue with using nltk's resource locator,
# allow the path to be manually specified
flags.DEFINE_string("punkt_tokenizer_file", None,
                    "Location of pickeled punkt sentence tokenizer to use, if "
                    "None load the tokenizer using nltk")

FLAGS = flags.FLAGS


class NltkTokenizer(object):
  """Tokenizer that uses `nltk`."""

  def __init__(self):
    self._word_tokenizer = nltk.TreebankWordTokenizer()
    if FLAGS.punkt_tokenizer_file is not None:
      self._sent_tokenizer = py_utils.load_pickle(FLAGS.punkt_tokenizer_file)
    else:
      self._sent_tokenizer = nltk.load("tokenizers/punkt/english.pickle")

  def tokenize(self, text):
    """Turn an english natural language string into tokens.

    Args:
      text: to transform

    Returns:
      tokens: list of string tokens
    """
    return py_utils.flatten_list(self._word_tokenizer.tokenize(s)
                                 for s in self._sent_tokenizer.tokenize(text))

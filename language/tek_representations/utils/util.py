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
"""Preprocessing utils.

Methods for sentence splitting, ngrams, and other preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq

from absl import app
from absl import flags


FLAGS = flags.FLAGS

stopwords = set([
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there',
    'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own',
    'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of',
    'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
    'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
    'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her',
    'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
    'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when',
    'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will',
    'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
    'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself',
    'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i',
    'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a',
    'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'
])


def get_corpus(corpus_file):
  """Load corpus."""
  corpus = None
  if corpus is None:
    raise NotImplementedError('Key-value store not implemented.')
  return corpus


class SentenceScore(object):
  """Wrapper around sentence and it's score."""

  def __init__(self, sentence, score):
    self.sentence = sentence
    self.score = score

  def __lt__(self, other):
    return self.score < other.score


class TopKMaxList(object):
  """Heap Wrapper."""

  def __init__(self, size):
    self.size = size
    self.heap = []

  def append(self, key, value):
    if len(self.heap) < self.size:
      heapq.heappush(self.heap, SentenceScore(key, value))
    else:
      heapq.heappushpop(self.heap, SentenceScore(key, value))

  def get_top(self):
    sent_scores = sorted(self.heap, reverse=True)
    return [(ss.sentence, ss.score) for ss in sent_scores]


def get_ngrams(tokens, n, lower_case=True, remove_stopwords=True):
  ngrams = []
  maybe_lower = (lambda x: x.lower()) if lower_case else (lambda x: x)
  tokens = tokens if not remove_stopwords else (
      [t for t in tokens if t not in stopwords])
  for i in range(1, n + 1):
    ngrams += [
        maybe_lower(' '.join(tokens[j:j + i])) for j in range(len(tokens))
    ]
  return ngrams


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)

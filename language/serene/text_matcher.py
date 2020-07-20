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
"""A simple TFIDF text matcher and function to run it."""

import pickle
import random


from absl import logging
from language.serene import fever_pb2
from language.serene import types
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow.compat.v2 as tf
import tqdm



class TextMatcher:
  """A simple TFIDF Text matcher."""

  def __init__(
      self,
      ngram_range = (1, 2), min_df=2, max_df=.9):
    """Init parameters for text matcher.

    For details, refer to
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    since the parameters are used in calling that.

    Args:
      ngram_range: Tuple of n-grams to use (e.g., unigram and bigram)
      min_df: Max allowed term frequency before excluding from vocab
      max_df: Min required term frequency before excluding from vocab
    """
    self._tfidf: Optional[TfidfVectorizer] = None
    self._ngram_range = ngram_range
    self._min_df = min_df
    self._max_df = max_df

  def train(self, sentences):
    self._tfidf = TfidfVectorizer(
        ngram_range=self._ngram_range,
        min_df=self._min_df, max_df=self._max_df)
    self._tfidf.fit(sentences)

  def score(self,
            claim,
            candidates,
            text_key = 'text'):
    """Return the score for each candidate, order does not change.

    Args:
      claim: The claim to match
      candidates: The candidates to rank
      text_key: Key in the candidate json that contains the text to score

    Returns:
      The score for each candidate
    """
    if self._tfidf is None:
      raise ValueError('You must train or load a model before predicting')
    if not candidates:
      return []
    # make candidates indexable via numpy style indices
    candidates = np.array(candidates, dtype=np.object)
    # (1, vocab_size)
    claim_repr = self._tfidf.transform([claim])
    # (n_candidates, vocab_size)
    candidates_repr = self._tfidf.transform([c[text_key] for c in candidates])
    # (1, n_candidates)
    product = candidates_repr.dot(claim_repr.T).T.toarray()
    return product.reshape(-1).tolist()

  def predict(
      self,
      claim, candidates,
      text_key = 'text'):
    """Scores claim against candidates and returns ordered candidates.

    Args:
      claim: The claim to match
      candidates: The candidates to rank
      text_key: Key in the candidate json that contains the text to score

    Returns:
      sorted candidates and a score for each.
    """
    if self._tfidf is None:
      raise ValueError('You must train or load a model before predicting')
    if not candidates:
      return []
    # make candidates indexable via numpy style indices
    candidates = np.array(candidates, dtype=np.object)
    # (1, vocab_size)
    claim_repr = self._tfidf.transform([claim])
    # (n_candidates, vocab_size)
    candidates_repr = self._tfidf.transform([c[text_key] for c in candidates])
    # (1, n_candidates)
    product = candidates_repr.dot(claim_repr.T).T.toarray()
    # Take the first row, since that is the only row and the one that
    # contains the scores against the claim
    preds = (-product).argsort(axis=1)[0]
    scores = -np.sort(-product, axis=1)[0]
    scores_and_candidates = []
    for match_score, candidate in zip(scores, candidates[preds]):
      scores_and_candidates.append((match_score, candidate))
    return scores_and_candidates

  def save(self, data_dir):
    if self._tfidf is None:
      raise ValueError('Attempted to save nonexistent model')
    with tf.io.gfile.GFile(data_dir, 'wb') as f:
      pickle.dump({
          'tfidf': self._tfidf,
          'ngram_range': self._ngram_range,
          'min_df': self._min_df,
          'max_df': self._max_df,
      }, f)

  def load(self, data_dir):
    with tf.io.gfile.GFile(data_dir, 'rb') as f:
      params = pickle.load(f)
      self._tfidf = params['tfidf']
      self._ngram_range = params['ngram_range']
      self._min_df = params['min_df']
      self._max_df = params['max_df']




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
"""BM25 retriever implementation."""

import gensim.summarization.bm25
import nltk
import numpy as np


class BM25Retriever(object):
  """Retriever based on BM25."""

  def __init__(self, documents):
    self.documents = documents
    # Map from document title to idx.
    self.idx_map = {}
    self.bm25 = self._init_bm25()

  def _init_bm25(self):
    """Initialize BM25 retriever."""
    samples_for_retrieval_tokenized = []
    for idx, document in enumerate(self.documents):
      tokenized_example = nltk.tokenize.word_tokenize(document.text)
      samples_for_retrieval_tokenized.append(tokenized_example)
      self.idx_map[document.title] = idx
    return gensim.summarization.bm25.BM25(samples_for_retrieval_tokenized)

  def _compute_scores(self, query):
    tokenized_query = nltk.tokenize.word_tokenize(query)
    bm25_scores = self.bm25.get_scores(tokenized_query)
    scores = []
    for idx in range(len(self.documents)):
      scores.append(-bm25_scores[idx])
    return np.array(scores)

  def get_docs_and_scores(self, query, topk=100):
    """Retrieve documents based on BM25 scores.

    Args:
      query: The query to retrieve documents for.
      topk: Return top-k document ids.

    Returns:
      A List of (document title, score) tuples of length `topk`.
    """
    scores = self._compute_scores(query)
    sorted_docs_ids = np.argsort(scores)
    topk_doc_ids = sorted_docs_ids[:topk]
    return [(self.documents[idx].title, scores[idx]) for idx in topk_doc_ids]

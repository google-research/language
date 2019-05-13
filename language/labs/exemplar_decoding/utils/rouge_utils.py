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
"""ROUGE metric implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial  # pylint: disable=g-importing-member
from language.labs.exemplar_decoding.utils.data import id2text
from language.labs.exemplar_decoding.utils.data import SPECIAL_TOKENS
import numpy as np
import tensorflow as tf


def _safe_divide(x, y):
  if y == 0:
    return 0.0
  else:
    return x /y


def _len_lcs(x, y):
  """Returns the length of the Longest Common Subsequence between two seqs.

  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """Computes the length of the LCS between two seqs.

  The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _f_lcs(llcs, m, n):
  """Computes the LCS-based F-measure score.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = _safe_divide(llcs, m)
  p_lcs = _safe_divide(llcs, n)
  beta = _safe_divide(p_lcs, r_lcs)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = _safe_divide(num, denom)
  return f_lcs


def _get_ngrams(n, text):
  """Calculates n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def rouge_l(eval_sentences, eval_len, ref_sentences, ref_len,
            vocab, use_bpe=False):
  """Rouge-L.

  Args:
    eval_sentences: prediction to be evaluated.
    eval_len: lengths of the predictions.
    ref_sentences: reference sentences.
    ref_len: lengths of the references.
    vocab: vocabulary.
    use_bpe: to use BPE or not.

  Returns:
    Rouge-L
  """
  f1_scores = []
  for e, el, r, rl in zip(eval_sentences, eval_len, ref_sentences, ref_len):
    e = id2text(e[:el], vocab=vocab, use_bpe=use_bpe).split()
    r = r[:rl]
    r = [x for x in r if x not in SPECIAL_TOKENS]
    lcs = _len_lcs(e, r)
    f1_scores.append(_f_lcs(lcs, len(r), len(e)))
  return np.mean(f1_scores, dtype=np.float32)


def rouge_n(eval_sentences, eval_len, ref_sentences, ref_len, n,
            vocab, use_bpe=False, predict_mode=False):
  """Rouge N."""
  f1_scores = []
  for e, el, r, rl in zip(eval_sentences, eval_len, ref_sentences, ref_len):
    e = id2text(e[:el], vocab=vocab, use_bpe=use_bpe).split()
    r = r[:rl]
    e = [x for x in e if x not in SPECIAL_TOKENS]
    r = [x for x in r if x not in SPECIAL_TOKENS]

    if n == 1 and predict_mode:
      tf.logging.info("prediction: %s", " ".join(e))
      tf.logging.info("reference: %s", " ".join(r))

    eval_ngrams = _get_ngrams(n, e)
    ref_ngrams = _get_ngrams(n, r)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    precision = _safe_divide(overlapping_count, eval_count)
    recall = _safe_divide(overlapping_count, ref_count)
    f1_scores.append(
        _safe_divide(2 * precision * recall, precision + recall))
  return np.mean(f1_scores, dtype=np.float32)


def rouge_n_metric(predictions, prediction_len, labels, label_len, n,
                   vocab, use_bpe=False, predict_mode=False):
  return tf.metrics.mean(tf.py_func(
      partial(rouge_n, n=n, vocab=vocab, use_bpe=use_bpe,
              predict_mode=predict_mode),
      [predictions, prediction_len, labels, label_len],
      tf.float32))


def rouge_l_metric(predictions, prediction_len, labels, label_len,
                   vocab, use_bpe=False):
  return tf.metrics.mean(tf.py_func(
      partial(rouge_l, vocab=vocab, use_bpe=use_bpe),
      [predictions, prediction_len, labels, label_len],
      tf.float32))


def get_metrics(predictions, prediction_len, labels, label_len,
                vocab, use_bpe=False, predict_mode=False):
  """Rouge-metrics.

  Args:
    predictions: prediction to be evaluated.
    prediction_len: lengths of the predictions.
    labels: reference sentences.
    label_len: lengths of the references.
    vocab: vocabulary.
    use_bpe: to use BPE or not.
    predict_mode: set to true to print predictions.

  Returns:
    Rouge-metrics
  """
  rouge_1_m = rouge_n_metric(
      predictions, prediction_len, labels, label_len, 1,
      vocab=vocab, use_bpe=use_bpe, predict_mode=predict_mode)
  rouge_2_m = rouge_n_metric(
      predictions, prediction_len, labels, label_len, 2,
      vocab=vocab, use_bpe=use_bpe)
  rouge_l_m = rouge_l_metric(
      predictions, prediction_len, labels, label_len,
      vocab=vocab, use_bpe=use_bpe)
  return {
      "rouge_1": rouge_1_m,
      "rouge_2": rouge_2_m,
      "rouge_l": rouge_l_m
  }

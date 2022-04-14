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
"""Utilities for NQ."""

import collections
import io
import pickle
import re
import string
from typing import List, Dict, Callable, Sequence, Tuple

from absl import logging
import attr
from language.search_agents import environment_pb2
import numpy as np
import tensorflow as tf

REMOVE_ARTICLES_REGEX = re.compile(r'\b(a|an|the)\b', re.UNICODE)
EXCLUDE = frozenset(string.punctuation)


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(REMOVE_ARTICLES_REGEX, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    return ''.join(ch for ch in text if ch not in EXCLUDE)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1_single(prediction, ground_truth):
  """Computes F1 score for a single prediction/ground_truth pair.

  This function computes the token-level intersection between the predicted and
  ground-truth answers, consistent with the SQuAD evaluation script.

  This is different from another common way of computing F1 (e.g. used by BiDAF)
  that uses the intersection between predicted answer span and ground-truth
  spans. The main difference is that this method gives credit to correct partial
  answers that don't match any full answer span, while the other one wouldn't.

  Args:
    prediction: predicted string.
    ground_truth: ground truth string.

  Returns:
    Token-wise F1 score between the two input strings.
  """
  # Handle empty answers for Squad 2.0.
  if not ground_truth and not prediction:
    return 1.0

  prediction_tokens = prediction.split()
  ground_truth_tokens = ground_truth.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0
  precision = num_same / len(prediction_tokens)
  recall = num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def compute_f1(prediction, gold_answers):
  """Computes F1 score for a single prediction and a list of gold answers.

  See 'compute_f1_single' for details. Expects all input strings to be unicode.

  Args:
    prediction: predicted string.
    gold_answers: a list of ground truth strings.

  Returns:
    Maximum of the token-wise F1 score between the prediction and each gold
    answer.
  """
  if not gold_answers:
    return 0.0
  else:
    f1_scores = []
    prediction = normalize_answer(prediction)
    for answer in gold_answers:
      f1_scores.append(compute_f1_single(prediction, normalize_answer(answer)))

    return max(f1_scores)


def precision(document_list: Sequence[environment_pb2.Document],
              gold_answer: Sequence[str]) -> float:
  """Computes the precision of a document set.

  Args:
    document_list: The list of document over which to compute the precision.
    gold_answer: The list of gold answers.

  Returns:
    The precision of the document sequence. A document is a 'hit' if it
    *contains* one of the gold answers in its content.
  """
  return np.mean([
      gold_answer_present(doc.content, list(gold_answer))
      for doc in document_list
  ])


def dcg_score(relevances: List[float]) -> float:
  """DCG score computation.

  Args:
    relevances: List[float], The list of relevance scores.

  Returns:
    The discounted cumulative gain for k == len(relevances).
  """
  return float(
      sum([
          relevance / np.log2(i + 2) for i, relevance in enumerate(relevances)
      ]))


def ndcg_score(relevances: List[float]) -> float:
  """nDCG score computation.

  We normalize using the "ideal" result set which has only relevant results.
  This scales the DCG scores for any number of results k into the interval
  [0, 1].

  Args:
    relevances: List[float], The list of relevance scores.

  Returns:
    The normalized discounted cumulative gain for k == len(relevances).
  """
  if not relevances:
    return 0.0
  return dcg_score(relevances) / dcg_score([1.0] * len(relevances))


def mrr_score(relevances: List[float], hit_val=1.0) -> float:
  """MRR score computation.

  Args:
    relevances: List[float], The list of relevance scores.
    hit_val: float, The value from which a relevance score is considered a
      'hit'.

  Returns:
    The mean reciprocal rank of the first time the hit_val has been reached in
    the relevances.
  """
  relevances = np.array(relevances)
  if not any(relevances >= hit_val):
    return 0.
  return 1. / (np.argmax(relevances >= hit_val) + 1)


def gold_answer_present(passage: str, gold_answer: Sequence[str]) -> bool:
  normalized_passage = normalize_answer(passage)
  return any([
      normalize_answer(answer) in normalized_passage for answer in gold_answer
  ])


def is_correct_answer(answer: str, gold_answer: List[str]) -> bool:
  gold_answer = [normalize_answer(a) for a in gold_answer]
  return normalize_answer(answer) in gold_answer


def compute_em(answer: str, gold_answer: List[str]) -> float:
  return float(
      normalize_answer(answer) in [normalize_answer(ga) for ga in gold_answer])


def bert_tokens_to_words(bert_tokens: List[str]) -> List[str]:
  """Maps word-pieces to the full word they are a part of.

  Given a word-piece tokenized sequence, returns a sequence of the same length
  which, at position i, contains the full word to which the word piece at
  position i belongs.
  For example:
    input:   wh   ##y  do   ##es  it  do  th   ##is
    output:  why  why  does does  it  do  this this

  Args:
    bert_tokens:  A sequence of word pieces.

  Returns:
    A sequence of equal length as `bert_tokens` with the full word to which each
    word-piece in `bert_tokens` belongs.
  """

  words = []
  i = 0
  while i < len(bert_tokens):
    full_word = bert_tokens[i]
    num_subwords = 1
    while i < (len(bert_tokens) - 1) and bert_tokens[i + 1].startswith('##'):
      full_word += bert_tokens[i + 1].replace('##', '')
      i += 1
      num_subwords += 1
    i += 1
    words += [full_word] * num_subwords
  return words


def load_pickle_from_file(path: str):
  with tf.io.gfile.GFile(path, 'rb') as f:
    bytes_object = f.read()
  bytes_io = io.BytesIO(bytes_object)
  return pickle.Unpickler(bytes_io, encoding='latin-1').load()


class IDFLookup:
  """Singleton for the idflookup table loaded from cns."""

  _instance = None
  _MAX_IDF = 17.407

  @staticmethod
  def get_instance(path: str):
    if IDFLookup._instance is None:
      IDFLookup(path)
    return IDFLookup._instance

  def __init__(self, path: str):
    if IDFLookup._instance:
      raise Exception('This class is a singleton!')

    IDFLookup._instance = self

    if not path or not tf.io.gfile.exists(path):
      logging.warning(
          'Path {} is not a valid CNS path. IDF lookup will be initialized as empty defaultdict.'
      )
      self.lookup = collections.defaultdict(float)

    else:
      logging.info('Loading idf lookup from %s', path)
      if path.endswith('.pickle'):
        self.lookup = load_pickle_from_file(path)
      else:
        raise KeyError('Wrong path type for the idf lookup table.')

      # make it defaultdict with max number as default
      self.lookup = collections.defaultdict(lambda: self._MAX_IDF, self.lookup)

  def __getitem__(self, key):
    return self.lookup[key]


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Text:
  tokens: List[str]

  def expand(self, tokens_to_id_fn):
    return tokens_to_id_fn(self.tokens)

  def __str__(self):
    return f'[{" ".join(self.tokens)}]'


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class ObsFragment:
  """A piece of the state presentation, comprising text and annotations."""

  text: Text
  type_values: Dict[str, List[str]]
  float_values: Dict[str, List[float]]

  def __str__(self):
    return '\n'.join(
        [str(self.text),
         str(self.type_values),
         str(self.float_values)])

  def token_list(self) -> List[str]:
    return self.text.tokens[:]

  def type_lists(self) -> List[List[str]]:
    types = []
    for _, v in sorted(self.type_values.items()):
      types.append(v[:])
    return types

  def feature_lists(self) -> List[List[float]]:
    features = []
    for _, v in sorted(self.float_values.items()):
      features.append(v[:])
    return features

  def expand(
      self, type_vocabs: collections.OrderedDict, float_names: List[str],
      tokens_to_id_fn: Callable[[List[str]], List[int]]
  ) -> Tuple[List[int], List[List[int]], List[List[float]]]:
    """Turns this fragment into encoder-compatible inputs.

    - Maps the `text` tokens to integer ids, also expanding contextual
    pre-terminals such as `Vap0` by way of `type_vocabs`.
    - Maps the `type_values` to integer ids and assembles the different kinds
      into a list.
    - Assembles the `float_values` into a list.

    Args:
      type_vocabs:  Segment types and their valid values.
      float_names:  Name of the float features to expand.
      tokens_to_id_fn:  Performs the mapping from text to integer ids.

    Returns:
      A list with three elements, [text_ids, type_ids, float_values].
      Note that `type_ids` and `float_values` themselves are lists of lists,
      whereas `text_ids` is simply a list of ints.
    """

    text_ids = self.text.expand(tokens_to_id_fn)

    all_type_values = []
    for type_name, type_vocab in type_vocabs.items():
      type_values = self.type_values[type_name]
      if len(type_values) != len(text_ids):
        raise ValueError(
            'Type values for {} ({}) must match length of {}'.format(
                type_name, type_values, str(self.text)))
      all_type_values.append([type_vocab.index(t) for t in type_values])

    all_float_values = []
    for fname in float_names:
      fvalues = self.float_values[fname]
      if len(text_ids) != len(fvalues):
        raise ValueError(
            'Float value for {} ({}) must match length of {}'.format(
                fname, fvalues, str(self.text)))
      all_float_values.append(fvalues)

    return (text_ids, all_type_values, all_float_values)

  @staticmethod
  def combine_and_expand(fragments: List['ObsFragment'], length: int,
                         **expand_args):
    """Expands each element of `fragments` and returns encoder input lists."""

    all_token_ids = []
    all_type_ids = []  # type: List[List[int]]
    all_float_features = []  # type: List[List[float]]

    for fragment in fragments:
      (token_ids, type_ids, float_features) = fragment.expand(**expand_args)
      all_token_ids.extend(token_ids)

      if not all_type_ids:
        all_type_ids = type_ids
      else:
        for old_type_ids, new_type_ids in zip(all_type_ids, type_ids):
          old_type_ids.extend(new_type_ids)

      if not all_float_features:
        all_float_features = float_features
      else:
        for old_features, new_features in zip(all_float_features,
                                              float_features):
          old_features.extend(new_features)

    if len(all_token_ids) > length:  # Truncate to `length`.
      all_token_ids = all_token_ids[:length]
      all_type_ids = [type_ids[:length] for type_ids in all_type_ids]
      all_float_features = [
          features[:length] for features in all_float_features
      ]

    padding_length = length - len(all_token_ids)
    if padding_length > 0:
      int_padding = [0] * padding_length
      float_padding = [0.0] * padding_length
      all_token_ids.extend(int_padding)
      all_type_ids = [type_ids + int_padding for type_ids in all_type_ids]
      all_float_features = [
          features + float_padding for features in all_float_features
      ]

    return all_token_ids, all_type_ids, all_float_features


def escape_for_lucene(query: str) -> str:
  """Reimplementation of QueryParser.escape.

  See
  https://github.com/apache/lucene-solr/blob/d894a7e8d75967fd0574bc1c98860b6062c21773/lucene/queryparser/src/java/org/apache/lucene/queryparser/classic/QueryParserBase.java#L971

  Args:
    query:  Query string.

  Returns:
    Properly escaped version of query.
  """
  result_buffer = []
  for c in query:
    # These characters are part of the query syntax and must be escaped.
    if (c == '\\' or c == '+' or c == '-' or c == '!' or c == '(' or c == ')' or
        c == ':' or c == '^' or c == '[' or c == ']' or c == '\"' or c == '{' or
        c == '}' or c == '~' or c == '*' or c == '?' or c == '|' or c == '&' or
        c == '/'):
      result_buffer.append('\\')
    result_buffer.append(c)

  return ''.join(result_buffer)

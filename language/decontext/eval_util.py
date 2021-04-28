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
"""Helper functions for evaluate decontextualization."""

import collections
import re
import string

from absl import logging
import nltk
import numpy as np

RawSARIScore = collections.namedtuple(
    'Raw_SARI_Score', 'add_tp, add_fp, add_fn, del_tp, del_fp, del_fn')


def get_p_r_f1(tp, fp, fn):
  if tp == 0:
    return 0, 0, 0
  else:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


def get_avg(l):
  if isinstance(l[0], bool):
    l = [int(item) for item in l]
  return sum(l) * 1.0 / len(l)


def normalize_text(input_text):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r'\b(a|an|the)\b', ' ', s)

  def replace_punctuation(s):
    to_replace = set(string.punctuation)
    return ''.join('' if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return ' '.join(s.split())

  text = input_text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)
  return text


def get_sent(pred):
  if pred.category == 'DONE':
    return pred.decontextualized_sentence
  else:
    return pred.original_sentence


def get_sent_dict(prediction_dict):
  """"Get decontextualized sentences from a decontext label."""
  pred_sent_dict = {}
  for ex_id in prediction_dict:
    pred_sent_dict[ex_id] = get_sent(prediction_dict[ex_id])
  return pred_sent_dict


def get_ngram_counter(ids, n):
  """Get a Counter with the ngrams of the given ID list.

  Args:
    ids: np.array or a list corresponding to a single sentence
    n: n-gram size

  Returns:
    collections.Counter with ID tuples as keys and 1s as values.
  """
  ngram_list = [tuple(ids[i:i + n]) for i in range(len(ids) + 1 - n)]
  ngrams = set(ngram_list)
  counts = collections.Counter()
  for ngram in ngrams:
    counts[ngram] = 1
  return counts


def get_raw_sari_score(source_ids, prediction_ids, list_of_targets):
  """Compute the SARI score for a single prediction and one or more targets.

  Args:
    source_ids: a list / np.array of SentencePiece IDs
    prediction_ids: a list / np.array of SentencePiece IDs
    list_of_targets: a list of target ID lists / np.arrays

  Returns:
    the SARI add and deletion raw counts
    (true positive addition tokens, false positive addition tokens,
    false negative addition tokens, true positive deletion tokens,
    false positive deletion tokens, false negative deletion tokens).
  """
  source_counts = get_ngram_counter(source_ids, 1)
  prediction_counts = get_ngram_counter(prediction_ids, 1)
  # All ngrams in the targets with count r/num_targets, where r is the number
  # of targets where the ngram occurs.
  weighted_target_counts = collections.Counter()
  num_nonempty_targets = 0
  for target_ids_i in list_of_targets:
    target_counts_i = get_ngram_counter(target_ids_i, 1)
    if target_counts_i:
      weighted_target_counts += target_counts_i
      num_nonempty_targets += 1

  for gram in weighted_target_counts.keys():
    weighted_target_counts[gram] /= num_nonempty_targets

  # DEL.
  source_not_prediction_counts = source_counts - prediction_counts
  source_not_target_counts = source_counts - weighted_target_counts
  del_tp = sum(
      (source_not_prediction_counts & source_not_target_counts).values())
  del_fp = sum(source_not_prediction_counts.values()) - del_tp
  del_fn = sum(source_not_target_counts.values()) - del_tp

  # ADD.
  added_to_prediction_counts = prediction_counts - source_counts
  add_tp = sum((added_to_prediction_counts & weighted_target_counts).values())
  add_fp = sum(added_to_prediction_counts.values()) - add_tp
  add_fn = sum((weighted_target_counts - source_counts).values()) - add_tp

  return (add_tp, add_fp, add_fn, del_tp, del_fp, del_fn)


def compute_raw_sari_score(original, pred, references):
  """Compute SARI score.

    The score is introduced in the following paper to measure
    sentence similarity.

    Optimizing Statistical Machine Translation for Text
    Simplification Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and
    Chris Callison-Burch In Transactions of the Association for Computational
    Linguistics (TACL) 2016
    http://cs.jhu.edu/~napoles/res/tacl2016-optimizing.pdf
    This basically compares the overlap between the original sentence,
    reference sentences, and the predicted sentence.

    This score only considers unigrams, and compute raw true positive,
    false positive, and false negative counts instead of computing
    precision, recall, f1 per example.

  Args:
    original: the original sentence.
    pred: predicted sentence.
    references: a list of reference sentences.

  Returns:
    SARIScore.
    It has three sub components: add, keep, and deletion scores.
  """
  # first tokenize.
  if not references:
    raise ValueError('Annotation cannot be empty.')
  # Tokenize.
  tokenized_original = nltk.tokenize.word_tokenize(original)
  tokenized_prediction = nltk.tokenize.word_tokenize(pred)
  tokenized_references = [
      nltk.tokenize.word_tokenize(annot) for annot in references
  ]
  # Create vocab.
  vocab = set(tokenized_prediction + tokenized_original)
  for tokenized_reference in tokenized_references:
    vocab.update(tokenized_reference)
  vocab_dict = {v: i for i, v in enumerate(vocab)}
  # Map to id.
  source_ids = np.array([vocab_dict[t] for t in tokenized_original])
  pred_ids = np.array([vocab_dict[t] for t in tokenized_prediction])
  reference_ids_list = []
  for reference in tokenized_references:
    reference_ids_list.append(np.array([vocab_dict[t] for t in reference]))
  # Compute the score.
  (add_tp, add_fp, add_fn, del_tp, del_fp, del_fn) = (
      get_raw_sari_score(source_ids, pred_ids, reference_ids_list))
  return RawSARIScore(add_tp, add_fp, add_fn, del_tp, del_fp, del_fn)


def process_annotation(annotation_dict, allow_single_annotations=False):
  """Process annotation dictionary to generate references.

  Not all annotations will be a reference set to which prediction will be
  evaluated against. The median annotation will reserved to approximate human
  performance. In case there's only one reference annotation, the
  `allow_single_annotations` argument determines whether it is reserved.

  The output dictionaries will only contain examples where at least half the
  annotators found decontextualization is feasible.

  Args:
    annotation_dict: dictionary with a list of decontext labels.
    allow_single_annotations: bool, when there's only a single annotation,
      whether to pass that single annotation to the reference_sents_dict.

  Returns:
    For all dictionaries, key is the example id.

    original_sent_dict: value is the original sentence.
    reference_sents_dict: value is a list of reference decontextualized
    sentences.
    median_annotation_dict: key is an example id,
      value is the median length human annotation,
      considered to approximate human references.

  """
  reference_sents_dict = {}
  median_annotation_dict = {}
  original_sent_dict = {}

  for ex_id in list(annotation_dict.keys()):
    annotations = annotation_dict[ex_id]
    if not annotations:
      raise ValueError('Empty annotations encountered while processing '
                       'reference annotations.')
    not_impossible = [
        get_sent(label)
        for label in annotations
        if label.category != 'IMPOSSIBLE'
    ]
    not_impossible = sorted(not_impossible, key=len)
    if 2 * len(not_impossible) < len(annotations):
      # Skip the example if the majority of annotators marked it as IMPOSSIBLE.
      continue
    median_val = (len(not_impossible) - 1) // 2
    if len(not_impossible) == 1 and allow_single_annotations:
      median_annotation_dict[ex_id] = not_impossible[median_val]
    else:
      median_annotation_dict[ex_id] = not_impossible.pop(median_val)
    reference_sents_dict[ex_id] = not_impossible
    original_sent_dict[ex_id] = annotations[0].original_sentence
  logging.info('%d examples needs to be decontextualized out of %d',
               len(median_annotation_dict), len(annotation_dict))
  return original_sent_dict, reference_sents_dict, median_annotation_dict


def compute_sentence_generation_scores(original_sent_dict, reference_dict,
                                       pred_dict):
  """Compute the decontextualization performances."""

  per_i_match_scores = []
  per_i_match_only_changed_scores = []

  per_i_changed = []
  per_i_raw_length_ratios = []

  per_i_raw_sari_add_tps = []
  per_i_raw_sari_del_tps = []
  per_i_raw_sari_add_fps = []
  per_i_raw_sari_del_fps = []
  per_i_raw_sari_add_fns = []
  per_i_raw_sari_del_fns = []

  for ex_id in list(reference_dict.keys()):
    normalized_others = [
        normalize_text(annot) for annot in reference_dict[ex_id]
    ]
    orig_sent = original_sent_dict[ex_id]
    model_pred_sent = pred_dict[ex_id]

    normalized_orig_sent = normalize_text(orig_sent)
    normalized_pred_sent = normalize_text(model_pred_sent)
    is_changed = normalized_pred_sent != normalized_orig_sent
    is_matched = normalized_pred_sent in normalized_others

    per_i_changed.append(is_changed)
    per_i_raw_length_ratios.append(len(model_pred_sent) / len(orig_sent))

    per_i_match_scores.append(is_matched)

    if normalized_orig_sent not in normalized_others:
      per_i_match_only_changed_scores.append(is_matched)

    raw_sari = compute_raw_sari_score(normalized_orig_sent,
                                      normalized_pred_sent, normalized_others)

    per_i_raw_sari_add_fns.append(raw_sari.add_fn)
    per_i_raw_sari_del_fns.append(raw_sari.del_fn)
    per_i_raw_sari_add_tps.append(raw_sari.add_tp)
    per_i_raw_sari_del_tps.append(raw_sari.del_tp)
    per_i_raw_sari_add_fps.append(raw_sari.add_fp)
    per_i_raw_sari_del_fps.append(raw_sari.del_fp)

  logging.info('-' * 100)
  logging.info('N=%d', len(per_i_raw_length_ratios))
  logging.info(
      'Avg. Length Increase Ratio (length of decontextualized '
      'sentence divided by the original sentence):%.2f',
      (get_avg(per_i_raw_length_ratios)))
  logging.info('Avg. Change Ratio (percent examples modified): %.1f',
               (get_avg(per_i_changed) * 100))
  logging.info('Sentence Match Score: %.1f',
               (get_avg(per_i_match_scores) * 100))

  logging.info('Sentence Match on Changed Examples: %.1f %d',
               get_avg(per_i_match_only_changed_scores) * 100,
               len(per_i_match_only_changed_scores))
  (sari_add_tps, sari_add_fns, sari_add_fps, sari_del_tps, sari_del_fns,
   sari_del_fps) = (get_avg(per_i_raw_sari_add_tps),
                    get_avg(per_i_raw_sari_add_fns),
                    get_avg(per_i_raw_sari_add_fps),
                    get_avg(per_i_raw_sari_del_tps),
                    get_avg(per_i_raw_sari_del_fns),
                    get_avg(per_i_raw_sari_del_fps))

  logging.info(
      'SARI ADD: Added Unigram Overlap Raw Count  tp:%.2f fn:%.2f fp:%.2f',
      sari_add_tps, sari_add_fns, sari_add_fps)
  logging.info(
      'SARI DEL: Deleted Unigram Overlap Raw Count tp:%.2f fn:%.2f fp:%.2f ',
      sari_del_tps, sari_del_fns, sari_del_fps)
  add_p, add_r, add_f1 = get_p_r_f1(sari_add_tps, sari_add_fps, sari_add_fns)
  del_p, del_r, del_f1 = get_p_r_f1(sari_del_tps, sari_del_fps, sari_del_fns)
  logging.info('SARI ADD p:%.2f r:%.2f f1:%.2f', add_p, add_r, add_f1)
  logging.info('SARI DEL p:%.2f r:%.2f f1:%.2f', del_p, del_r, del_f1)
  return dict(
      add_p=add_p,
      add_r=add_r,
      add_f1=add_f1,
      del_p=del_p,
      del_r=del_r,
      del_f1=del_f1)


def score_classification(annotation_dict, prediction_dict):
  """Computes the classification score."""
  match_list = []
  category_prediction_list = []
  for ex_id in list(annotation_dict.keys()):
    annotations = annotation_dict[ex_id]
    predict_label = (prediction_dict[ex_id].category == 'IMPOSSIBLE')
    category_prediction_list.append(predict_label)
    annotation_labels = [
        (label.category == 'IMPOSSIBLE') for label in annotations
    ]
    match_list.extend([(predict_label == annotation_label)
                       for annotation_label in annotation_labels])
  binary_agreement_score = get_avg(match_list)
  impossible_ratio = get_avg(category_prediction_list)
  logging.info('Impossible Category Ratio: %.2f', impossible_ratio)
  logging.info('Annotation Match Score: %.2f', binary_agreement_score)
  return dict(
      impossible_ratio=impossible_ratio,
      binary_agreement_score=binary_agreement_score)

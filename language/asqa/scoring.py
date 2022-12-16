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
# pylint: disable=g-explicit-length-test,dangerous-default-value,redefined-outer-name,g-explicit-bool-comparison, missing-function-docstring,missing-module-docstring,raise-missing-from,g-complex-comprehension
import argparse
import collections
import json
import re
import string

import nltk
import numpy as np
from rouge_score import rouge_scorer
from rouge_score import scoring

nltk.download('punkt')


# answer normalization
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def _rouge_calculation(hypotheses,
                       references1,
                       references2=[],
                       metrics=['rougeLsum']):
  """Internal function for rouge scoring.

  If two references are provided,
  the best score is chosen for each instance.

  Args:
    hypotheses: list of predicted long answers
    references1: list of references to score hypotheses against
    references2: optional list of references to score hypotheses against
    metrics: evaluation metric

  Returns:
    dictionary representation of rouge scores
  """

  if references2 == []:
    references2 = references1

  scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
  aggregator = scoring.BootstrapAggregator()

  for i in range(len(hypotheses)):
    scores1 = scorer.score(references1[i], hypotheses[i])
    scores2 = scorer.score(references2[i], hypotheses[i])
    if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
      aggregator.add_scores(scores1)
    else:
      aggregator.add_scores(scores2)

  scores = {m: [] for m in metrics}

  for m in metrics:
    fmeasure = aggregator.aggregate()[m].mid.fmeasure
    scores[m].append(fmeasure)

  for m in scores:
    scores[m] = 100 * sum(scores[m]) / len(scores[m])

  return scores


def rouge(hypotheses,
          references1,
          references2=None,
          metrics=['rougeLsum'],
          target_keys=None):
  """Main function for rouge scoring.

  If two references are provided,
  the best score is chosen for each instance.

  Args:
    hypotheses: {key: answer} dict of predicted long answers
    references1: {key: reference} dict of references to score hypotheses against
    references2: optional {key: reference} dict of references to score
      hypotheses against
    metrics: list of evaluation metrics
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    dictionary representation of rouge scores
  """

  # stemmer = PorterStemmer()
  h, r1, r2 = [], [], []

  for key in references1:
    if target_keys is not None and key not in target_keys:
      continue

    h.append(hypotheses[key])
    r1.append(references1[key])

    if references2 is not None:
      r2.append(references2[key])

  if 'rougeLsum' in metrics:
    h = ['\n'.join(nltk.sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(nltk.sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(nltk.sent_tokenize(text.lower())) for text in r2]
  scores = _rouge_calculation(h, r1, r2, metrics)

  return scores


def _exact_presence(short_answers, context):
  """Verify if any of the answers is present in the given context.

  Args:
    short_answers: list of short answers to look for in the context
    context: a paragraph to search for short answers

  Returns:
    true if any of the short answers is present in the context
  """

  n_short_answers = [normalize_answer(sa) for sa in short_answers]
  n_context = normalize_answer(context)

  for ans in n_short_answers:
    if ans in n_context:
      return True

  return False


def str_em(predictions, asqa, target_keys=None):
  """Compute STR-EM metric.

  Args:
    predictions: {key: answer} dictionary of predicted answers
    asqa: dict representation of asqa
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    Value of the STR-EM metric
  """
  acc = []

  for key, context in predictions.items():
    if target_keys is not None and key not in target_keys:
      continue

    loc_acc = []

    for qa_pair in asqa[key]['qa_pairs']:
      loc_acc.append(_exact_presence(qa_pair['short_answers'], context))

    acc.append(np.mean(loc_acc))

  return 100 * np.mean(acc)


def compute_len(predictions, target_keys=None):
  """Compute average lenght of predictions.

  Args:
    predictions: {key: answer} dict of predicted answers
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    average length of predicted answers
  """

  res = 0
  cntr = 0

  for key in predictions:
    if target_keys is not None and key not in target_keys:
      continue

    res += len(predictions[key].split())
    cntr += 1

  return res / cntr


def _get_tokens(s):
  """Split the string into tokens.

  Args:
    s: string to be split

  Returns:
    list of tokens
  """

  if not s:
    return []
  return normalize_answer(s).split()


def _compute_f1(a_gold, a_pred):
  """Compute F1 score between two strings.

  Args:
    a_gold: string one
    a_pred: string two

  Returns:
        f1 score
  """

  gold_toks = _get_tokens(a_gold)
  pred_toks = _get_tokens(a_pred)

  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())

  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)

  if num_same == 0:
    return 0

  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)

  return f1


def _compute_exact(a_gold, a_pred):
  """Check whether two strings are equal up to normalization.

  Args:
    a_gold: string one
    a_pred: string two

  Returns:
    1 if two strings are equal up to normalization and 0 otherwise
  """
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def score_qa_accuracy(predictions, asqa, target_keys=None):
  """Compute QA metrics.

  Args:
    predictions: {qa_key: short_answer} output of ROBERTA predictions of short
      answers to disambiguated questions
    asqa: dict representation of the asqa dataset
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    QA metrics (QA-EM, QA-F1, QA-Hit)
  """

  em, f1, bins = [], [], []

  for key, instance in asqa.items():

    if target_keys is not None and key not in target_keys:
      continue

    loc_counter, loc_em, loc_f1 = 0, 0, 0

    for idx, qa_pair in enumerate(instance['qa_pairs']):
      answers = qa_pair['short_answers']
      full_key = key + '_' + str(idx)
      prediction = predictions[full_key]

      if not isinstance(prediction, list):
        prediction = [prediction]

      loc_em += max([_compute_exact(a, p) for a in answers for p in prediction])
      loc_f1 += max([_compute_f1(a, p) for a in answers for p in prediction])

      loc_counter += 1

    em.append(loc_em / loc_counter)
    f1.append(loc_f1 / loc_counter)
    bins.append(loc_em == loc_counter)

  return {
      'QA-EM': 100 * np.mean(em),
      'QA-F1': 100 * np.mean(f1),
      'QA-Hit': 100 * np.mean(bins)
  }


def compute_all_scores(hypotheses,
                       asqa,
                       regime='eval',
                       target_idx=0,
                       rouge_metrics=['rougeLsum'],
                       target_keys=None):
  """This function computes values of the following metrics: LENGTH, ROUGE-L, STR-EM, QA-EM, QA-F1, QA-HIT, OVERALL SCORE.

  Args:
    hypotheses: a dict of the form {"answers": {key1: answer1, key2: answer2,
      ...}
      "qa": {qa_key1: short_answer1, qa_key2: short_answer2, ...}} the first
        part of the dict ("answers") is mandatory and needs to contain predicted
        long answers to ASQA questions the second part ("qa") is option and
        needs to contain predicted short answers to disambiguated questions
    asqa: dict representation of the ASQA dataset
    regime: if "eval", then two references are used to compute ROUGE, if "train"
      then only one reference is used for ROUGE
    target_idx: specifies which of the two references to use in the train mode
    rouge_metrics: list of ROUGE metrics to compute
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    All scores
  """

  if regime not in ['eval', 'train']:
    raise ValueError('Regime must be either \"train\" or \"eval\"')

  references1 = {}
  for key in asqa:
    references1[key] = asqa[key]['annotations'][
        0 if regime == 'eval' else target_idx]['long_answer']
  references2 = {
      key: asqa[key]['annotations'][1]['long_answer'] for key in asqa
  } if regime == 'eval' else None

  scores = rouge(
      hypotheses['answers'],
      references1,
      references2,
      metrics=rouge_metrics,
      target_keys=target_keys)
  scores['length'] = compute_len(hypotheses['answers'], target_keys)
  scores['str_em'] = str_em(hypotheses['answers'], asqa, target_keys)

  if 'qa' in hypotheses:
    qa_scores = score_qa_accuracy(hypotheses['qa'], asqa, target_keys)

    for m in qa_scores:
      scores[m] = qa_scores[m]

    if 'rougeLsum' not in scores:
      scores['ovscore'] = 'Undefined'
    else:
      scores['ovscore'] = np.sqrt(scores['QA-F1'] * scores['rougeLsum'])

  return scores


def parse_args(argv=None):
  parse = argparse.ArgumentParser()
  parse.add_argument('--asqa', type=str, help='Path to the ASQA data')

  parse.add_argument(
      '--split',
      type=str,
      default='dev',
      help='What data split you want to evaluate on')

  parse.add_argument(
      '--predictions', type=str, help='Path to model predictions')
  parse.add_argument(
      '--roberta_output', type=str, help='Path to Roberta output')
  parse.add_argument('--out_dir', type=str, help='Output path')
  # parse the arguments
  return parse.parse_args(argv)


if __name__ == '__main__':
  args = parse_args()
  try:
    with open(args.asqa, 'r') as handler:
      asqa = json.load(handler)[args.split]
  except FileNotFoundError:
    raise ValueError('Cannot open ASQA, abort')
  except KeyError:
    raise ValueError('Wrong split is provided, abort')

  try:
    with open(args.predictions, 'r') as handler:
      predictions = json.load(handler)
  except FileNotFoundError:
    raise ValueError('Cannot open predictions, abort')

  hypotheses = {'answers': predictions}

  if args.roberta_output is not None:
    try:
      with open(args.roberta_output, 'r') as handler:
        qa_preds = json.load(handler)
    except FileNotFoundError:
      raise ValueError('Cannot open predictions, abort')

    hypotheses['qa'] = qa_preds

  scores = compute_all_scores(hypotheses, asqa)
  print(json.dumps(scores, indent=2))
  out_fn = args.out_dir + '/final_eval_results.json'
  with open(out_fn, 'w') as outfile:
    json.dump(scores, outfile)

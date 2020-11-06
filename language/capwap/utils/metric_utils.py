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
"""Utilities for computing metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import re
import string


from language.capwap.utils import nltk_utils
from language.capwap.utils import reward_utils
import tensorflow.compat.v1 as tf

# ------------------------------------------------------------------------------
#
# Data loading helpers.
#
# ------------------------------------------------------------------------------


def clean_tokens(tokens):
  """Tokens are lower-cased, stripped of punctuation, and filtered for NULL."""
  tokens = [w.lower().strip(string.punctuation) for w in tokens]
  tokens = [w for w in tokens if w]
  tokens = tokens or ["NULL"]
  return tokens


def load_predictions(filename, vocab):
  """Load predictions from file.

  Wordpieces are converted back (approximately) into PTB-style tokens.

  Args:
    filename: Path to prediction file.
    vocab: Instance of text_utils.Vocab.

  Returns:
    predictions: Dictionary of image_id --> question_id --> list(caption).
  """
  predictions = collections.defaultdict(dict)
  with tf.io.gfile.GFile(filename, "r") as f:
    for line in f:
      entry = json.loads(line)
      image_id = entry["image_id"]
      question_id = entry.get("question_id", -1)
      beams = []
      for ids in entry["token_ids"]:
        wordpieces = [vocab.i2t(i) for i in ids]
        caption = vocab.detokenize(wordpieces)
        ptb_tokens = clean_tokens(nltk_utils.word_tokenize(caption))
        beams.append(" ".join(ptb_tokens))
      predictions[image_id][question_id] = beams
  return predictions


def load_answers(question_ids, filename, vocab):
  """Load gold answers from file.

  Wordpieces are converted back (approximately) into PTB-style tokens.

  Args:
    question_ids: Question ids to load.
    filename: Path to answer file.
    vocab: Instance of text_utils.Vocab.

  Returns:
    answers: Dictionary of image_id --> question_id --> set(answer).
  """
  with tf.io.gfile.GFile(filename, "r") as f:
    vqa = json.load(f)

  answers = collections.defaultdict(lambda: collections.defaultdict(list))
  for image_id, entry in vqa.items():
    for question_id, qa in entry.items():
      image_id = int(image_id)
      question_id = int(question_id)
      if question_id in question_ids:
        for answer in qa["answers"]:
          answer = vocab.detokenize(answer.split(" "))
          answer = clean_tokens(nltk_utils.word_tokenize(answer))
          answers[image_id][question_id].append(" ".join(answer))

  return answers


def load_references(image_ids, coco_annotations):
  """Load COCO references from file.

  References are tokenized in the PTB-style.

  Args:
    image_ids: Image ids to load references for.
    coco_annotations: Path to COCO data directory.

  Returns:
    references: Dictionary of image_id --> list(captions)
  """
  references = collections.defaultdict(list)
  val_file = "%s/captions_%s2014.json" % (coco_annotations, "val")
  train_file = "%s/captions_%s2014.json" % (coco_annotations, "train")

  for filename in [val_file, train_file]:
    with tf.io.gfile.GFile(filename, "r") as f:
      data = json.load(f)
    for annotation in data["annotations"]:
      image_id = annotation["image_id"]
      if image_id not in image_ids:
        continue
      caption = clean_tokens(nltk_utils.word_tokenize(annotation["caption"]))
      references[image_id].append(" ".join(caption))
  return references


def load_spans(filename, vocab):
  """Load spans from file.

  Wordpieces are converted back (approximately) into PTB-style tokens.

  Args:
    filename: Path to predicted spans file.
    vocab: Instance of text_utils.Vocab.

  Returns:
    spans: Dictionary of image_id --> question_id --> beam_id --> span.
  """
  spans = {}
  with tf.io.gfile.GFile(filename, "r") as f:
    for line in f:
      entry = json.loads(line)
      image_id = entry["image_id"]
      if image_id not in spans:
        spans[image_id] = {}
      question_id = entry["question_id"]
      if question_id not in spans[image_id]:
        spans[image_id][question_id] = {}
      beam_id = entry["beam_id"]
      span = vocab.detokenize([vocab.i2t(i) for i in entry["span"]])
      ptb_tokens = clean_tokens(nltk_utils.word_tokenize(span))
      spans[image_id][question_id][beam_id] = " ".join(ptb_tokens)

  # Flatten to lists.
  for image_id, qas in spans.items():
    for question_id, beams in qas.items():
      qas[question_id] = [beams[i] for i in range(len(beams))]

  return spans


# ------------------------------------------------------------------------------
#
# QA metric helpers.
#
# ------------------------------------------------------------------------------


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
  """F1 score."""
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = (
      collections.Counter(prediction_tokens)
      & collections.Counter(ground_truth_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def compute_em(prediction, ground_truth):
  """Exact match score."""
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_indicator(prediction, ground_truth):
  """Indicator score."""
  return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
  """Take the average best score against all ground truth answers.

  This is a bit different than SQuAD in that there are multiple answers
  **and** predictions that we average over. For some situations (e.g., *top k*
  beams or multiple human references) we might want to calculate the average
  performance. In most cases, however, predictions will be a list of length 1.

  Args:
    metric_fn: Callable on (prediction, ground_truth).
    predictions: List of whitespace separated prediction tokens.
    ground_truths: List of whitespace separated answer tokens.

  Returns:
    max_score: Max output of metric_fn.
  """
  all_metrics = []
  for prediction in predictions:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
      score = metric_fn(prediction, ground_truth)
      scores_for_ground_truths.append(score)
    all_metrics.append(max(scores_for_ground_truths))
  return sum(all_metrics) / len(all_metrics)


def qa_score(spans, answers):
  """Compute official exact match and F1 scores.

  Args:
    spans: Dictionary of image_id --> question_id --> list(span).
    answers: Dictionary of image_id --> question_id --> list(answers).

  Returns:
    results: Dictionary of exact_match and f1_score.
  """
  em = f1 = total = 0.0
  for image_id, questions in spans.items():
    for question_id, spans in questions.items():
      ground_truths = answers[image_id][question_id]
      em += metric_max_over_ground_truths(compute_em, spans, ground_truths)
      f1 += metric_max_over_ground_truths(compute_f1, spans, ground_truths)
      total += 1
  results = dict(exact_match=em / total, f1_score=f1 / total)
  tf.logging.info("Computed QA score on %d values.", total)
  return results


def indicator_score(captions, answers):
  """Compute indicator score (1 if answer is contained in the caption, else 0).

  Args:
    captions: Dictionary of image_id --> question_id --> list(caption).
    answers: Dictionary of image_id --> question_id --> list(answers).

  Returns:
    results: Dictionary of indicator score.
  """
  indicator = total = 0.0
  for image_id, questions in captions.items():
    for question_id, ground_truths in answers[image_id].items():
      # If the specific question_id is not here, then we use the caption for the
      # whole image (i.e., we're not doing conditional decoding). Let's just
      # assert that there is indeed only one caption in this case.
      if question_id not in questions:
        assert len(questions) == 1
        image_captions = next(iter(questions.values()))
      else:
        image_captions = questions[question_id]
      indicator += metric_max_over_ground_truths(compute_indicator,
                                                 image_captions, ground_truths)
      total += 1
  results = dict(indicator=indicator / total)
  tf.logging.info("Computed indicator score on %d values.", total)
  return results


# ------------------------------------------------------------------------------
#
# Reference-based metrics.
#
# ------------------------------------------------------------------------------


def bleu_score(predictions, references):
  """Compute BLEU score for predictions."""
  score = 0
  return dict(bleu_4=score)


def rouge_score(predictions, references):
  """Compute ROUGE score for predictions."""
  score = 0
  return dict(rouge=score)


def cider_score(predictions, references):
  """Compute CIDEr score for predictions."""
  score = 0
  return dict(cider=score)


# ------------------------------------------------------------------------------
#
# Functions to run full evaluations.
#
# ------------------------------------------------------------------------------


def evaluate_captions(caption_file, coco_annotations, vocab):
  """Evaluate reference-based metrics on predicted captions.

  Args:
    caption_file: Path to predicted captions.
    coco_annotations: Path to COCO data directory.
    vocab: Instance of text_utils.Vocab.

  Returns:
    results: Dictionary of BLEU, ROUGE, and CIDEr scores.
  """
  results = {}

  tf.logging.info("Loading predictions...")
  captions = load_predictions(caption_file, vocab)

  # Change to image_id --> [top caption].
  flat = {}
  for image_id, questions in captions.items():
    for captions in questions.values():
      flat[image_id] = captions[:1]
      break
  captions = flat

  tf.logging.info("Loading references...")
  references = load_references(set(captions.keys()), coco_annotations)

  tf.logging.info("Computing BLEU...")
  results.update(bleu_score(captions, references))

  tf.logging.info("Computing ROUGE...")
  results.update(rouge_score(captions, references))

  tf.logging.info("Computing CIDEr...")
  results.update(cider_score(captions, references))

  return results


def evaluate_questions(caption_file, vocab, question_file, params=None):
  """Evaluate QA-based metrics on predicted captions.

  If multiple captions are associated with an image/question pair, the average
  max metric score with the answer set is taken.

  Args:
    caption_file: Path to predicted captions.
    vocab: Instance of text_utils.Vocab.
    question_file: Path to dataset questions.
    params: Dictionary of model parameters.

  Returns:
    results: Dictionary of indicator, exact_match, and F1 scores.
  """
  results = {}

  tf.logging.info("Loading predictions...")

  # 1) Load full captions.
  captions = load_predictions(caption_file, vocab)

  # 2) Generate spans.
  span_file = os.path.splitext(caption_file)[0] + ".question_spans"
  tf.logging.info("Computing spans...")
  reward_utils.write_spans(
      caption_file=caption_file,
      question_file=question_file,
      output_file=span_file,
      vocab=vocab,
      params=params)

  # 3) Load spans from disk.
  tf.logging.info("Loading spans...")
  spans = load_spans(span_file, vocab)
  question_ids = set()
  for qas in spans.values():
    for qid in qas.keys():
      question_ids.add(qid)

  # 4) Compute QA scores.
  tf.logging.info("Loading answers...")
  answers = load_answers(question_ids, question_file, vocab)

  tf.logging.info("Computing QA score...")
  results.update(qa_score(spans, answers))

  tf.logging.info("Computing indicator score...")
  results.update(indicator_score(captions, answers))

  return results

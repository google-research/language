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
"""Util functions to handle preprocessing, shared across scripts."""
import collections as cll
import random
import re
import string

import numpy as np


def get_squad_question_starters():
  q_tokens = [
      u"Why", u"On", u"Along", u"During", u"At", u"A", u"For", u"According",
      u"What", u"How", u"Who", u"When", u"In", u"Which", u"Where", u"The",
      u"To", u"From", u"By", u"what", u"After", u"Whose", u"What's"
  ]
  q_probs = [1.0 / len(q_tokens) for _ in q_tokens]
  return q_tokens, q_probs


def get_boolq_question_starters():
  q_tokens = [
      u"is", u"can", u"does", u"are", u"do", u"did", u"was", u"has", u"will",
      u"the", u"have"
  ]
  q_probs = [1.0 / len(q_tokens) for _ in q_tokens]
  return q_tokens, q_probs


def build_question_starter_vocab(sents_data):
  """Build a vocab of first word question tokens."""
  starter_vocab = []

  for instance in sents_data["data"]:
    for para in instance["paragraphs"]:
      for qa in para["qas"]:
        question = qa["question"]
        starter_vocab.append(question.split()[0])

  starter_vocab = cll.Counter(starter_vocab)
  starter_vocab = [(k, v) for k, v in starter_vocab.items() if v >= 200]
  starter_vocab.sort(key=lambda x: x[1], reverse=True)
  start_vocab_total = sum([x[1] for x in starter_vocab])

  starter_vocab_tokens = [x[0] for x in starter_vocab]
  starter_vocab_probs = [float(x[1]) / start_vocab_total for x in starter_vocab]

  return starter_vocab_tokens, starter_vocab_probs


def build_question_starter_vocab_boolq(sents_data):
  """Build a vocab of first word  question tokens."""
  starter_vocab = []

  for instance in sents_data:
    question = instance["question"]
    starter_vocab.append(question.split()[0])

  starter_vocab = cll.Counter(starter_vocab)
  starter_vocab = [(k, v) for k, v in starter_vocab.items() if v >= 50]
  starter_vocab.sort(key=lambda x: x[1], reverse=True)
  start_vocab_total = sum([x[1] for x in starter_vocab])

  starter_vocab_tokens = [x[0] for x in starter_vocab]
  starter_vocab_probs = [float(x[1]) / start_vocab_total for x in starter_vocab]

  return starter_vocab_tokens, starter_vocab_probs


def build_thief_vocab(thief_paragraphs):
  """Return unigram freqs of thief dataset used for building random paras."""
  thief_vocab = cll.Counter("\n".join(thief_paragraphs).split())
  thief_vocab = [(k, v) for k, v in thief_vocab.items()]
  thief_vocab.sort(key=lambda x: x[1], reverse=True)
  thief_vocab_total = sum([x[1] for x in thief_vocab])

  thief_tokens = [x[0] for x in thief_vocab]
  thief_probs = [float(x[1]) / thief_vocab_total for x in thief_vocab]

  thief_lens = [len(x.split()) for x in thief_paragraphs]

  return thief_tokens, thief_probs, thief_lens


def uniform_sampling_paragraph(thief_vocab, para_len=None):
  """Sample words uniformly randomly from a vocabulary to build a paragraph."""
  if para_len is None:
    # randomly choose a length from 75 to 500
    para_len = np.random.randint(75, 500)
  return " ".join([random.choice(thief_vocab) for _ in range(para_len)])


def frequency_sampling_paragraph(thief_vocab, thief_probs, para_len=None):
  """Sample words according to a unigram frequency to build a paragraph."""
  if para_len is None:
    # randomly choose a length from 75 to 500
    para_len = np.random.randint(75, 500)
  assert len(thief_probs) == len(thief_vocab)

  bow_para = np.random.multinomial(para_len, thief_probs)
  nonzero_indices = np.nonzero(bow_para)[0]

  bow_indices = []
  for index, freq in zip(nonzero_indices, bow_para[nonzero_indices]):
    bow_indices.extend([index for _ in range(freq)])

  assert len(bow_indices) == para_len
  random.shuffle(bow_indices)

  return " ".join([thief_vocab[i] for i in bow_indices])


def choose_random_question(para_text, question_sampling_scheme):
  """Sample a paragraph's words to form a question."""
  if "anchor_gaussian" in question_sampling_scheme:
    q_length = np.random.randint(5, 15)
    para_tokens = para_text.split()
    anchor = np.random.randint(0, len(para_tokens))

    question = []
    std_dev = max(int(len(para_tokens) / 8.0), int(q_length))

    for _ in range(q_length):
      token_index = -1
      while token_index < 0 or token_index >= len(para_tokens):
        token_index = int(np.round(np.random.normal(loc=anchor, scale=std_dev)))
      question.append(para_tokens[token_index])

    return " ".join(question)

  elif "random" in question_sampling_scheme:
    q_length = np.random.randint(5, 15)
    para_tokens = para_text.split()
    question = []

    for _ in range(q_length):
      question.append(random.choice(para_tokens))

    return " ".join(question)


def postprocess_question(question, q_tokens, q_probs, sampling_scheme):
  """Add a ? at the end of SQuAD questions and start questions with Wh* word."""
  assert len(q_tokens) == len(q_probs)
  if "uniform" in sampling_scheme:
    q_word = random.choice(q_tokens)
  else:
    q_word = q_tokens[np.argmax(np.random.multinomial(1, q_probs))]
  # using q_word.encode throws a can't concat str to bytes. Review and check the same for then ext function's encode 
  question = q_word + " " + question + "?"
  return question


def postprocess_question_boolq(question, q_tokens, q_probs, sampling_scheme):
  """Start BoolQ questions with a yes/no question starter word."""
  assert len(q_tokens) == len(q_probs)
  if "uniform" in sampling_scheme:
    q_word = random.choice(q_tokens)
  else:
    q_word = q_tokens[np.argmax(np.random.multinomial(1, q_probs))]
  return q_word.encode("ascii", "ignore") + " " + question


def para_question_overlap(para_text, question):
  """Overlap paragraph and question based on word level F1."""
  normalized_question = normalize_text(question)
  num_question_tokens = len(normalized_question.split())
  best_f1 = 0.0
  best_f1_indices = [0, 1]
  para_tokens = para_text.split()

  for ans_len in range(1, num_question_tokens + 1):
    for i in range(len(para_tokens) - ans_len):
      end = i + ans_len
      normalized_para_substr = normalize_text(" ".join(para_tokens[i:end]))
      current_f1 = f1_score(normalized_para_substr, normalized_question)
      if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_indices = [i, end]

  start, end = best_f1_indices
  answer_string = " ".join(para_tokens[start:end])
  return answer_string


def normalize_text(s):
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


def f1_score(prediction, ground_truth):
  """Calculate word level F1 score."""
  prediction_tokens = prediction.split()
  ground_truth_tokens = ground_truth.split()
  common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

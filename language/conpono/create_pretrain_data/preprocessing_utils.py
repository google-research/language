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
"""Utility functions shared between pretraining proc code for books and wiki."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from bert import tokenization
import tensorflow.compat.v1 as tf


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, label):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.label = label

  def __str__(self):
    s = ""
    for sent in self.tokens:
      s += "tokens: %s\n" % (" ".join(
          [tokenization.printable_text(x) for x in sent]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def convert_instance_to_tf_example(tokenizer, instance, max_seq_length):
  """Convert instance object to TFExample.

  Args:
      tokenizer: instance of language.bert.tokenization.Tokenizer
      instance: TrainingInstance object
      max_seq_length: max number of tokens for fixed size model input

  Returns:
      A tuple containing a tfexample of the instance and the features used to
      create the tfexample in a dictionary
  """

  input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
  input_mask = [1] * len(input_ids)
  segment_ids = list(instance.segment_ids)
  assert len(input_ids) <= max_seq_length

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)

  # These are not needed but are made to fit the BERT api
  masked_lm_positions = [0] * max_seq_length
  masked_lm_ids = [0] * max_seq_length
  masked_lm_weights = [1.0] * len(masked_lm_ids)
  features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
  features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
  features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
  if isinstance(instance.label, list):
    features["order"] = create_int_feature(instance.label)
  else:
    features["next_sentence_labels"] = create_int_feature([instance.label])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example, features


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_paragraph_order_from_document(document, max_seq_length, rng):
  """Create TrainingInstances for paragraph reconstruction."""
  instances = []

  # Account for [CLS] and a [SEP] after each sent (5)
  max_num_tokens = max_seq_length - 6

  for paragraph in document:
    if len(paragraph) < 5:  # skip paragraphs that are too short
      continue
    # We'll need fixed length inputs so we'll only take the first 5 sentences
    paragraph = paragraph[:5]

    # Shuffle paragraph with index and output
    paragraph_with_labels = list(enumerate(paragraph))
    rng.shuffle(paragraph_with_labels)
    order, paragraph = zip(*paragraph_with_labels)

    truncate_paragraph(paragraph, max_num_tokens, rng)
    assert sum([len(p) for p in paragraph]) <= max_num_tokens

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for idx, sent in enumerate(paragraph):
      for token in sent:
        tokens.append(token)
        segment_ids.append(idx)
      tokens.append("[SEP]")
      segment_ids.append(0)

    instance = TrainingInstance(tokens, segment_ids, list(order))
    instances.append(instance)

  return instances


def create_instances_from_document(document, max_seq_length, rng):
  """Creates `TrainingInstance`s for a single document."""

  # For every document (list of paragraphs, list of sentences) we want to create
  # A training instance which is a pair of sentences and their order label.
  instances = []

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  for paragraph in document:
    for sent_idx, sent in enumerate(paragraph):
      if sent_idx >= len(paragraph) - 1:
        continue
      sent_a = sent
      sent_b = paragraph[sent_idx + 1]
      for label in [0, 1]:
        # listify sents so that it copies the object rather than just makes a
        # pointer because truncation will modify the object
        if label == 0:
          tokens_a, tokens_b = list(sent_a), list(sent_b)
        else:  # swap order for incorrect ordering
          tokens_a, tokens_b = list(sent_b), list(sent_a)

        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        instance = TrainingInstance(tokens, segment_ids, label)
        instances.append(instance)

  return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def truncate_paragraph(paragraph, max_num_tokens, rng):
  """Truncates a 5 sequences to a maximum sequence length."""
  while True:
    total_length = sum([len(p) for p in paragraph])
    if total_length <= max_num_tokens:
      break

    max_index = paragraph.index(max(paragraph))
    assert len(paragraph[max_index]) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del paragraph[max_index][0]
    else:
      paragraph[max_index].pop()

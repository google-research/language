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
import numpy as np
import tensorflow.compat.v1 as tf

LONG_CTX = "long_ctx"
ONE_SENT_CTX = "one_sent_ctx"


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, label, label_types):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.label = label
    self.label_types = label_types

  def __str__(self):
    s = ""
    for sent in self.tokens[0]:
      s += "tokens: %s\n" % (" ".join(
          [tokenization.printable_text(x) for x in sent]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids[0]]))
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
  input_ids_list = [
      tokenizer.convert_tokens_to_ids(tokens) for tokens in instance.tokens
  ]
  input_mask_list = [[1] * len(input_ids) for input_ids in input_ids_list]
  segment_ids_list = list(instance.segment_ids)

  for input_ids in input_ids_list:
    assert len(input_ids) <= max_seq_length

  features = collections.OrderedDict()
  for i in range(len(input_ids_list)):
    while len(input_ids_list[i]) < max_seq_length:
      input_ids_list[i].append(0)
      input_mask_list[i].append(0)
      segment_ids_list[i].append(0)
    assert len(input_ids_list[i]) == max_seq_length
    assert len(input_mask_list[i]) == max_seq_length
    assert len(segment_ids_list[i]) == max_seq_length

    features["input_ids" + str(i)] = create_int_feature(input_ids_list[i])
    features["input_mask" + str(i)] = create_int_feature(input_mask_list[i])
    features["segment_ids" + str(i)] = create_int_feature(segment_ids_list[i])

  # These are not needed but are made to fit the BERT api
  masked_lm_positions = [0] * max_seq_length
  masked_lm_ids = [0] * max_seq_length
  masked_lm_weights = [1.0] * len(masked_lm_ids)
  features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
  features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
  features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

  features["labels"] = create_int_feature(instance.label)
  features["label_types"] = create_int_feature(instance.label_types)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example, features


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def sample_rand_from_paragraphs(document, paragraph, num):
  all_sents = []
  for para in document:
    if para == paragraph:
      continue
    all_sents.extend(para)
  return list(np.random.choice(all_sents, num, replace=False))


def sample_rand_from_docs(other_docs, num):
  all_sents = []
  for para in other_docs:
    all_sents.extend(para)
  return list(np.random.choice(all_sents, num, replace=False))


def check_doc_len(document, paragraph, num):
  all_sents = []
  for para in document:
    if para == paragraph:
      continue
    all_sents.extend(para)
  return len(all_sents) >= num


def create_instances_from_document(document, max_seq_length, rng, other_docs,
                                   d_format):
  """Creates `TrainingInstance`s for a single document."""

  # For every document (list of paragraphs, list of sentences) we want to create
  # A training instance which is a pair of sentences and their order label.
  instances = []

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  for paragraph in document:
    for sent_idx, sent in enumerate(paragraph):
      # for each sentence set up the following targets
      # target sentences for -5 < k < 5, but only 4 at a time
      # random samples from same paragraph
      # random samples from document
      # random samples from other docs
      # total of 32 options for each k
      if sent_idx >= len(paragraph) - 1:
        continue
      if d_format == LONG_CTX:
        if sent_idx >= len(paragraph) - 4:  # long context needs to stop earlier
          continue
      context = sent
      if d_format == LONG_CTX:
        context = []
        for inner_sent in paragraph[sent_idx:sent_idx + 4]:
          context.extend(inner_sent)

      if d_format == LONG_CTX:
        target_idx = np.array(range(sent_idx - 2, sent_idx + 6))
        assert len(target_idx) == 8
      else:
        target_idx = np.array(range(sent_idx - 2, sent_idx + 3))
        assert len(target_idx) == 5

      if target_idx[0] < 0:
        target_idx -= target_idx[0]
      elif target_idx[-1] >= len(paragraph):
        target_idx -= (target_idx[-1] - len(paragraph) + 1)
      remove_idx = np.where(target_idx == sent_idx)
      target_idx = np.delete(target_idx, remove_idx)
      if d_format == LONG_CTX:
        for _ in range(3):
          target_idx = np.delete(target_idx, remove_idx)
      assert len(target_idx) == 4

      # map target_idx to label_type
      rel_positions = range(-4, 5)
      del rel_positions[4]
      label_type_map = {i: j for (i, j) in zip(rel_positions, range(8))}
      if d_format == ONE_SENT_CTX:
        k_func = lambda targ, sent_idx: targ - sent_idx
      else:

        def k_func(targ, sent_idx):
          return (targ - sent_idx) - (0 if sent_idx > targ else 3)

      label_types = [
          label_type_map[k_func(targ, sent_idx)] for targ in target_idx
      ]

      sent_a = paragraph[target_idx[0]]
      sent_b = paragraph[target_idx[1]]
      sent_c = paragraph[target_idx[2]]
      sent_d = paragraph[target_idx[3]]

      if not check_doc_len(document, paragraph, 23):
        continue
      neg_sents = sample_rand_from_paragraphs(document, paragraph, 23)
      neg_sents += sample_rand_from_docs(other_docs, 23)

      for target_sent in [sent_a, sent_b, sent_c, sent_d] + neg_sents:
        if d_format == ONE_SENT_CTX:
          truncate_next_sent(context, target_sent, max_num_tokens, rng)
        else:
          truncate_seq_pair(context, target_sent, max_num_tokens, rng)

      token_list = []
      segment_list = []
      for target_sent in [sent_a, sent_b, sent_c, sent_d] + neg_sents:
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in context:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in target_sent:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        token_list.append(tokens)
        segment_list.append(segment_ids)

      instance = TrainingInstance(token_list, segment_list, list(target_idx),
                                  label_types)
      instances.append(instance)

  return instances


def truncate_next_sent(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    assert len(tokens_b) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del tokens_b[0]
    else:
      tokens_b.pop()


def truncate_long_ctx(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    assert len(tokens_a) >= 1, "Context got cut to zero"

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del tokens_a[0]
    else:
      tokens_a.pop()


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

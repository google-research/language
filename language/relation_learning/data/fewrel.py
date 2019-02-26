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
"""Processors for extracting FewRel datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

_RESERVED_WORDPIECES = [
    "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[E1]", "[/E1]", "[E2]", "[/E2]"
]


def reserved_wordpieces():
  """Returns list of wordpieces reserved with special function."""
  return _RESERVED_WORDPIECES


def _tokenize_with_entity_markers(tokens, tokenizer, e1, e2):
  """Apply wordpiece tokenization with entity markers around entities."""

  def tokenize(start, end):
    return tokenizer.tokenize(" ".join(tokens[start:end]))

  if e1[0] < e2[0]:
    return (tokenize(0, e1[0]) + ["[E1]"] + tokenize(e1[0], e1[1] + 1) +
            ["[/E1]"] + tokenize(e1[1] + 1, e2[0]) + ["[E2]"] +
            tokenize(e2[0], e2[1] + 1) + ["[/E2]"] + tokenize(e2[1] + 1, None))
  else:
    return (tokenize(0, e2[0]) + ["[E2]"] + tokenize(e2[0], e2[1] + 1) +
            ["[/E2]"] + tokenize(e2[1] + 1, e1[0]) + ["[E1]"] +
            tokenize(e1[0], e1[1] + 1) + ["[/E1]"] + tokenize(e1[1] + 1, None))


class RelationInputExample(object):
  """A single training/test example for SemEval 2010 Task 8 task."""

  def __init__(self, guid, wordpieces, label, e1, e2):
    """Constructs a RelationInputExample.

    Args:
      guid: Unique id for the example.
      wordpieces: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: string. The label of the example. This should be specified for
        train and dev examples, but not for test examples.
      e1: Pair of indices demarking entity 1.
      e2: Pair of indices demarking entity 2.
    """
    self.guid = guid
    self.wordpieces = wordpieces
    self.label = label
    self.e1 = e1
    self.e2 = e2

  def __str__(self):
    return "%s %s %s %s %s" % (self.guid, " ".join(self.wordpieces), self.label,
                               self.e1, self.e2)


class FewShotRelationInputExample(object):
  """A single training/test example for FewRel task."""

  def __init__(self, guid, sets, query):
    """Constructs a FewShotRelationInputExample.

    Args:
      guid: Unique id for the example.
      sets: dictionary of examples, aggregated by label.
      query: example to classify (into one of the sets).
    """
    self.guid = guid
    self.sets = sets
    self.query = query

  def __str__(self):
    out = "%s" % self.query
    for _, setlist in self.sets.iteritems():
      for item in setlist:
        out += "%s" % item
    return out


class FewRelProcessor(object):
  """Processor for the FewRel dataset."""

  def __init__(self, tokenizer, max_seq_length, add_entity_markers=False):
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length
    self._add_entity_markers = add_entity_markers

  @classmethod
  def _read_json(cls, intput_file):
    """Reads a JSON encoded file."""
    with tf.gfile.Open(intput_file, "r") as f:
      return json.load(f)

  def _json_entry_to_example(self, entry, candidate_index, example_index):
    """Converts FewRel entry (candidate or query) into RelationInputExample."""
    text = " ".join(entry["tokens"])
    e1_tokens = [entry["h"][2][0][0], entry["h"][2][0][-1]]
    e2_tokens = [entry["t"][2][0][0], entry["t"][2][0][-1]]
    e1 = [0, 0]
    e2 = [0, 0]
    if self._add_entity_markers:
      wordpieces = _tokenize_with_entity_markers(
          entry["tokens"], self._tokenizer, e1_tokens, e2_tokens)
    else:
      wordpieces = self._tokenizer.tokenize(text)

    if len(wordpieces) > self._max_seq_length - 2:
      tf.logging.info("[_create_entrys] Truncating sentence [%s] %s",
                      len(wordpieces), " ".join(wordpieces).encode("utf-8"))
      wordpieces = wordpieces[0:(self._max_seq_length - 2)]
    wordpieces = ["[CLS]"] + wordpieces + ["[SEP]"]

    token_index = 0
    cur_token_length = 0
    for i, wordpiece in enumerate(wordpieces):
      if wordpiece in reserved_wordpieces():
        continue

      # For the start of entity, index should be first wordpiece.
      if e1[0] == 0 and e1_tokens[0] == token_index:
        e1[0] = i
      # For the end of entity, index should be last wordpiece.
      if e1_tokens[1] == token_index:
        e1[1] = i
      if e2[0] == 0 and e2_tokens[0] == token_index:
        e2[0] = i
      if e2_tokens[1] == token_index:
        e2[1] = i

      cur_token_length += len(wordpiece) - 2 * int(wordpiece.startswith("##"))
      if cur_token_length == len(entry["tokens"][token_index]):
        token_index += 1
        cur_token_length = 0

    return RelationInputExample(
        guid="%s:%s" % (candidate_index, example_index),
        wordpieces=wordpieces,
        label=candidate_index,
        e1=e1,
        e2=e2)

  def process_file(self, file_name):
    """Gets a collection of `RelationInputExample`s for the train set."""
    data = self._read_json(os.path.join(file_name))

    num_classes = len(data[0]["meta_train"])
    num_examples_per_class = len(data[0]["meta_train"][0])

    tf.logging.info("Number of tests: %s, classes: %s, examples per class %s",
                    len(data), num_classes, num_examples_per_class)

    output = []
    for entry in data:
      candidate_sets = entry["meta_train"]
      query = entry["meta_test"]

      # Process the query context.
      query_example = self._json_entry_to_example(query, -1, -1)

      # Process the candidate contexts.
      output_sets = {}
      for (candidate_index, candidate_set) in enumerate(candidate_sets):
        output_candidate_set = []
        for (example_index, candidate) in enumerate(candidate_set):
          relation_input_example = self._json_entry_to_example(
              candidate, candidate_index, example_index)
          output_candidate_set.append(relation_input_example)
        output_sets[candidate_index] = output_candidate_set

      output.append(
          FewShotRelationInputExample(
              guid="", sets=output_sets, query=query_example))

    return output, num_classes, num_examples_per_class

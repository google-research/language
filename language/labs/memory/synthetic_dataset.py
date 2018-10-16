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
"""Generates sequence of binary patterns or kv pairs to test associative memory.

Pattern task: Given N patterns, retrieve the right pattern via its degraded
version, where some of the bits are set to 0.

Symbolic key-value task: Given a string of concatenated key-value pairs,
retrieve the right value given the key.

See [Miconi et al. 2018] Differentiable Plasticity
(https://arxiv.org/abs/1804.02464) and [Ba et al. 2016] Using Fast Weights to
to Attend to the Recent Past (https://arxiv.org/abs/1610.06258v1) for details
of task design.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string

import numpy as np
import tensorflow as tf


def generate_pattern_data(num_patterns, pattern_size):
  """Generates a sequence of patterns followed by a degraded query pattern.

  Args:
    num_patterns (int): Number of unique patterns in the sequence
    pattern_size (int): Dimensionality of each pattern

  Returns:
    seq: Numpy array, the sequence of patterns to be presented
      shape [num_patterns + 1, pattern_size]
    target: Numpy array, the pattern we expect to retrieve for the degraded
      query pattern
      shape [pattern_size,]
    target_idx (int): The index into the list of unique patterns for the target
    patterns: List of np arrays, all unique patterns
  """
  patterns = []
  for _ in range(num_patterns):
    pattern = np.random.choice([-1, 1], size=(pattern_size,), p=[.5, .5])
    patterns.append(pattern)

  # Choose one pattern to degrade
  target_idx = random.choice(range(num_patterns))
  target = patterns[target_idx]
  degraded = target.copy()
  degraded_idxs = np.random.choice(
      pattern_size, pattern_size // 2, replace=False)
  degraded[degraded_idxs] = 0
  patterns.append(degraded)

  seq = np.array(patterns)

  return seq, target, target_idx, patterns


def generate_pattern_data_selective(num_patterns, num_patterns_store,
                                    pattern_size):
  """Generates a sequence of patterns followed by a degraded query pattern.

  Args:
    num_patterns (int): Number of unique patterns in the sequence
    num_patterns_store (int): Number of patterns we actually have to store
    pattern_size (int): Dimensionality of each pattern

  Returns:
    seq: Numpy array, the sequence of patterns to be presented.
      shape [num_patterns + 1, pattern_size]
    target: Numpy array, the pattern we expect to retrieve for the degraded
      query pattern.
      shape [pattern_size,]
    target_idx (int): The index into the list of unique patterns for the target.
    patterns: List of np arrays, all unique patterns.
      Patterns we need to remember (that may be queried) have their last bit set
      to 1, otherwise 0.
  """
  patterns = []
  for _ in range(num_patterns):
    pattern = np.random.choice([-1, 1], size=(pattern_size,), p=[.5, .5])
    patterns.append(pattern)

  # Choose patterns that are important to remember
  remember_idxs = np.random.choice(
      range(num_patterns), size=num_patterns_store, replace=False)
  patterns = [
      np.append(p, [1]) if i in remember_idxs else np.append(p, [0])
      for i, p in enumerate(patterns)
  ]

  # Choose one pattern to degrade
  target_idx = random.choice(range(num_patterns))
  target = patterns[target_idx]
  degraded = target.copy()
  degraded_idxs = np.random.choice(
      pattern_size, pattern_size // 2, replace=False)
  degraded[degraded_idxs] = 0
  patterns.append(degraded)

  seq = np.array(patterns)

  return seq, target, target_idx, patterns


def generate_symbolic_data(num_pairs):
  """Generates a sequence of key-value pairs followed by a query key.

  Args:
    num_pairs (int): Number of pairs

  Returns:
    seq_text (str): Sequence of kv pairs, followed by a ?,
      followed by the query key.
    seq_encoded (numpy arr): Sequence of kv pairs, encoded into vocab indices.
    target_val (str): Digit, the value we expect to retrieve for the key.
    target_val_encoded (int): Encoded target_val
    target_idx (int): The index into the list of pairs for the target
  """
  pairs = zip(
      np.random.choice(list(string.ascii_lowercase), num_pairs, replace=False),
      np.random.choice(list("0123456789"), num_pairs)
  )

  vocab = get_symbolic_vocab()

  # Choose a query key
  target_idx = random.choice(range(num_pairs))
  target_key, target_val_text = pairs[target_idx]
  target_val_encoded = vocab.index(target_val_text)

  seq_text = "".join([k + v for k, v in pairs]) + "?" + target_key
  seq_encoded = [vocab.index(char) for char in seq_text]

  return seq_text, seq_encoded, target_val_text, target_val_encoded, target_idx


def get_pattern_dataset(n=100000,
                        num_patterns=3,
                        pattern_size=50,
                        selective=False,
                        num_patterns_store=None):
  """Generates a dataset of sequences of patterns and retrieval targets.

  Args:
    n: Number of examples
    num_patterns: Number of unique patterns in the sequence
    pattern_size: Dimensionality of each pattern
    selective (bool): True if only a subset of patterns needs to be stored.
    num_patterns_store: Number of patterns to store if selective=True.

  Returns:
    A tf.data.Dataset created from a dict with property "seqs,"
    containing the sequences of randomly generated binary patterns, and
    "targets," containing the ground-truth pattern to retrieve for the last
    degraded query pattern in the sequence.
  """
  seqs = []
  targets = []

  for _ in range(n):
    if selective:
      if num_patterns_store is None:
        num_patterns_store = num_patterns // 10
      seq, target, _, _ = generate_pattern_data_selective(
          num_patterns, num_patterns_store, pattern_size)
    else:
      seq, target, _, _ = generate_pattern_data(num_patterns, pattern_size)

    seqs.append(seq)
    targets.append(target)

  return tf.data.Dataset.from_tensor_slices({
      "seqs": np.array(seqs, dtype=np.float32),
      "targets": np.array(targets, dtype=np.int32)
  })


def get_symbolic_dataset(_, n=100000, num_pairs=5):
  """Generates a dataset of sequences of key-value pairs and retrieval targets.

  Args:
    n: Number of examples
    num_pairs: Number of pairs in each sequence

  Returns:
    A tf.data.Dataset created from a dict with property "seqs,"
    containing the sequences of randomly generated key-value pairs, and
    "targets," containing the ground-truth value to retrieve for the query key.
  """

  seqs = []
  targets = []

  for _ in range(n):
    _, seq_encoded, _, target_encoded, _ = generate_symbolic_data(num_pairs)
    seqs.append(seq_encoded)
    targets.append(target_encoded)

  return tf.data.Dataset.from_tensor_slices({
      "seqs": np.array(seqs, dtype=np.int32),
      "targets": np.array(targets, dtype=np.int32)
  })


def get_symbolic_vocab():
  """Gets the vocabulary for the symbolic task.

  Returns:
    A list with a-z, 0-9, and ?.
  """
  return list(string.ascii_lowercase) + list(string.digits + "?")


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
"""Utilites for handling spans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def max_scoring_span(start_scores, end_scores):
  """Find the maximum-scoring valid span given start and end scores.

  Args:
    start_scores: [batch_size, seq_len]
    end_scores: [batch_size, seq_len]

  Returns:
    best_start: [batch_size]
    best_end: [batch_size]
    best_score: [batch_size]
  """

  def _cumulative_max(accumulation, score):
    cum_max, cum_bp, cum_index, direction = accumulation
    current_max = tf.maximum(cum_max, score)
    current_bp = tf.where(tf.greater(cum_max, score), cum_bp, cum_index)
    current_index = cum_index + direction
    return current_max, current_bp, current_index, direction

  seq_len = tf.shape(start_scores)[-1]

  def _find_best_span(args):
    """Compute the best span."""
    current_start_scores, current_end_scores = args

    # [seq_len], [seq_len]
    start_max, start_backpointers, _, _ = tf.scan(
        fn=_cumulative_max,
        elems=current_start_scores,
        initializer=(float("-inf"), -1, 0, 1),
        back_prop=False,
        reverse=False)
    end_max, end_backpointers, _, _ = tf.scan(
        fn=_cumulative_max,
        elems=current_end_scores,
        initializer=(float("-inf"), -1, seq_len - 1, -1),
        back_prop=False,
        reverse=True)

    # []
    total_max = start_max + end_max
    best_index = tf.argmax(total_max, -1)
    best_start = start_backpointers[best_index]
    best_end = end_backpointers[best_index]
    best_score = total_max[best_index]

    return best_start, best_end, best_score

  # [batch_size], [batch_size]
  best_start, best_end, best_score = tf.map_fn(
      fn=_find_best_span,
      elems=(start_scores, end_scores),
      dtype=(tf.int32, tf.int32, tf.float32),
      back_prop=False)
  return best_start, best_end, best_score

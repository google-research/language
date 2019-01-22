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
"""Simplified version of Document Reader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.layers import cudnn_layers
from language.common.utils import tensor_utils
import tensorflow as tf


def _attend_to_question(context_emb, question_emb, question_mask, hidden_size):
  """Compute similarity scores with the dot product of two linear layers.

  Args:
    context_emb: <float32> [batch_size, max_context_len, hidden_size]
    question_emb: <float32> [batch_size, max_question_len, hidden_size]
    question_mask: <float32> [batch_size, max_question_len]
    hidden_size: Integer indicating the size of the projection.

  Returns:
    attended_emb: <float32> [batch_size, max_context_len, hidden_size]
  """
  with tf.variable_scope("question_projection"):
    # [batch_size, max_question_len, hidden_size]
    projected_question_emb = tf.layers.dense(question_emb, hidden_size)
  with tf.variable_scope("context_projection"):
    # [batch_size, max_context_len, hidden_size]
    projected_context_emb = tf.layers.dense(context_emb, hidden_size)

  # [batch_size, max_context_len, max_question_len]
  attention_scores = tf.matmul(
      projected_context_emb, projected_question_emb, transpose_b=True)
  attention_scores += tf.expand_dims(tf.log(question_mask), 1)
  attention_weights = tf.nn.softmax(attention_scores)

  # [batch_size, max_context_len, hidden_size]
  attended_emb = tf.matmul(attention_weights, question_emb)
  return attended_emb


def _attention_pool(question_emb, question_mask):
  """Reduce variable length question embeddings to fixed length via attention.

  Args:
    question_emb: <float32> [batch_size, max_question_len, hidden_size]
    question_mask: <float32> [batch_size, max_question_len]

  Returns:
    pooled_emb: <float32> [batch_size, hidden_size]
  """
  # [batch_size, max_question_len, 1]
  attention_scores = tf.layers.dense(question_emb, 1)
  attention_scores += tf.expand_dims(tf.log(question_mask), -1)
  attention_weights = tf.nn.softmax(attention_scores, 1)

  # [batch_size, 1, hidden_size]
  pooled_emb = tf.matmul(attention_weights, question_emb, transpose_a=True)

  # [batch_size, hidden_size]
  pooled_emb = tf.squeeze(pooled_emb, 1)
  return pooled_emb


def _bilinear_score(context_emb, question_emb):
  """Compute a bilinear score between the context and question embeddings.

  Args:
    context_emb: <float32> [batch_size, max_context_len, hidden_size]
    question_emb: <float32> [batch_size, hidden_size]

  Returns:
    scores: <float32> [batch_size, max_context_len]
  """
  # [batch_size, hidden_size]
  projected_question_emb = tf.layers.dense(question_emb,
                                           tensor_utils.shape(context_emb, -1))

  # [batch_size, max_context_len, 1]
  scores = tf.matmul(context_emb, tf.expand_dims(projected_question_emb, -1))

  return tf.squeeze(scores, -1)


def score_endpoints(question_emb,
                    question_len,
                    context_emb,
                    context_len,
                    hidden_size,
                    num_layers,
                    dropout_ratio,
                    mode,
                    use_cudnn=None):
  """Compute two scores over context words based on the input embeddings.

  Args:
    question_emb: <float32> [batch_size, max_question_len, hidden_size]
    question_len: <int32> [batch_size]
    context_emb: <float32>[batch_size, max_context_len, hidden_size]
    context_len: <int32> [batch_size]
    hidden_size: Size of hidden layers.
    num_layers: Number of LSTM layers.
    dropout_ratio: The probability of dropping out hidden units.
    mode: Object of type tf.estimator.ModeKeys.
    use_cudnn: Specify the use of cudnn. `None` denotes automatic selection.

  Returns:
    start_scores: <float32> [batch_size, max_context_words]
    end_scores: <float32> [batch_size, max_context_words]
  """
  # [batch_size, max_question_len]
  question_mask = tf.sequence_mask(
      question_len, tensor_utils.shape(question_emb, 1), dtype=tf.float32)

  # [batch_size, max_context_len, hidden_size]
  attended_emb = _attend_to_question(
      context_emb=context_emb,
      question_emb=question_emb,
      question_mask=question_mask,
      hidden_size=hidden_size)

  # [batch_size, max_context_len, hidden_size * 2]
  context_emb = tf.concat([context_emb, attended_emb], -1)

  with tf.variable_scope("contextualize_context"):
    # [batch_size, max_context_len, hidden_size]
    contextualized_context_emb = cudnn_layers.stacked_bilstm(
        input_emb=context_emb,
        input_len=context_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        mode=mode,
        use_cudnn=use_cudnn)
  with tf.variable_scope("contextualize_question"):
    # [batch_size, max_question_len, hidden_size]
    contextualized_question_emb = cudnn_layers.stacked_bilstm(
        input_emb=question_emb,
        input_len=question_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        mode=mode,
        use_cudnn=use_cudnn)
  if mode == tf.estimator.ModeKeys.TRAIN:
    contextualized_context_emb = tf.nn.dropout(contextualized_context_emb,
                                               1.0 - dropout_ratio)
    contextualized_question_emb = tf.nn.dropout(contextualized_question_emb,
                                                1.0 - dropout_ratio)

  # [batch_size, hidden_size]
  pooled_question_emb = _attention_pool(contextualized_question_emb,
                                        question_mask)

  if mode == tf.estimator.ModeKeys.TRAIN:
    pooled_question_emb = tf.nn.dropout(pooled_question_emb,
                                        1.0 - dropout_ratio)

  # [batch_size, max_context_len]
  with tf.variable_scope("start_scores"):
    start_scores = _bilinear_score(contextualized_context_emb,
                                   pooled_question_emb)
  with tf.variable_scope("end_scores"):
    end_scores = _bilinear_score(contextualized_context_emb,
                                 pooled_question_emb)
  context_log_mask = tf.log(
      tf.sequence_mask(
          context_len, tensor_utils.shape(context_emb, 1), dtype=tf.float32))
  start_scores += context_log_mask
  end_scores += context_log_mask
  return start_scores, end_scores

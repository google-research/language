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
"""Decomposable attention model adapted for NQ long answer task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags

from language.common.utils import tensor_utils
from language.question_answering.layers import decomposable_attention as decatt

import tensorflow as tf

flags.DEFINE_integer("hidden_size", 400, "Hidden size.")

flags.DEFINE_integer("hidden_layers", 2, "Number of hidden layers.")

flags.DEFINE_float("dropout_ratio", 0.1,
                   "The dropout to use in feed forward networks.")

flags.DEFINE_string("overlap_reduce_type", "max",
                    'Overlap reduce type. Options are "sum" and "max".')

flags.DEFINE_integer("max_pos_embs", 20,
                     "Maximum number of position embeddings")

flags.DEFINE_integer("pos_emb_dim", 5, "Dimension of position embeddings.")

FLAGS = flags.FLAGS


def _word_overlap_helper(question_ids, context_ids):
  # <int32> [batch_size, 1, question_len, 1]
  temp_question_ids = tf.expand_dims(tf.expand_dims(question_ids, 1), -1)

  # <int32> [batch_size, num_contexts, 1, context_len]
  temp_context_ids = tf.expand_dims(context_ids, 2)

  # Test equality with broadcasting.
  # Returns <bool> [batch_size, num_contexts, question_len, context_len]
  return tf.equal(temp_question_ids, temp_context_ids)


def _compute_word_overlap(context_ids, context_len, question_ids, question_len,
                          reduce_type, weighted, vocab_df):
  """Compute word overlap between question and context ids.

  Args:
    context_ids: <int32> [batch_size, num_contexts, max_context_len]
    context_len: <int32> [batch_size, num_contexts]
    question_ids: <int32> [batch_size, max_question_len]
    question_len: <int32> [batch_size]
    reduce_type: String for reduce type when computing overlap. Choices are: max
      - Allows at most one match per question word. sum - Sums over all matches
      for each question word.
    weighted: Boolean indicate whether or not weight the overlap by IDF.
    vocab_df: Tensor of shape [vocab_size] for word frequency. Computes this at
      the document-level if not given.

  Returns:
    overlap: <float32> [batch_size, num_contexts]

  Raises:
    Exception: If invalid reduce_type is provided.
  """
  # <float> [batch_size, num_contexts, question_len, context_len]
  overlap = tf.to_float(
      _word_overlap_helper(question_ids=question_ids, context_ids=context_ids))

  # <float> [batch_size, question_len]
  question_mask = tf.sequence_mask(
      question_len, tf.shape(question_ids)[1], dtype=tf.float32)

  # <float> [batch_size, num_contexts, context_len]
  context_mask = tf.sequence_mask(
      context_len, tf.shape(context_ids)[2], dtype=tf.float32)

  overlap *= tf.expand_dims(tf.expand_dims(question_mask, 1), -1)
  overlap *= tf.expand_dims(context_mask, 2)

  if weighted:
    if vocab_df is None:
      # Use document-level IDF computed with respect to the current batch.
      flat_context_ids = tf.to_int32(tf.reshape(context_ids, [-1]))

      # <float> [number of unique words]
      vocab_df = tf.bincount(
          flat_context_ids,
          minlength=tf.reduce_max(question_ids) + 1,
          dtype=tf.float32)

      # Replace all zeros with ones.
      vocab_df = tf.where(
          tf.equal(vocab_df, 0), x=tf.ones_like(vocab_df), y=vocab_df)

    # <float>[batch_size, question_len] expanded to
    # <float> [batch_size, 1, question_len, 1]
    question_df = tf.gather(vocab_df, question_ids)
    question_df = tf.expand_dims(tf.expand_dims(question_df, 1), -1)

    # <float> [batch_size, num_contexts, question_len, context_len]
    overlap = tf.divide(tf.to_float(overlap), question_df)

  if reduce_type == "max":
    # <float> [batch_size, num_contexts]
    overlap = tf.reduce_sum(tf.reduce_max(overlap, axis=[3]), axis=[2])
  elif reduce_type == "sum":
    # <float> [batch_size, num_contexts]
    overlap = tf.reduce_sum(overlap, axis=[2, 3])
  else:
    raise Exception("Reduce type %s is invalid." % reduce_type)

  return overlap


def _get_position_scores(batch_size, num_contexts, max_pos_embs, pos_emb_dim):
  """Get scores that only depend on position.

  Args:
    batch_size: <int32> []
    num_contexts: <int32> []
    max_pos_embs: <int32> for max number of contexts that get a unique embedding
      (all contexts above this number are mapped to the same embedding).
    pos_emb_dim: <int32>  for position embedding dimension.

  Returns:
    <float32> [batch_size, num_contexts, pos_emb_dim] for position embeddings.
  """
  pos_embedding_params = tf.get_variable(
      name="position_embeddings",
      shape=[max_pos_embs, pos_emb_dim],
      trainable=True)

  # <int32> [num_contexts]
  context_pos = tf.range(0, num_contexts)
  context_pos = tf.clip_by_value(
      t=context_pos, clip_value_min=-1, clip_value_max=max_pos_embs - 1)

  # <float> [num_contexts, pos_emb_dim]
  pos_embs = tf.nn.embedding_lookup(
      params=pos_embedding_params, ids=context_pos, partition_strategy="div")

  # <float> [batch_size, num_contexts, pos_emb_dim]
  pos_embs = tf.tile(tf.expand_dims(pos_embs, 0), [batch_size, 1, 1])
  return pos_embs


def _get_non_neural_features(question_tok_wid, question_lens, context_tok_wid,
                             context_lens):
  """Gets the three non-neural features and the context mask.

  Args:
    question_tok_wid: <int32> [batch_size, question_len]
    question_lens: <int32> [batch_size]
    context_tok_wid: <int32> [batch_size, num_context, context_len]
    context_lens: <int32> [batch_size, num_context]

  Returns:
    question_tok_wid: [batch_size, num_contexts, 1]
    context_tok_wid: [batch_size, num_contexts, 1]
    pos_embs: [batch_size, num_contexts, pos_emb_dim]
  """
  # <int32> [batch_size, num_contexts]
  weighted_num_overlap = _compute_word_overlap(
      context_ids=context_tok_wid,
      context_len=context_lens,
      question_ids=question_tok_wid,
      question_len=question_lens,
      reduce_type=FLAGS.overlap_reduce_type,
      vocab_df=None,
      weighted=True)

  # <int32> [batch_size, num_contexts]
  unweighted_num_overlap = _compute_word_overlap(
      context_ids=context_tok_wid,
      context_len=context_lens,
      question_ids=question_tok_wid,
      question_len=question_lens,
      reduce_type=FLAGS.overlap_reduce_type,
      vocab_df=None,
      weighted=False)

  # <int32> [batch_size, num_contexts, pos_emb_dim]
  pos_embs = _get_position_scores(
      batch_size=tf.shape(context_tok_wid)[0],
      num_contexts=tf.shape(context_tok_wid)[1],
      max_pos_embs=FLAGS.max_pos_embs,
      pos_emb_dim=FLAGS.pos_emb_dim)

  weighted_num_overlap = tf.expand_dims(weighted_num_overlap, -1)
  unweighted_num_overlap = tf.expand_dims(unweighted_num_overlap, -1)

  return weighted_num_overlap, unweighted_num_overlap, pos_embs


def build_model(question_tok_wid, question_lens, context_tok_wid, context_lens,
                embedding_weights, mode):
  """Wrapper around for Decomposable Attention model for NQ long answer scoring.

  Args:
    question_tok_wid: <int32> [batch_size, question_len]
    question_lens: <int32> [batch_size]
    context_tok_wid: <int32> [batch_size, num_context, context_len]
    context_lens: <int32> [batch_size, num_context]
    embedding_weights: <float> [vocab_size, embed_dim]
    mode: One of the keys from tf.estimator.ModeKeys.

  Returns:
    context_scores: <float> [batch_size, num_context]
  """
  # <float> [batch_size, question_len, embed_dim]
  question_emb = tf.nn.embedding_lookup(embedding_weights, question_tok_wid)
  # <float> [batch_size, num_context, context_len, embed_dim]
  context_emb = tf.nn.embedding_lookup(embedding_weights, context_tok_wid)

  question_emb = tf.layers.dense(
      inputs=question_emb,
      units=FLAGS.hidden_size,
      activation=None,
      name="reduce_emb",
      reuse=False)

  context_emb = tf.layers.dense(
      inputs=context_emb,
      units=FLAGS.hidden_size,
      activation=None,
      name="reduce_emb",
      reuse=True)

  batch_size, num_contexts, max_context_len, embed_dim = (
      tensor_utils.shape(context_emb))
  _, max_question_len, _ = tensor_utils.shape(question_emb)

  # <float> [batch_size * num_context, context_len, embed_dim]
  flat_context_emb = tf.reshape(context_emb, [-1, max_context_len, embed_dim])

  # <int32> [batch_size * num_context]
  flat_context_lens = tf.reshape(context_lens, [-1])

  # <float> [batch_size * num_context, question_len, embed_dim]
  question_emb_tiled = tf.tile(
      tf.expand_dims(question_emb, 1), [1, num_contexts, 1, 1])
  flat_question_emb_tiled = tf.reshape(question_emb_tiled,
                                       [-1, max_question_len, embed_dim])

  # <int32> [batch_size * num_context]
  question_lens_tiled = tf.tile(
      tf.expand_dims(question_lens, 1), [1, num_contexts])
  flat_question_lens_tiled = tf.reshape(question_lens_tiled, [-1])

  # <float> [batch_size * num_context, hidden_size]
  flat_decatt_emb = decatt.decomposable_attention(
      emb1=flat_question_emb_tiled,
      len1=flat_question_lens_tiled,
      emb2=flat_context_emb,
      len2=flat_context_lens,
      hidden_size=FLAGS.hidden_size,
      hidden_layers=FLAGS.hidden_layers,
      dropout_ratio=FLAGS.dropout_ratio,
      mode=mode)

  # <float> [batch_size, num_context, hidden_size]
  decatt_emb = tf.reshape(flat_decatt_emb,
                          [batch_size, num_contexts, FLAGS.hidden_size])

  weighted_num_overlap, unweighted_num_overlap, pos_embs = (
      _get_non_neural_features(
          question_tok_wid=question_tok_wid,
          question_lens=question_lens,
          context_tok_wid=context_tok_wid,
          context_lens=context_lens))

  final_emb = tf.concat(
      [decatt_emb, weighted_num_overlap, unweighted_num_overlap, pos_embs], -1)

  # Final linear layer to get score.
  # <float> [batch_size, num_context]
  context_scores = tf.layers.dense(inputs=final_emb, units=1, activation=None)
  context_scores = tf.squeeze(context_scores, -1)

  return context_scores

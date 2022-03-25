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
"""Utilities for deriving QA-driven rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import json
import os
import string

from language.capwap.datasets import rc_dataset
from language.capwap.utils import experiment_utils
from language.capwap.utils import io_utils
from language.capwap.utils import tensor_utils
from language.capwap.utils import text_utils
from language.capwap.utils import transformer_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub

RolloutOutputs = collections.namedtuple(
    "RolloutOutputs", ["token_ids", "mask", "scores", "rewards"])

# ------------------------------------------------------------------------------
#
# QA model helpers.
#
# ------------------------------------------------------------------------------


def max_scoring_span(start_scores, end_scores, max_length, no_answer_bias=0):
  """Compute max scoring span, using the sum of start and end scores.

  Args:
    start_scores: <float32> [batch_size, seq_len]
    end_scores: <float32> [batch_size, seq_len]
    max_length: <int32> Max answer length.
    no_answer_bias: <float32> Log-odds threshold for "no-answer" selection. I.e.
      if log p(span=i,j)/p(span=NULL) > no_answer_bias, then select i, j as the
      span, and NULL otherwise.

  Returns:
    start: <int32> [batch_size]
    end: <int32> [batch_size]
  """
  # Create sparse tensor of size [seq_len].
  seq_len = tensor_utils.shape(start_scores, -1)
  no_answer_bias = tf.scatter_nd([[0]], [no_answer_bias], [seq_len])
  no_answer_bias = tf.cast(no_answer_bias, tf.float32)

  # Apply bias to CLS token logits.
  no_answer_bias = tf.div(no_answer_bias, 2)
  start_scores += tf.expand_dims(no_answer_bias, 0)
  end_scores += tf.expand_dims(no_answer_bias, 0)

  # Compute outer sum, and mask to be upper triangular.
  # This gives a matrix of start[i] + end[j] scores, where j >= i.
  scores = tf.expand_dims(start_scores, 2) + tf.expand_dims(end_scores, 1)
  mask = (1 - tf.matrix_band_part(tf.ones_like(scores), 0, max_length - 1))
  scores -= mask * 1e-4

  def map_fn(inputs):
    flattened = tf.reshape(inputs, [-1])
    argmax = tf.argmax(flattened, output_type=tf.int32)
    indices = tensor_utils.unravel_index_2d(argmax, inputs.shape)
    score = flattened[argmax]
    return indices, score

  # Return i, j indices of max-scoring entry.
  with tf.device("/cpu"):
    endpoints, span_scores = tf.map_fn(
        fn=map_fn, elems=scores, dtype=(tf.int32, tf.float32))
  start = endpoints[:, 0]
  end = endpoints[:, 1]

  return start, end, span_scores


def _get_rc_model_input(
    question_ids,
    question_mask,
    context_ids,
    context_mask,
    vocab,
):
  """Create RC module input from separate batched components.

  Args:
    question_ids: <int32> [batch_size, question_len]
    question_mask: <int32> [batch_size, question_len]
    context_ids: <int32> [batch_size, context_len]
    context_mask: <int32> [batch_size, context_len]
    vocab: Instance of text_utils.Vocab.

  Returns:
    input_ids: <int32> [batch_size, rc_input_len]
    input_mask: <int32> [batch_size, rc_input_len]
    segment_ids: <int32> [batch_size, rc_input_len]
  """
  # Get batch size.
  batch_size = tensor_utils.shape(context_ids, 0)

  # Get special tokens.
  cls = vocab.t2i(vocab.CLS)
  sep = vocab.t2i(vocab.SEP)

  # Join question, context, and special tokens.
  cls_batch = tf.fill([batch_size, 1], cls)
  sep_batch = tf.fill([batch_size, 1], sep)
  input_ids = tf.concat(
      [cls_batch, question_ids, sep_batch, context_ids, sep_batch], axis=1)

  # Create and join segment ids.
  segment_a_ids = tf.fill(
      [batch_size, tensor_utils.shape(question_ids, 1) + 2], 0)
  segment_b_ids = tf.fill(
      [batch_size, tensor_utils.shape(context_ids, 1) + 1], 1)
  segment_ids = tf.concat([segment_a_ids, segment_b_ids], axis=1)

  # Create joined mask, accounting for special tokens gaps.
  gap_mask = tf.fill([batch_size, 1], 1)
  input_mask = tf.concat(
      [gap_mask, question_mask, gap_mask, context_mask, gap_mask], axis=1)
  bool_mask = tf.cast(input_mask, tf.bool)

  # Select unmasked items and move all padding to the end.
  # Right now this looks like this:
  #   [CLS] X X X [PAD] ... [SEP] Y Y Y [PAD] ... [SEP] [PAD] ...
  # And we want to change it to look like this:
  #   [CLS] X X X [SEP] Y Y Y [SEP] [PAD] ...
  input_ids = tensor_utils.boolean_mask(input_ids, bool_mask)
  input_mask = tensor_utils.boolean_mask(input_mask, bool_mask)
  segment_ids = tensor_utils.boolean_mask(segment_ids, bool_mask)

  return input_ids, input_mask, segment_ids


def rc_span(
    question_ids,
    question_mask,
    context_ids,
    context_mask,
    rc_model,
    vocab,
    max_length=10,
    no_answer_bias=0,
):
  """Computes exact match score from QA model run on context.

  Args:
    question_ids: <int32> [batch_size, question_len]
    question_mask: <int32> [batch_size, question_len]
    context_ids: <int32> [batch_size, context_len]
    context_mask: <int32> [batch_size, context_len]
    rc_model: Extractive question answering model.
    vocab: Instance of text_utils.Vocab.
    max_length: Max answer length.
    no_answer_bias: Log-odds ratio for answer span over NULL.

  Returns:
    score: <float32> [batch_size]
  """
  # Mask out stop id in context if present.
  stop_id = vocab.t2i(vocab.SEP)
  stop_mask = tf.cast(tf.not_equal(context_ids, stop_id), tf.int32)
  context_mask *= stop_mask

  # Prepare rc inputs.
  input_ids, input_mask, segment_ids = _get_rc_model_input(
      question_ids=question_ids,
      question_mask=question_mask,
      context_ids=context_ids,
      context_mask=context_mask,
      vocab=vocab)

  # Get start/end logits from RC model.
  outputs = rc_model(
      inputs=dict(
          input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids),
      signature="extractive_qa",
      as_dict=True)

  # Dimensions
  batch_size = tensor_utils.shape(input_ids, 0)
  context_len = tensor_utils.shape(input_ids, 1)

  # Decode span.
  start_logits = tf.reshape(outputs["start_logits"], [-1, context_len])
  end_logits = tf.reshape(outputs["end_logits"], [-1, context_len])
  start, end, span_scores = max_scoring_span(
      start_scores=start_logits,
      end_scores=end_logits,
      max_length=max_length,
      no_answer_bias=no_answer_bias)

  # Expand shape to be compatible for broadcasting.
  start = tf.reshape(start, [-1, 1])
  end = tf.reshape(end, [-1, 1])

  # Create mask where mask[i, j] = True if i >= start and j <= end.
  # [batch_size, max_rc_input_len]
  mask = tf.tile(tf.expand_dims(tf.range(context_len), 0), [batch_size, 1])
  mask = tf.logical_and(tf.greater_equal(mask, start), tf.less_equal(mask, end))

  # Gather padded answer span from context.
  answer_span = tensor_utils.boolean_mask(input_ids, mask)

  return answer_span, span_scores


# ------------------------------------------------------------------------------
#
# QA reward function helpers.
#
# ------------------------------------------------------------------------------


def _get_normalized_set(vocab):
  """Build set of special token ids to ignore when computing matching metrics.

  Args:
    vocab: An instance of text_utils.Vocab.

  Returns:
    remove: A set of wordpiece token indices to ignore.
  """
  punct = {vocab.t2i(t) for t in string.punctuation if t in vocab}
  articles = {vocab.t2i(t) for t in ["a", "an", "the"] if t in vocab}
  plural = {vocab.t2i(t) for t in ["##s"] if t in vocab}
  contractions = {vocab.t2i(t) for t in ["s", "'s"] if t in vocab}
  remove = punct | articles | plural | contractions | {0}
  return remove


def exact_match(answer_ids, prediction_ids, vocab):
  """Compute exact match score between answer tokens and prediction tokens.

  Args:
    answer_ids: <int32> [batch_size, answer_length]
    prediction_ids: <int32> [batch_size, prediction_length]
    vocab: Instance of text_utils.Vocab.

  Returns:
    score: <float32> [batch_size] tensor of {0.0, 1.0}.
  """
  batch_size = tensor_utils.shape(answer_ids, 0)

  # Get cleanable words.
  remove_ids = list(_get_normalized_set(vocab))
  remove_ids = tf.reshape(remove_ids, [1, 1, -1])
  remove_ids = tf.tile(remove_ids, [batch_size, 1, 1])

  # Clean answer: remove tokens that are in the normalized set.
  should_keep = tf.reduce_all(
      tf.not_equal(tf.expand_dims(answer_ids, -1), remove_ids), axis=-1)
  answer_ids = tensor_utils.boolean_mask(answer_ids, should_keep)

  # Clean context: remove tokens that are in the normalized set.
  should_keep = tf.reduce_all(
      tf.not_equal(tf.expand_dims(prediction_ids, -1), remove_ids), axis=-1)
  prediction_ids = tensor_utils.boolean_mask(prediction_ids, should_keep)

  # Cleaned lengths.
  answer_len = tensor_utils.shape(answer_ids, 1)
  prediction_len = tensor_utils.shape(prediction_ids, 1)

  # Pad the shorter one to the length of the longer.
  padding = tf.maximum(0, prediction_len - answer_len)
  answer_ids = tf.pad(answer_ids, [[0, 0], [0, padding]])
  padding = tf.maximum(0, answer_len - prediction_len)
  prediction_ids = tf.pad(prediction_ids, [[0, 0], [0, padding]])

  # Check for equality: Padded A == Padded B?
  is_equal = tf.reduce_all(tf.equal(answer_ids, prediction_ids), axis=1)
  score = tf.cast(is_equal, tf.float32)

  return score


def f1_score(answer_ids, prediction_ids, vocab):
  """Compute F1 score between answer tokens and prediction tokens.

  Args:
    answer_ids: <int32> [batch_size, answer_length]
    prediction_ids: <int32> [batch_size, prediction_length]
    vocab: Instance of text_utils.Vocab.

  Returns:
    score: <float32> [batch_size] tensor of [0.0, 1.0].
  """
  # Order insensitive, so we just create a vocabulary sized bit tensor where
  # the vocabulary items that are not to be counted are masked out.
  vocab_size = len(vocab)
  remove_ids = list(_get_normalized_set(vocab))
  remove_mask = tf.expand_dims(tf.one_hot(remove_ids, vocab_size), 0)
  remove_mask = tf.reduce_sum(remove_mask, axis=1)
  remove_mask = tf.cast(tf.equal(remove_mask, 0), tf.float32)

  # [batch_size, vocab_size]
  answer_ids = tf.reduce_sum(tf.one_hot(answer_ids, vocab_size), axis=1)
  answer_ids *= remove_mask

  # [batch_size, vocab_size]
  prediction_ids = tf.reduce_sum(tf.one_hot(prediction_ids, vocab_size), axis=1)
  prediction_ids *= remove_mask

  # Compute multiset intersection, and count the size.
  intersection = tf.minimum(prediction_ids, answer_ids)
  intersection = tf.reduce_sum(intersection, axis=1)

  # Compute F1 score:
  #   Re(A, B) = |A \cap B| / |B|
  #   Pr(A, B) = |A \cap B| / |A|
  #   F1(A, B) = 2 * (Pr * Re) / (Pr + Re)
  recall = tf.div_no_nan(intersection, tf.reduce_sum(answer_ids, axis=1))
  precision = tf.div_no_nan(intersection, tf.reduce_sum(prediction_ids, axis=1))
  score = 2 * tf.div_no_nan(precision * recall, precision + recall)

  return score


def indicator_score(answer_ids, answer_mask, context_ids, vocab):
  """Compute indicator score of answer and context.

  Checks if the answer tokens are a subspan of the context.

  Args:
    answer_ids: <int32> [batch_size, answer_length]
    answer_mask: <int32> [batch_size, answer_length]
    context_ids: <int32> [batch_size, context_length]
    vocab: Instance of text_utils.Vocab.

  Returns:
    score: <float32> [batch_size] tensor of {0.0, 1.0}.
  """
  batch_size = tensor_utils.shape(answer_ids, 0)

  # Get cleanable words.
  remove_ids = list(_get_normalized_set(vocab))
  remove_ids = tf.reshape(remove_ids, [1, 1, -1])
  remove_ids = tf.tile(remove_ids, [batch_size, 1, 1])

  # Clean answer: remove tokens that are in the normalized set.
  should_keep = tf.reduce_all(
      tf.not_equal(tf.expand_dims(answer_ids, -1), remove_ids), axis=-1)
  answer_ids = tensor_utils.boolean_mask(answer_ids, should_keep)
  answer_mask = tensor_utils.boolean_mask(answer_mask, should_keep)

  # Clean context: remove tokens that are in the normalized set.
  should_keep = tf.reduce_all(
      tf.not_equal(tf.expand_dims(context_ids, -1), remove_ids), axis=-1)
  context_ids = tensor_utils.boolean_mask(context_ids, should_keep)

  # Cleaned lengths.
  answer_len = tensor_utils.shape(answer_ids, 1)
  context_len = tensor_utils.shape(context_ids, 1)

  # Pad start of context (to select NULL for over-length indices).
  context_ids = tf.pad(context_ids, [[0, 0], [1, 0]])
  context_len += 1

  # Sliding window approach: take the full context of length N and gather
  # it into a tensor with all windows of length M (a N x M tensor).
  # [context_len, answer_len]
  window_idx = tf.range(answer_len)
  window_idx = tf.tile(tf.expand_dims(window_idx, 0), [context_len, 1])
  offsets = tf.expand_dims(tf.range(context_len), 1)
  window_idx += offsets
  window_idx *= tf.cast(tf.less(window_idx, context_len), tf.int32)

  # [batch_size, context_len * answer_len]
  window_idx = tf.reshape(window_idx, [1, -1])
  window_idx = tf.tile(window_idx, [batch_size, 1])

  # [batch_size, context_len * answer_len]
  batch_idx = tf.range(batch_size)
  batch_idx = tf.expand_dims(batch_idx, 1)
  batch_idx = tf.tile(batch_idx, [1, context_len * answer_len])

  # [batch_size, context_len, answer_len]
  batch_idx = tf.reshape(batch_idx, [-1])
  window_idx = tf.reshape(window_idx, [-1])
  coords = tf.stack([batch_idx, window_idx], axis=1)
  window_ids = tf.gather_nd(context_ids, coords)
  window_ids = tf.reshape(window_ids, [batch_size, context_len, answer_len])

  # [batch_size, context_len, answer_len]
  answer_mask = tf.expand_dims(answer_mask, 1)
  window_ids *= answer_mask

  # Check for equality. The whole window has to match the answer, but only
  # one window has to count to be a positive indicator value.
  answer_ids = tf.expand_dims(answer_ids, 1)
  is_equal = tf.reduce_all(tf.equal(answer_ids, window_ids), axis=-1)
  score = tf.cast(tf.reduce_any(is_equal, axis=-1), tf.float32)

  return score


def compute_qa_rewards(
    question_ids,
    question_mask,
    context_ids,
    context_mask,
    answer_ids,
    answer_mask,
    rc_model,
    vocab,
    max_answer_length=10,
    no_answer_bias=0.0,
):
  """Compute all QA-based rewards.

  Args:
    question_ids: <int32> [batch_size, question_length]
    question_mask: <int32> [batch_size, question_length]
    context_ids: <int32> [batch_size, context_length]
    context_mask: <int32> [batch_size, context_length]
    answer_ids: <int32> [batch_size, answer_length]
    answer_mask: <int32> [batch_size, answer_length]
    rc_model: TF Hub module for extractive QA.
    vocab: Instance of text_utils.Vocab.
    max_answer_length: Maximum span length to predict.
    no_answer_bias: Log-odds threshold for answering a span over NULL.

  Returns:
    rewards: A dictionary with indicator, exact match, F1 scores,
             and predicted span information (ids and logits).
  """
  rewards = {}
  rewards["indicator"] = indicator_score(
      answer_ids=answer_ids,
      answer_mask=answer_mask,
      context_ids=context_ids,
      vocab=vocab)
  span_ids, span_scores = rc_span(
      question_ids=question_ids,
      question_mask=question_mask,
      context_ids=context_ids,
      context_mask=context_mask,
      rc_model=rc_model,
      vocab=vocab,
      max_length=max_answer_length,
      no_answer_bias=no_answer_bias)
  rewards["span_ids"] = span_ids
  rewards["span_scores"] = span_scores
  rewards["exact_match"] = exact_match(
      answer_ids=answer_ids, prediction_ids=span_ids, vocab=vocab)
  rewards["f1_score"] = f1_score(
      answer_ids=answer_ids, prediction_ids=span_ids, vocab=vocab)
  return rewards


# ------------------------------------------------------------------------------
#
# Policy rollout helpers.
#
# ------------------------------------------------------------------------------


def compute_rollouts(
    model,
    rc_model,
    features,
    encoder_cache,
    encoder_cache_mask,
    vocab,
    params,
):
  """Rollout model and compute rewards for each sample.

  Args:
    model: utils.transformer_utils.TransformerModel instance.
    rc_model: TF Hub module for extractive QA.
    features: Input features (questions and answers).
    encoder_cache: Transformer cache for encoded input.
    encoder_cache_mask: Input mask for the Transformer cache.
    vocab: Instance of text_utils.Vocab.
    params: Model parameters.

  Returns:
    rollout: Instance of RolloutOutputs.
  """
  # 1) First rollout the model with top-K beam search.
  rollout = transformer_utils.beam_search_decode(
      model=model,
      encoder_cache=encoder_cache,
      encoder_cache_mask=encoder_cache_mask,
      start_id=vocab.t2i(vocab.CLS),
      stop_id=vocab.t2i(vocab.SEP),
      segment_id=0,
      num_steps=params["decode_length"],
      beam_size=params["num_rollouts"],
      alpha=params["beam_length_penalty"],
      reuse=tf.AUTO_REUSE)

  # [batch_size, num_rollouts, rollout_length]
  batch_size = tensor_utils.shape(rollout.token_ids, 0)
  num_rollouts = tensor_utils.shape(rollout.token_ids, 1)
  rollout_ids = rollout.token_ids
  rollout_mask = rollout.mask

  # [batch_size * num_rollouts, rollout_length]
  rollout_length = tensor_utils.shape(rollout_ids, -1)
  rollout_ids = tf.reshape(rollout_ids, [-1, rollout_length])
  rollout_mask = tf.reshape(rollout_mask, [-1, rollout_length])

  # 2) Compute the QA rewards on the rollouts.
  # [batch_size * num_rollouts, question_length]
  question = tensor_utils.tile_batch(features["question_inputs"], num_rollouts)

  # [batch_size * num_rollouts, answer_length]
  answer = tensor_utils.tile_batch(features["answer_outputs"], num_rollouts)

  # [batch_size * num_rollouts]
  rewards = compute_qa_rewards(
      question_ids=question.token_ids,
      question_mask=question.mask,
      answer_ids=answer.token_ids,
      answer_mask=answer.mask,
      context_ids=rollout_ids[:, 1:],
      context_mask=rollout_mask[:, 1:],
      rc_model=rc_model,
      vocab=vocab,
      max_answer_length=params["answer_length"],
      no_answer_bias=params["no_answer_bias"])

  # [batch_size, num_rollouts, ...]
  reshaped_rewards = {}
  for k, v in rewards.items():
    if len(v.shape) > 1:
      v = tf.reshape(v, [batch_size, num_rollouts, -1])
    else:
      v = tf.reshape(v, [batch_size, num_rollouts])
    reshaped_rewards[k] = v

  # 3) Combine rollouts and rewards.
  rollouts = RolloutOutputs(
      token_ids=rollout.token_ids,
      mask=rollout.mask,
      scores=rollout.scores,
      rewards=reshaped_rewards)

  return rollouts


def sample_from_rollouts(rollouts, baseline=None, reward_type="exact_match"):
  """Sample a single example from the given rollouts.

  Args:
    rollouts: Instance of RolloutOutputs.
    baseline: <float32> [batch_size] Baseline value b for R'(y) = R(y) - b.
    reward_type: Choice between indicator, exact_match, and F1.

  Returns:
    rollout: Instance of text_utils.TextInputs.
    reward: <float32> [batch_size]
  """
  batch_size = tensor_utils.shape(rollouts.token_ids, 0)
  rollout_length = tensor_utils.shape(rollouts.token_ids, 2)

  # Self-critical baseline.
  if baseline is None:
    baseline = tf.zeros([batch_size])

  # [batch_size, num_rollouts]
  rewards = rollouts.rewards[reward_type] - tf.expand_dims(baseline, 1)

  # Mask zero reward samples.
  masked_scores = tf.where(
      tf.not_equal(rewards, 0), tf.zeros_like(rollouts.scores),
      tf.ones_like(rollouts.scores) * -1e8)

  # [batch_size, 1]
  sample_idx = tf.distributions.Categorical(logits=masked_scores).sample()
  sample_idx = tf.reshape(sample_idx, [batch_size, 1])

  # [batch_size]
  reward = tf.reshape(tensor_utils.gather(rewards, sample_idx), [-1])

  # [batch_size, rollout_length]
  token_ids = tf.reshape(
      tensor_utils.gather(rollouts.token_ids, sample_idx), [batch_size, -1])
  mask = tf.reshape(
      tensor_utils.gather(rollouts.mask, sample_idx), [batch_size, -1])
  segment_ids = tf.zeros_like(token_ids)
  positions = tf.tile(
      tf.expand_dims(tf.range(rollout_length), 0), [batch_size, 1])

  # Create text input.
  rollout = text_utils.TextInputs(
      token_ids=token_ids,
      mask=mask,
      segment_ids=segment_ids,
      positions=positions)

  return rollout, reward


# ------------------------------------------------------------------------------
#
# Miscellaneous QA model prediction.
#
# ------------------------------------------------------------------------------


def write_spans(caption_file, question_file, output_file, vocab, params):
  """Write spans to disk using RC model.

  Args:
    caption_file: Path to caption predictions.
    question_file: Path to question.
    output_file: Path to write spans selected by the QA model.
    vocab: Instance of text_utils.Vocab.
    params: Model parameters.
  """

  # Wrapper for QA model.
  def model_fn(features, labels, mode, params):
    """A model function satisfying the tf.estimator API."""
    del labels
    assert mode == tf_estimator.ModeKeys.PREDICT, "Mode should be PREDICT."
    span, _ = rc_span(
        question_ids=features["question_inputs"].token_ids,
        question_mask=features["question_inputs"].mask,
        context_ids=features["context_inputs"].token_ids,
        context_mask=features["context_inputs"].mask,
        rc_model=hub.Module(params["rc_model"]),
        vocab=vocab,
        no_answer_bias=params.get("no_answer_bias", -1e4))
    return tf_estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=dict(
            question_id=features["question_id"],
            image_id=features["image_id"],
            beam_id=features["beam_id"],
            span=span))

  # Get estimator.
  params = copy.copy(params)
  params["model_dir"] = experiment_utils.get_tempdir()
  estimator = experiment_utils.get_estimator(model_fn, params)

  # Write predictions.
  tf.logging.info("Writing predictions to disk...")
  tf.io.gfile.makedirs(os.path.dirname(output_file))
  with tf.io.gfile.GFile(output_file, "w") as f:
    iterator = estimator.predict(
        input_fn=functools.partial(
            rc_dataset.get_dataset,
            caption_file=caption_file,
            qa_file=question_file,
            scratch_file=experiment_utils.get_tempfile(),
            vocab=vocab),
        yield_single_examples=True)
    i = 0
    for ex in iterator:
      i += 1
      f.write(json.dumps(ex, cls=io_utils.NumpyEncoder) + "\n")
      if i % 1000 == 0:
        tf.logging.info("Wrote %d predictions", i)
    tf.logging.info("Done. Wrote %d predictions.", i)

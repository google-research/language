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
"""Main file for running a Natural Questions long-answers experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags

from language.common.inputs import dataset_utils
from language.common.inputs import embedding_utils
from language.common.utils import tensor_utils
from language.question_answering.datasets import nq_long_dataset
from language.question_answering.models import nq_long_decatt_model
from language.question_answering.utils import nq_long_utils

import tensorflow as tf

flags.DEFINE_string("embeddings_path", None, "Path to pretrained embeddings.")
flags.DEFINE_integer("max_vocab_size", 4000000, "Maximum vocab size.")

flags.DEFINE_integer("max_contexts", 50, "Maximum number of contexts.")

flags.DEFINE_integer("max_context_len", 1000, "Maximum length of the contexts.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")

flags.DEFINE_integer("num_epochs", 30, "Number of training epochs.")

flags.DEFINE_float("null_weight", 0.3, "Weight for examples with null labels")

FLAGS = flags.FLAGS


def model_function(features, labels, mode, params, embeddings):
  """A model function satisfying the tf.estimator API.

  Args:
    features: Dictionary of feature tensors with keys:
        - question_tok: <string> [batch_size, max_question_len]
        - context_tok: <string> [batch_size, max_num_context, max_context_len]
        - question_tok_len: <int32> [batch_size]
        - num_context: <int32> [batch_size]
        - context_tok_len: <int32> [batch_size]
        - question_tok_wid: <int32> [batch_size, max_question_len]
        - context_tok_wid: <int32> [batch_size, max_num_context,
          max_context_len]
         - long_answer_indices: <int32> [batch_size]
    labels: <int32> [batch_size] for answer index (-1 = NULL).
    mode: One of the keys from tf.estimator.ModeKeys.
    params: Dictionary of hyperparameters.
    embeddings: An embedding_utils.PretrainedWordEmbeddings object.

  Returns:
    estimator_spec: A tf.estimator.EstimatorSpec object.
  """
  del params  # Unused.

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Add a dummy batch dimension if we are exporting the predictor.
    features = {k: tf.expand_dims(v, 0) for k, v in features.items()}

  embedding_weights, embedding_scaffold = embeddings.get_params(trainable=False)

  # Features.
  question_tok_len = features["question_tok_len"]
  question_tok_wid = features["question_tok_wid"]
  context_tok_wid = features["context_tok_wid"]
  num_context = features["num_context"]
  context_tok_len = features["context_tok_len"]

  # Truncate the contexts and labels to a certain maximum length.
  context_tok_wid, num_context, context_tok_len = (
      nq_long_utils.truncate_contexts(
          context_token_ids=context_tok_wid,
          num_contexts=num_context,
          context_len=context_tok_len,
          max_contexts=FLAGS.max_contexts,
          max_context_len=FLAGS.max_context_len))

  non_null_context_scores = nq_long_decatt_model.build_model(
      question_tok_wid=question_tok_wid,
      question_lens=question_tok_len,
      context_tok_wid=context_tok_wid,
      context_lens=context_tok_len,
      embedding_weights=embedding_weights,
      mode=mode)

  # Mask out contexts that are padding.
  num_context_mask = tf.log(
      tf.sequence_mask(
          num_context,
          tensor_utils.shape(non_null_context_scores, 1),
          dtype=tf.float32))
  non_null_context_scores += num_context_mask

  # <float> [batch_size, 1]
  null_score = tf.zeros([tf.shape(question_tok_wid)[0], 1])

  # Offset everything by 1 to account for null context.
  # [batch_size, 1 + max_contexts]
  context_scores = tf.concat([null_score, non_null_context_scores], 1)

  if mode != tf.estimator.ModeKeys.PREDICT:
    labels = nq_long_utils.truncate_labels(labels, FLAGS.max_contexts)

    # In the data, NULL is given index -1 but this is not compatible with
    # softmax so shift by 1.
    labels = labels + 1

    # Reweight null examples.
    weights = nq_long_utils.compute_null_weights(labels, FLAGS.null_weight)

    # When computing the loss we take only the first label.
    loss_labels = labels[:, 0]

    # []
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=loss_labels, logits=context_scores, weights=weights)

    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # <int32> [batch_size]
    eval_predictions = tf.to_int32(tf.argmax(context_scores, 1))

    non_null_match, non_null_gold, non_null_predictions = (
        nq_long_utils.compute_match_stats(eval_predictions, labels))

    precision, precision_op = (
        tf.metrics.mean(non_null_match, weights=non_null_predictions))
    recall, recall_op = (tf.metrics.mean(non_null_match, weights=non_null_gold))

    f1, f1_op = (
        nq_long_utils.f1_metric(
            precision=precision,
            precision_op=precision_op,
            recall=recall,
            recall_op=recall_op))

    # Bogus metric until we figure out how to connect Ming Wei's eval code.
    eval_metric_ops = {
        "precision": (precision, precision_op),
        "recall": (recall, recall_op),
        "f1": (f1, f1_op)
    }
  else:
    loss = None
    train_op = None
    eval_metric_ops = {}

  # In the export, we never predict NULL since the eval metric will compute the
  # best possible F1.
  export_long_answer_idx = tf.to_int32(tf.argmax(non_null_context_scores, 1))
  export_long_answer_score = tf.reduce_max(non_null_context_scores, 1)
  predictions = dict(idx=export_long_answer_idx, score=export_long_answer_score)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Remove the dummy batch dimension if we are exporting the predictor.
    predictions = {k: tf.squeeze(v, 0) for k, v in predictions.items()}

  estimator_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      predictions=predictions,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      scaffold=embedding_scaffold)

  return estimator_spec


def _string_to_tokens_dataset_mapper(keys_to_map, suffix="_tok"):
  """Wrapper for mapper that tokenizes and truncates by length."""

  def _mapper(dataset):
    """Tokenizes strings using tf.string_split and truncates by length."""
    for k in keys_to_map:
      # pylint: disable=g-explicit-length-test
      if len(dataset[k].get_shape()) == 0:  # Used for questions.
        # pylint: enable=g-explicit-length-test
        # <string> [num_tokens]
        tokens = tf.string_split([dataset[k]]).values
      else:  # Used for contexts.
        # <string> [num_context, num_tokens] (sparse)
        sparse_tokens = tf.string_split(dataset[k])

        # <string>[num_tokens, max_num_tokens] (dense)
        tokens = tf.sparse_tensor_to_dense(sparse_tokens, default_value="")

      dataset[k + suffix] = tokens
      # Compute exact length of each context.
      dataset[k + suffix + "_len"] = tf.count_nonzero(
          tokens, axis=-1, dtype=tf.int32)
    return dataset

  return _mapper


def _num_context_mapper(keys_to_map, prefix="num_"):
  """Wrapper for mapper that computes number of contexts."""

  def _mapper(dataset):
    """Computes number of contexts."""
    for k in keys_to_map:
      size = tf.shape(dataset[k])[-1]
      dataset[prefix + k] = size
    return dataset

  return _mapper


def serving_input_receiver_function(embeddings):
  """Returns a placeholder-driven input_fn for SavedModel."""
  placeholders = dict(
      question=tf.placeholder(dtype=tf.string, shape=[], name="question"),
      context=tf.placeholder(dtype=tf.string, shape=[None], name="context"))
  features = preprocess_mapper(placeholders, embeddings.get_lookup_table())
  return tf.estimator.export.ServingInputReceiver(features, placeholders)


def compare_metrics(best_eval_result, current_eval_result):
  """Compares two evaluation results."""
  # Bad evaluation comparison. Will be replaced by official evaluation.
  return best_eval_result["f1"] < current_eval_result["f1"]


def preprocess_mapper(features, lookup_table):
  """Model-specific preprocessing of features from the dataset."""
  features = _num_context_mapper(["context"])(features)
  features = _string_to_tokens_dataset_mapper(["question", "context"])(features)

  # Add the word IDs to the dataset ("question_wid" and "context_wid").
  features = dataset_utils.string_to_int_mapper(["question_tok", "context_tok"],
                                                mapping=lookup_table,
                                                suffix="_wid")(
                                                    features)

  return features


def input_function(is_train, embeddings):
  """An input function satisfying the tf.estimator API."""
  # A dataset with keys `question`, `context`, `answer_start`, and `answer_end`.
  dataset = nq_long_dataset.get_dataset(is_train)

  dataset = dataset.map(
      functools.partial(
          preprocess_mapper, lookup_table=embeddings.get_lookup_table()),
      num_parallel_calls=12)

  if is_train:
    dataset = dataset.repeat(FLAGS.num_epochs)
    dataset = dataset.shuffle(10000)

  dataset = dataset.padded_batch(
      batch_size=FLAGS.batch_size, padded_shapes=dataset.output_shapes)

  # The tf.estimator API expects the dataset to be a (features, labels) pair.
  dataset = dataset.map(lambda d: (d, d["long_answer_indices"]))

  dataset = dataset.prefetch(1)
  return dataset


def experiment_functions():
  """Get the necessary functions to run an experiment."""
  # Build the memory-intensive embeddings once and enclose them in all the
  # functions that use it.
  embeddings = embedding_utils.PretrainedWordEmbeddings(
      embeddings_path=FLAGS.embeddings_path,
      max_vocab_size=FLAGS.max_vocab_size,
      num_oov_buckets=1000,
      lowercase=True)
  model_fn = functools.partial(model_function, embeddings=embeddings)
  train_input_fn = functools.partial(
      input_function, is_train=True, embeddings=embeddings)
  eval_input_fn = functools.partial(
      input_function, is_train=False, embeddings=embeddings)
  serving_input_receiver_fn = functools.partial(
      serving_input_receiver_function, embeddings=embeddings)
  return (model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn)

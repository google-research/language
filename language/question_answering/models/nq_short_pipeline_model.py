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
"""Main file for running a Natural Questions short-answers experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from absl import flags

from language.common.inputs import char_utils
from language.common.inputs import dataset_utils
from language.common.inputs import embedding_utils
from language.common.layers import common_layers
from language.common.utils import tensor_utils
from language.question_answering.datasets import nq_short_pipeline_dataset
from language.question_answering.layers import document_reader
from language.question_answering.utils import span_utils
import tensorflow as tf

flags.DEFINE_string("embeddings_path", None, "Path to pretrained embeddings.")
flags.DEFINE_integer("max_vocab_size", 100000, "Maximum vocab size.")
flags.DEFINE_integer("char_emb_size", 8, "Embedding size of each character.")
flags.DEFINE_integer("char_kernel_width", 5, "Width of character convolutions.")
flags.DEFINE_integer("num_char_filters", 100, "Number of character filters.")
flags.DEFINE_integer("max_context_len", 1024, "Maximum length of the contexts.")
flags.DEFINE_integer("total_batch_size", 32768, "Batch size wrt. total tokens.")
flags.DEFINE_float("dropout_ratio", 0.5, "Dropout ratio.")
flags.DEFINE_integer("num_layers", 5, "Number of LSTM layers.")
flags.DEFINE_integer("hidden_size", 256, "Size of hidden layers.")
flags.DEFINE_integer("num_epochs", 30, "Number of training epochs.")

FLAGS = flags.FLAGS


def model_function(features, labels, mode, params, embeddings):
  """A model function satisfying the tf.estimator API.

  Args:
    features: Dictionary of feature tensors with keys:
        - question: <string> [batch_size, max_question_len]
        - question_len: <int32> [batch_size]
        - question_cid: <int32> [batch_size, max_question_len, max_chars]
        - question_wid: <int32> [batch_size, max_question_len]
        - context: <string> [batch_size, max_context_len]
        - context_len: <int32> [batch_size]
        - context_cid: <int32> [batch_size, max_context_len, max_chars]
        - context_wid: <int32> [batch_size, max_context_len]
        - answer_start: <int32> [batch_size]
        - answer_end: <int32> [batch_size]
    labels: Pair of tensors containing the answer start and answer end.
    mode: One of the keys from tf.estimator.ModeKeys.
    params: Unused parameter dictionary.
    embeddings: An embedding_utils.PretrainedWordEmbeddings object.

  Returns:
    estimator_spec: A tf.estimator.EstimatorSpec object.
  """
  del params

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Add a dummy batch dimension if we are exporting the predictor.
    features = {k: tf.expand_dims(v, 0) for k, v in features.items()}

  embedding_weights, embedding_scaffold = embeddings.get_params(trainable=False)

  def _embed(prefix):
    """Embed the input text based and word and character IDs."""
    word_emb = tf.nn.embedding_lookup(embedding_weights,
                                      features[prefix + "_wid"])
    char_emb = common_layers.character_cnn(
        char_ids=features[prefix + "_cid"],
        emb_size=FLAGS.char_emb_size,
        kernel_width=FLAGS.char_kernel_width,
        num_filters=FLAGS.num_char_filters)
    concat_emb = tf.concat([word_emb, char_emb], -1)

    if mode == tf.estimator.ModeKeys.TRAIN:
      concat_emb = tf.nn.dropout(concat_emb, 1.0 - FLAGS.dropout_ratio)
    return concat_emb

  with tf.variable_scope("embed"):
    # [batch_size, max_question_len, hidden_size]
    question_emb = _embed("question")

  with tf.variable_scope("embed", reuse=True):
    # [batch_size, max_context_len, hidden_size]
    context_emb = _embed("context")

  # [batch_size, max_context_len]
  start_logits, end_logits = document_reader.score_endpoints(
      question_emb=question_emb,
      question_len=features["question_len"],
      context_emb=context_emb,
      context_len=features["context_len"],
      hidden_size=FLAGS.hidden_size,
      num_layers=FLAGS.num_layers,
      dropout_ratio=FLAGS.dropout_ratio,
      mode=mode,
      use_cudnn=False if mode == tf.estimator.ModeKeys.PREDICT else None)

  if mode != tf.estimator.ModeKeys.PREDICT:
    # [batch_size]
    start_labels, end_labels = labels

    # Since we truncate long contexts, some of the labels will not be
    # recoverable. In that case, we mask these invalid labels.
    valid_start_labels = tf.less(start_labels, features["context_len"])
    valid_end_labels = tf.less(end_labels, features["context_len"])
    tf.summary.histogram("valid_start_labels", tf.to_float(valid_start_labels))
    tf.summary.histogram("valid_end_labels", tf.to_float(valid_end_labels))

    dummy_labels = tf.zeros_like(start_labels)

    # []
    start_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.where(valid_start_labels, start_labels, dummy_labels),
        logits=start_logits,
        weights=tf.to_float(valid_start_labels),
        reduction=tf.losses.Reduction.MEAN)
    end_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.where(valid_end_labels, end_labels, dummy_labels),
        logits=end_logits,
        weights=tf.to_float(valid_end_labels),
        reduction=tf.losses.Reduction.MEAN)
    loss = start_loss + end_loss
  else:
    loss = None

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(
        grads_and_vars=zip(gradients, variables),
        global_step=tf.train.get_global_step())
  else:
    # Don't build the train_op unnecessarily, since the ADAM variables can cause
    # problems with loading checkpoints on CPUs.
    train_op = None

  batch_size, max_context_len = tensor_utils.shape(features["context_wid"])
  tf.summary.histogram("batch_size", batch_size)
  tf.summary.histogram("non_padding", features["context_len"] / max_context_len)

  # [batch_size], [batch_size]
  start_predictions, end_predictions, predicted_score = (
      span_utils.max_scoring_span(start_logits, end_logits))

  # [batch_size, 2]
  predictions = dict(
      start_idx=start_predictions,
      end_idx=(end_predictions + 1),
      score=predicted_score)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Remove the dummy batch dimension if we are exporting the predictor.
    predictions = {k: tf.squeeze(v, 0) for k, v in predictions.items()}

  if mode == tf.estimator.ModeKeys.EVAL:
    text_summary = get_text_summary(
        question=features["question"],
        context=features["context"],
        start_predictions=start_predictions,
        end_predictions=end_predictions)

    # TODO(kentonl): Replace this with @mingweichang's official eval script.
    exact_match = tf.logical_and(
        tf.equal(start_predictions, start_labels),
        tf.equal(end_predictions, end_labels))

    eval_metric_ops = dict(
        exact_match=tf.metrics.mean(exact_match),
        text_summary=(text_summary, tf.no_op()))
  else:
    eval_metric_ops = None

  estimator_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      predictions=predictions,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      scaffold=embedding_scaffold)

  return estimator_spec


def get_text_summary(question, context, start_predictions, end_predictions):
  """Get a text summary of the question and the predicted answer."""
  question_text = tf.reduce_join(question, -1, separator=" ")

  def _get_prediction_text(args, window=5):
    """Get the prediction text for a single row in the batch."""
    current_context, start, end = args
    prediction_context_start = tf.maximum(start - window, 0)
    prediction_context_end = tf.minimum(end + 1 + window,
                                        tf.shape(current_context)[0])
    before = current_context[prediction_context_start:start]
    prediction = current_context[start:end + 1]
    after = current_context[end + 1:prediction_context_end]
    concat = tf.concat([before, ["**"], prediction, ["**"], after], 0)
    return tf.reduce_join(concat, separator=" ")

  prediction_text = tf.map_fn(
      fn=_get_prediction_text,
      elems=[context, start_predictions, end_predictions],
      dtype=tf.string,
      back_prop=False)

  return tf.summary.text("predictions",
                         tf.stack([question_text, prediction_text], -1))


def preprocess_mapper(features, lookup_table):
  """Model-specific preprocessing of features from the dataset."""
  # Truncate contexts that are too long.
  features["context"] = features["context"][:FLAGS.max_context_len]

  # Add the input lengths to the dataset ("question_len" and "context_len").
  features = dataset_utils.length_mapper(["question", "context"])(features)

  # Add the word IDs to the dataset ("question_wid" and "context_wid").
  features = dataset_utils.string_to_int_mapper(["question", "context"],
                                                mapping=lookup_table,
                                                suffix="_wid")(
                                                    features)

  # Add the character IDs to the dataset ("question_cid" and "context_cid").
  features = char_utils.token_to_char_ids_mapper(["question", "context"])(
      features)
  return features


def input_function(is_train, embeddings):
  """An input function satisfying the tf.estimator API."""
  # A dataset with keys `question`, `context`, `answer_start`, and `answer_end`.
  dataset = nq_short_pipeline_dataset.get_dataset(is_train)

  dataset = dataset.map(
      partial(preprocess_mapper, lookup_table=embeddings.get_lookup_table()),
      num_parallel_calls=12)

  if is_train:
    dataset = dataset.repeat(FLAGS.num_epochs)
    dataset = dataset.shuffle(10000)

    # Compute the batch size of each bucket such that the total number of
    # context words in each batch is bounded by FLAGS.total_batch_size during
    # training.
    bucket_boundaries = range(128, FLAGS.max_context_len, 128)
    bucket_batch_sizes = [
        int(FLAGS.total_batch_size / b) for b in bucket_boundaries
    ]
    bucket_batch_sizes.append(
        int(FLAGS.total_batch_size / FLAGS.max_context_len))
    dataset = dataset.apply(
        tf.contrib.data.bucket_by_sequence_length(
            element_length_func=lambda d: d["context_len"],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=dataset.output_shapes))
  else:
    # Don't do any bucketing during evals to avoid biased sampling.
    dataset = dataset.padded_batch(
        batch_size=int(FLAGS.total_batch_size / FLAGS.max_context_len),
        padded_shapes=dataset.output_shapes)

  # The tf.estimator API expects the dataset to be a (features, labels) pair.
  dataset = dataset.map(lambda d: (d, (d["answer_start"], d["answer_end"])))
  dataset = dataset.prefetch(10)
  return dataset


def serving_input_receiver_function(embeddings):
  """Returns a placeholder-driven input_fn for SavedModel."""
  placeholders = dict(
      question=tf.placeholder(dtype=tf.string, shape=[None], name="question"),
      context=tf.placeholder(dtype=tf.string, shape=[None], name="context"))
  features = preprocess_mapper(placeholders, embeddings.get_lookup_table())
  return tf.estimator.export.ServingInputReceiver(features, placeholders)


def compare_metrics(best_eval_result, current_eval_result):
  """Compares two evaluation results."""
  return best_eval_result["exact_match"] < current_eval_result["exact_match"]


def experiment_functions():
  """Get the necessary functions to run an experiment."""
  # Build the memory-intensive embeddings once and enclose them in all the
  # functions that use it.
  embeddings = embedding_utils.PretrainedWordEmbeddings(
      embeddings_path=FLAGS.embeddings_path,
      max_vocab_size=FLAGS.max_vocab_size,
      lowercase=True)
  model_fn = partial(model_function, embeddings=embeddings)
  train_input_fn = partial(input_function, is_train=True, embeddings=embeddings)
  eval_input_fn = partial(input_function, is_train=False, embeddings=embeddings)
  serving_input_receiver_fn = partial(
      serving_input_receiver_function, embeddings=embeddings)
  return model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn

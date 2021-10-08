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
# Lint as: python3
"""Retrieve-and-read model for text classification purposes."""

from bert import optimization
from language.common.utils import exporters
from language.orqa import ops as orqa_ops
from language.orqa.datasets import text_classification_dataset
from language.orqa.models import orqa_model
from language.orqa.utils import bert_utils
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def read(features, retriever_logits, blocks, mode, params):
  """Do reading."""
  tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(
      params["reader_module_path"])

  (orig_tokens, block_token_map, block_token_ids, blocks) = (
      bert_utils.tokenize_with_original_mapping(blocks, tokenizer))

  # NOTE: we assume that the batch size is 1.
  question_token_ids = tokenizer.tokenize(
      tf.expand_dims(features["question"], 0))
  question_token_ids = tf.cast(question_token_ids.flat_values, tf.int32)

  orig_tokens = orig_tokens.to_tensor(default_value="")
  block_lengths = tf.cast(block_token_ids.row_lengths(), tf.int32)
  block_token_ids = tf.cast(block_token_ids.to_tensor(), tf.int32)
  block_token_map = tf.cast(block_token_map.to_tensor(), tf.int32)

  fake_answer = tf.constant([""])
  answer_token_ids = tokenizer.tokenize(fake_answer).merge_dims(1, 2)
  answer_lengths = tf.cast(answer_token_ids.row_lengths(), tf.int32)
  answer_token_ids = tf.cast(answer_token_ids.to_tensor(), tf.int32)

  cls_token_id = vocab_lookup_table.lookup(tf.constant("[CLS]"))
  sep_token_id = vocab_lookup_table.lookup(tf.constant("[SEP]"))

  concat_inputs = orqa_ops.reader_inputs(
      question_token_ids=question_token_ids,
      block_token_ids=block_token_ids,
      block_lengths=block_lengths,
      block_token_map=block_token_map,
      answer_token_ids=answer_token_ids,
      answer_lengths=answer_lengths,
      cls_token_id=tf.cast(cls_token_id, tf.int32),
      sep_token_id=tf.cast(sep_token_id, tf.int32),
      max_sequence_len=params["reader_seq_len"])

  tf.summary.scalar("reader_nonpad_ratio",
                    tf.reduce_mean(tf.cast(concat_inputs.mask, tf.float32)))

  reader_module = hub.Module(
      params["reader_module_path"],
      tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
      trainable=True)

  concat_outputs = reader_module(
      dict(
          input_ids=concat_inputs.token_ids,
          input_mask=concat_inputs.mask,
          segment_ids=concat_inputs.segment_ids),
      signature="tokens",
      as_dict=True)

  with tf.variable_scope("pooler"):
    # [reader_beam_size, hidden_size]
    first_token_tensor = concat_outputs["pooled_output"]

    kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # [reader_beam_size, num_classes]
    reader_logits = tf.layers.dense(
        first_token_tensor,
        params["num_classes"],
        kernel_initializer=kernel_initializer)

  final_logits = reader_logits + tf.expand_dims(retriever_logits, -1)

  return final_logits


def compute_loss(labels, final_logits):
  """Compute loss."""
  logits_per_correct_class = tf.gather(final_logits, labels, axis=1)
  log_numerator = tf.reduce_logsumexp(logits_per_correct_class)
  log_denominator = tf.reduce_logsumexp(final_logits)
  loss = log_denominator - log_numerator
  tf.summary.scalar("loss", loss)
  return loss


def compute_eval_metrics(labels, predictions):
  """Compute eval metrics."""
  mask = tf.ones_like(labels)
  eval_metric_ops = {
      "accuracy":
          tf.metrics.accuracy(labels=labels, predictions=predictions),
      "total":  # We use true_positives here simply as a counter.
          tf.metrics.true_positives(labels=mask, predictions=mask),
  }
  return eval_metric_ops


def model_fn(features, labels, mode, params):
  """Model function."""
  reader_beam_size = params["reader_beam_size"]
  num_classes = params["num_classes"]
  if mode == tf.estimator.ModeKeys.PREDICT:
    retriever_beam_size = reader_beam_size
  else:
    retriever_beam_size = params["retriever_beam_size"]
  assert reader_beam_size <= retriever_beam_size

  with tf.device("/cpu:0"):
    retriever_outputs = orqa_model.retrieve(
        features=features,
        retriever_beam_size=retriever_beam_size,
        mode=mode,
        params=params)

  with tf.variable_scope("reader"):
    # [reader_beam_size, num_classes]
    final_logits = read(
        features=features,
        retriever_logits=retriever_outputs.logits[:reader_beam_size],
        blocks=retriever_outputs.blocks[:reader_beam_size],
        mode=mode,
        params=params)

  # [reader_beam_size]
  # We pick the most confident prediction amongst all retrievals.
  predictions = tf.argmax(
      tf.reshape(final_logits, [reader_beam_size * num_classes]))
  predictions = tf.math.floormod(predictions, num_classes)

  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
    train_op = None
    eval_metric_ops = None
  else:
    labels = tf.cast(labels, tf.int32)

    eval_metric_ops = compute_eval_metrics(
        labels=labels, predictions=predictions)

    loss = compute_loss(labels, final_logits)

    train_op = optimization.create_optimizer(
        loss=loss,
        init_lr=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=min(10000, max(100,
                                        int(params["num_train_steps"] / 10))),
        use_tpu=False)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions=predictions,
      eval_metric_ops=eval_metric_ops)


def exporter():
  """Create exporters."""
  return exporters.BestSavedModelAndCheckpointExporter(
      eval_spec_name="default",
      serving_input_receiver_fn=orqa_model.serving_fn,
      metric_name="accuracy")


def input_fn(is_train, name, params):
  """An input function satisfying the tf.estimator API."""
  dataset = text_classification_dataset.get_dataset(
      data_root=params["data_root"],
      name=name,
      split="train" if is_train else "dev")
  if is_train:
    dataset = dataset.repeat()

  def _extract_labels(d):
    return d, d.pop("answers")

  dataset = dataset.map(_extract_labels)
  dataset = dataset.prefetch(10)
  return dataset

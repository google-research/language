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
"""ORQA model."""
import collections
import json
import os

from bert import optimization
from language.common.utils import exporters
from language.common.utils import tensor_utils
from language.orqa import ops as orqa_ops
from language.orqa.datasets import orqa_dataset
from language.orqa.utils import bert_utils
from language.orqa.utils import eval_utils
from language.orqa.utils import scann_utils
import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from official.nlp.data import squad_lib

RetrieverOutputs = collections.namedtuple("RetrieverOutputs",
                                          ["logits", "blocks"])
ReaderOutputs = collections.namedtuple("ReaderOutputs", [
    "logits", "candidate_starts", "candidate_ends", "candidate_orig_starts",
    "candidate_orig_ends", "blocks", "orig_blocks", "orig_tokens", "token_ids",
    "gold_starts", "gold_ends"
])


def span_candidates(masks, max_span_width):
  """Generate span candidates.

  Args:
    masks: <int32> [num_retrievals, max_sequence_len]
    max_span_width: int

  Returns:
    starts: <int32> [num_spans]
    ends: <int32> [num_spans]
    span_masks: <int32> [num_retrievals, num_spans]
  """
  _, max_sequence_len = tensor_utils.shape(masks)
  def _spans_given_width(width):
    current_starts = tf.range(max_sequence_len - width + 1)
    current_ends = tf.range(width - 1, max_sequence_len)
    return current_starts, current_ends

  starts, ends = zip(*(_spans_given_width(w + 1)
                       for w in range(max_span_width)))

  # [num_spans]
  starts = tf.concat(starts, 0)
  ends = tf.concat(ends, 0)

  # [num_retrievals, num_spans]
  start_masks = tf.gather(masks, starts, axis=-1)
  end_masks = tf.gather(masks, ends, axis=-1)
  span_masks = start_masks * end_masks

  return starts, ends, span_masks


def mask_to_score(mask):
  return (1.0 - tf.cast(mask, tf.float32)) * -10000.0


def marginal_log_loss(logits, is_correct):
  """Loss based on the negative marginal log-likelihood."""
  # []
  log_numerator = tf.reduce_logsumexp(logits + mask_to_score(is_correct), -1)
  log_denominator = tf.reduce_logsumexp(logits, -1)
  return log_denominator - log_numerator


def retrieve(features, retriever_beam_size, mode, params):
  """Do retrieval."""
  tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(
      params["retriever_module_path"])

  question_token_ids = tokenizer.tokenize(
      tf.expand_dims(features["question"], 0))
  question_token_ids = tf.cast(
      question_token_ids.merge_dims(1, 2).to_tensor(), tf.int32)
  cls_token_id = vocab_lookup_table.lookup(tf.constant("[CLS]"))
  sep_token_id = vocab_lookup_table.lookup(tf.constant("[SEP]"))
  question_token_ids = tf.concat(
      [[[tf.cast(cls_token_id, tf.int32)]], question_token_ids,
       [[tf.cast(sep_token_id, tf.int32)]]], -1)

  retriever_module = hub.Module(
      params["retriever_module_path"],
      tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
      trainable=True)

  # [1, projection_size]
  question_emb = retriever_module(
      inputs=dict(
          input_ids=question_token_ids,
          input_mask=tf.ones_like(question_token_ids),
          segment_ids=tf.zeros_like(question_token_ids)),
      signature="projected")

  block_emb, searcher = scann_utils.load_scann_searcher(
      var_name="block_emb",
      checkpoint_path=os.path.join(params["retriever_module_path"], "encoded",
                                   "encoded.ckpt"),
      num_neighbors=retriever_beam_size)

  # [1, retriever_beam_size]
  retrieved_block_ids, _ = searcher.search_batched(question_emb)

  # [1, retriever_beam_size, projection_size]
  retrieved_block_emb = tf.gather(block_emb, retrieved_block_ids)

  # [retriever_beam_size]
  retrieved_block_ids = tf.squeeze(retrieved_block_ids)

  # [retriever_beam_size, projection_size]
  retrieved_block_emb = tf.squeeze(retrieved_block_emb)

  # [1, retriever_beam_size]
  retrieved_logits = tf.matmul(
      question_emb, retrieved_block_emb, transpose_b=True)

  # [retriever_beam_size]
  retrieved_logits = tf.squeeze(retrieved_logits, 0)

  blocks_dataset = tf.data.TFRecordDataset(
      params["block_records_path"], buffer_size=512 * 1024 * 1024)
  blocks_dataset = blocks_dataset.batch(
      params["num_block_records"], drop_remainder=True)
  blocks = tf.get_local_variable(
      "blocks",
      initializer=tf.data.experimental.get_single_element(blocks_dataset))
  retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
  return RetrieverOutputs(logits=retrieved_logits, blocks=retrieved_blocks)


def read(features, retriever_logits, blocks, mode, params, labels):
  """Do reading."""
  tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(
      params["reader_module_path"])

  orig_blocks = blocks

  (orig_tokens, block_token_map, block_token_ids, blocks) = (
      bert_utils.tokenize_with_original_mapping(blocks, tokenizer))

  question_token_ids = tokenizer.tokenize(
      tf.expand_dims(features["question"], 0))
  question_token_ids = tf.cast(question_token_ids.flat_values, tf.int32)

  orig_tokens = orig_tokens.to_tensor(default_value="")
  block_lengths = tf.cast(block_token_ids.row_lengths(), tf.int32)
  block_token_ids = tf.cast(block_token_ids.to_tensor(), tf.int32)
  block_token_map = tf.cast(block_token_map.to_tensor(), tf.int32)

  answer_token_ids = tokenizer.tokenize(labels).merge_dims(1, 2)
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

  concat_token_emb = concat_outputs["sequence_output"]

  # [num_spans], [num_spans], [reader_beam_size, num_spans]
  candidate_starts, candidate_ends, candidate_mask = span_candidates(
      concat_inputs.block_mask, params["max_span_width"])

  # Score with an MLP to enable start/end interaction:
  # score(s, e) = w·σ(w_s·h_s + w_e·h_e)
  kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)

  # [reader_beam_size, max_sequence_len, span_hidden_size * 2]
  projection = tf.layers.dense(
      concat_token_emb,
      params["span_hidden_size"] * 2,
      kernel_initializer=kernel_initializer)

  # [reader_beam_size, max_sequence_len, span_hidden_size]
  start_projection, end_projection = tf.split(projection, 2, -1)

  # [reader_beam_size, num_candidates, span_hidden_size]
  candidate_start_projections = tf.gather(
      start_projection, candidate_starts, axis=1)
  candidate_end_projection = tf.gather(end_projection, candidate_ends, axis=1)
  candidate_hidden = candidate_start_projections + candidate_end_projection

  candidate_hidden = tf.nn.relu(candidate_hidden)
  candidate_hidden = tf.keras.layers.LayerNormalization(axis=-1)(
      candidate_hidden)

  # [reader_beam_size, num_candidates, 1]
  reader_logits = tf.layers.dense(
      candidate_hidden, 1, kernel_initializer=kernel_initializer)

  # [reader_beam_size, num_candidates]
  reader_logits = tf.squeeze(reader_logits)
  reader_logits += mask_to_score(candidate_mask)
  reader_logits += tf.expand_dims(retriever_logits, -1)

  # [reader_beam_size, num_candidates]
  candidate_orig_starts = tf.gather(
      params=concat_inputs.token_map, indices=candidate_starts, axis=-1)
  candidate_orig_ends = tf.gather(
      params=concat_inputs.token_map, indices=candidate_ends, axis=-1)

  return ReaderOutputs(
      logits=reader_logits,
      candidate_starts=candidate_starts,
      candidate_ends=candidate_ends,
      candidate_orig_starts=candidate_orig_starts,
      candidate_orig_ends=candidate_orig_ends,
      blocks=blocks,
      orig_blocks=orig_blocks,
      orig_tokens=orig_tokens,
      token_ids=concat_inputs.token_ids,
      gold_starts=concat_inputs.gold_starts,
      gold_ends=concat_inputs.gold_ends)


def get_predictions(reader_outputs, params):
  """Get predictions."""
  tokenization_info = bert_utils.get_tokenization_info(
      params["reader_module_path"])
  with tf.io.gfile.GFile(tokenization_info["vocab_file"]) as vocab_file:
    vocab = tf.constant([l.strip() for l in vocab_file.readlines()])

  # []
  predicted_block_index = tf.argmax(tf.reduce_max(reader_outputs.logits, 1))
  predicted_candidate = tf.argmax(tf.reduce_max(reader_outputs.logits, 0))

  predicted_block = tf.gather(reader_outputs.blocks, predicted_block_index)
  predicted_orig_block = tf.gather(reader_outputs.orig_blocks,
                                   predicted_block_index)
  predicted_orig_tokens = tf.gather(reader_outputs.orig_tokens,
                                    predicted_block_index)
  predicted_orig_start = tf.gather(
      tf.gather(reader_outputs.candidate_orig_starts, predicted_block_index),
      predicted_candidate)
  predicted_orig_end = tf.gather(
      tf.gather(reader_outputs.candidate_orig_ends, predicted_block_index),
      predicted_candidate)
  predicted_orig_answer = tf.reduce_join(
      predicted_orig_tokens[predicted_orig_start:predicted_orig_end + 1],
      separator=" ")

  predicted_token_ids = tf.gather(reader_outputs.token_ids,
                                  predicted_block_index)
  predicted_tokens = tf.gather(vocab, predicted_token_ids)
  predicted_start = tf.gather(reader_outputs.candidate_starts,
                              predicted_candidate)
  predicted_end = tf.gather(reader_outputs.candidate_ends, predicted_candidate)
  predicted_normalized_answer = tf.reduce_join(
      predicted_tokens[predicted_start:predicted_end + 1], separator=" ")

  def _get_final_text(pred_text, orig_text):
    pred_text = six.ensure_text(pred_text, errors="ignore")
    orig_text = six.ensure_text(orig_text, errors="ignore")
    return squad_lib.get_final_text(
        pred_text=pred_text,
        orig_text=orig_text,
        do_lower_case=tokenization_info["do_lower_case"])

  predicted_answer = tf.py_func(
      func=_get_final_text,
      inp=[predicted_normalized_answer, predicted_orig_answer],
      Tout=tf.string)

  return dict(
      block_index=predicted_block_index,
      candidate=predicted_candidate,
      block=predicted_block,
      orig_block=predicted_orig_block,
      orig_tokens=predicted_orig_tokens,
      orig_start=predicted_orig_start,
      orig_end=predicted_orig_end,
      answer=predicted_answer)


def compute_correct_candidates(candidate_starts, candidate_ends, gold_starts,
                               gold_ends):
  """Compute correct span."""
  # [reader_beam_size, num_answers, num_candidates]
  is_gold_start = tf.equal(
      tf.expand_dims(tf.expand_dims(candidate_starts, 0), 0),
      tf.expand_dims(gold_starts, -1))
  is_gold_end = tf.equal(
      tf.expand_dims(tf.expand_dims(candidate_ends, 0), 0),
      tf.expand_dims(gold_ends, -1))

  # [reader_beam_size, num_candidates]
  return tf.reduce_any(tf.logical_and(is_gold_start, is_gold_end), 1)


def compute_loss(retriever_logits, retriever_correct, reader_logits,
                 reader_correct):
  """Compute loss."""
  # []
  retriever_loss = marginal_log_loss(retriever_logits, retriever_correct)

  # []
  reader_loss = marginal_log_loss(
      tf.reshape(reader_logits, [-1]), tf.reshape(reader_correct, [-1]))

  # []
  any_retrieved_correct = tf.reduce_any(retriever_correct)
  any_reader_correct = tf.reduce_any(reader_correct)

  retriever_loss *= tf.cast(any_retrieved_correct, tf.float32)

  reader_loss *= tf.cast(any_reader_correct, tf.float32)

  loss = retriever_loss + reader_loss

  tf.summary.scalar("num_read_correct",
                    tf.reduce_sum(tf.cast(reader_correct, tf.int32)))
  tf.summary.scalar("reader_loss", tf.reduce_mean(reader_loss))
  tf.summary.scalar("retrieval_loss", tf.reduce_mean(retriever_loss))

  # []
  loss = tf.reduce_mean(loss)
  return loss


def compute_eval_metrics(labels, predictions, retriever_correct,
                         reader_correct):
  """Compute eval metrics."""
  # []
  exact_match = tf.gather(
      tf.gather(reader_correct, predictions["block_index"]),
      predictions["candidate"])

  def _official_exact_match(predicted_answer, references):
    is_correct = eval_utils.is_correct(
        answers=[six.ensure_text(r, errors="ignore") for r in references],
        prediction=six.ensure_text(predicted_answer, errors="ignore"),
        is_regex=False)
    return is_correct

  official_exact_match = tf.py_func(
      func=_official_exact_match,
      inp=[predictions["answer"], labels],
      Tout=tf.bool)

  eval_metric_ops = dict(
      exact_match=tf.metrics.mean(exact_match),
      official_exact_match=tf.metrics.mean(official_exact_match),
      reader_oracle=tf.metrics.mean(tf.reduce_any(reader_correct)))
  for k in (5, 10, 50, 100, 500, 1000, 5000):
    eval_metric_ops["top_{}_match".format(k)] = tf.metrics.mean(
        tf.reduce_any(retriever_correct[:k]))
  return eval_metric_ops


def model_fn(features, labels, mode, params):
  """Model function."""
  if labels is None:
    labels = tf.constant([""])

  reader_beam_size = params["reader_beam_size"]
  if mode == tf.estimator.ModeKeys.PREDICT:
    retriever_beam_size = reader_beam_size
  else:
    retriever_beam_size = params["retriever_beam_size"]
  assert reader_beam_size <= retriever_beam_size

  with tf.device("/cpu:0"):
    retriever_outputs = retrieve(
        features=features,
        retriever_beam_size=retriever_beam_size,
        mode=mode,
        params=params)

  with tf.variable_scope("reader"):
    reader_outputs = read(
        features=features,
        retriever_logits=retriever_outputs.logits[:reader_beam_size],
        blocks=retriever_outputs.blocks[:reader_beam_size],
        mode=mode,
        params=params,
        labels=labels)

  predictions = get_predictions(reader_outputs, params)

  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
    train_op = None
    eval_metric_ops = None
  else:
    # [retriever_beam_size]
    retriever_correct = orqa_ops.has_answer(
        blocks=retriever_outputs.blocks, answers=labels)

    # [reader_beam_size, num_candidates]
    reader_correct = compute_correct_candidates(
        candidate_starts=reader_outputs.candidate_starts,
        candidate_ends=reader_outputs.candidate_ends,
        gold_starts=reader_outputs.gold_starts,
        gold_ends=reader_outputs.gold_ends)

    eval_metric_ops = compute_eval_metrics(
        labels=labels,
        predictions=predictions,
        retriever_correct=retriever_correct,
        reader_correct=reader_correct)

    # []
    loss = compute_loss(
        retriever_logits=retriever_outputs.logits,
        retriever_correct=retriever_correct,
        reader_logits=reader_outputs.logits,
        reader_correct=reader_correct)

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


def serving_fn():
  placeholders = dict(
      question=tf.placeholder(dtype=tf.string, shape=[], name="question"))
  return tf.estimator.export.ServingInputReceiver(placeholders, placeholders)


def exporter():
  """Create exporters."""
  return exporters.BestSavedModelAndCheckpointExporter(
      eval_spec_name="default",
      serving_input_receiver_fn=serving_fn,
      metric_name="exact_match")


def input_fn(is_train, name, params):
  """An input function satisfying the tf.estimator API."""
  dataset = orqa_dataset.get_dataset(
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


def get_predictor_for_model_fn(model_dir, model_function):
  """Build a predictor. Can't use SavedModel here due to the py_func."""
  with tf.io.gfile.GFile(os.path.join(model_dir, "params.json")) as f:
    params = json.load(f)


  best_checkpoint_pattern = os.path.join(model_dir, "export", "best_default",
                                         "checkpoint", "*.index")
  best_checkpoint = tf.io.gfile.glob(
      best_checkpoint_pattern)[0][:-len(".index")]
  serving_input_receiver = serving_fn()
  estimator_spec = model_function(
      features=serving_input_receiver.features,
      labels=None,
      mode=tf.estimator.ModeKeys.PREDICT,
      params=params)
  question_tensor = serving_input_receiver.receiver_tensors["question"]
  session = tf.train.MonitoredSession(
      session_creator=tf.train.ChiefSessionCreator(
          checkpoint_filename_with_path=best_checkpoint))

  def _predict(question):
    return session.run(
        estimator_spec.predictions, feed_dict={question_tensor: question})

  return _predict


def get_predictor(model_dir):
  """Build a ORQA predictor. Can't use SavedModel here due to the py_func."""
  return get_predictor_for_model_fn(model_dir, model_fn)

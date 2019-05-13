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
"""Model function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial  # pylint: disable=g-importing-member
import sys
import language.labs.exemplar_decoding.models.baselines as baseline
import language.labs.exemplar_decoding.models.common as common
import language.labs.exemplar_decoding.models.hypernet as hypernet
from language.labs.exemplar_decoding.utils import data
from language.labs.exemplar_decoding.utils import rouge_utils
import numpy as np
import tensorflow as tf


def repetitive_ngrams(n, text):
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram = tuple(text[i:i + n])
    if ngram in ngram_set:
      return True
    ngram_set.add(ngram)
  return False


def bad_tok(tokens, vocab):
  for t in tokens:
    if not vocab.is_token(t):
      return True
  return False


def remove_repetitive_trigram(preds, lengths, vocab, hps):
  """Select from the beam a prediction without repetitive trigrams."""
  ret_preds, ret_lengths = [], []
  for (pred, length) in zip(preds, lengths):
    flag = True
    for i in xrange(hps.beam_width):
      l = length[Ellipsis, i]
      p = pred[Ellipsis, i][:l]
      tokens = data.id2text(p, vocab=vocab, use_bpe=hps.use_bpe).split()
      flag = repetitive_ngrams(3, tokens) or bad_tok(tokens, vocab)
      if not flag:
        ret_preds.append(pred[Ellipsis, i])
        ret_lengths.append(length[Ellipsis, i])
        break
    if flag:
      ret_preds.append(pred[Ellipsis, 0])
      ret_lengths.append(length[Ellipsis, 0])

  predictions = np.int32(np.stack(ret_preds)), np.int32(np.stack(ret_lengths))
  return predictions


def model_function(features, labels, mode, vocab, hps):
  """Model function.

  Atttention seq2seq model, augmented with an encoder
  over the targets of the nearest neighbors.

  Args:
    features: Dictionary of input Tensors.
    labels: Unused.
    mode: train or eval. Keys from tf.estimator.ModeKeys.
    vocab: A list of strings of words in the vocabulary.
    hps: Hyperparams.

  Returns:
    A tf.estimator.EstimatorSpec object containing model outputs.
  """
  targets = features["targets"]
  del labels

  if hps.model in ["hypernet"]:
    model = hypernet
  elif hps.model in ["seq2seq", "nn2seq"]:
    model = baseline

  # [batch_size, dec_len]
  decoder_inputs = features["decoder_inputs"]
  encoder_outputs = model.encoder(
      features=features,
      mode=mode,
      vocab=vocab,
      hps=hps
  )

  if mode == tf.estimator.ModeKeys.TRAIN:
    if hps.predict_mode:
      sys.exit(0)
    outputs, _ = model.basic_decoder(
        features=features,
        mode=mode,
        vocab=vocab,
        encoder_outputs=encoder_outputs,
        hps=hps
    )
    decoder_outputs = outputs.decoder_outputs
    decoder_len = features["decoder_len"]

    weights = tf.sequence_mask(
        decoder_len,
        tf.shape(decoder_inputs)[1],
        dtype=tf.float32)
    loss, train_op = common.optimize_log_loss(
        decoder_tgt=targets,
        decoder_outputs=decoder_outputs,
        weights=weights,
        hps=hps)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  elif mode == tf.estimator.ModeKeys.EVAL:
    outputs, teacher_outputs = model.basic_decoder(
        features=features,
        mode=mode,
        vocab=vocab,
        encoder_outputs=encoder_outputs,
        hps=hps
    )
    if hps.beam_width > 0:
      assert hps.beam_width > 1
      outputs = model.beam_decoder(
          features=features,
          mode=mode,
          vocab=vocab,
          encoder_outputs=encoder_outputs,
          hps=hps
      )

    # [batch_size, max_dec_len]
    weights = tf.sequence_mask(
        features["decoder_len"],
        tf.shape(decoder_inputs)[1],
        dtype=tf.float32)

    loss, _ = common.optimize_log_loss(
        decoder_tgt=targets,
        decoder_outputs=teacher_outputs.decoder_outputs,
        weights=weights,
        hps=hps)

    decoder_outputs = outputs.decoder_outputs
    decoder_len = outputs.decoder_len
    if hps.beam_width == 0:
      predicted_ids = decoder_outputs.sample_id
      lengths = outputs.decoder_len
      predictions = {
          "outputs": predicted_ids,
          "lengths": lengths,
      }
    else:
      predicted_ids, lengths = tf.py_func(
          partial(remove_repetitive_trigram,
                  vocab=vocab,
                  hps=hps),
          [decoder_outputs.predicted_ids, outputs.decoder_len],
          (tf.int32, tf.int32)
      )
      predictions = {
          "outputs": predicted_ids,
          "lengths": lengths,
      }
    reference = {
        "outputs": features["reference"],
        "lengths": features["reference_len"]
    }
    eval_metric_ops = rouge_utils.get_metrics(
        predictions["outputs"], predictions["lengths"],
        reference["outputs"], reference["lengths"],
        vocab=vocab, use_bpe=hps.use_bpe, predict_mode=hps.predict_mode
    )

    # [batch_size, src_len]
    src_inputs = features["src_inputs"]
    src_len = features["src_len"]

    # [batch_size, neighbor_len]
    neighbor_inputs = features["neighbor_inputs"]
    neighbor_len = features["neighbor_len"]

    print_op = common.print_text(
        [("source", src_inputs, src_len, True),
         ("neighbor", neighbor_inputs, neighbor_len, True),
         ("targets", reference["outputs"], reference["lengths"], False),
         ("predictions", predictions["outputs"], predictions["lengths"], True)],
        vocab, use_bpe=hps.use_bpe, predict_mode=hps.predict_mode)
    with tf.control_dependencies([print_op]):
      loss = tf.identity(loss)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    outputs, teacher_outputs = model.basic_decoder(
        features=features,
        mode=mode,
        vocab=vocab,
        encoder_outputs=encoder_outputs,
        hps=hps
    )
    if hps.beam_width > 0:
      assert hps.beam_width > 1
      outputs = model.beam_decoder(
          features=features,
          mode=mode,
          vocab=vocab,
          encoder_outputs=encoder_outputs,
          hps=hps
      )

    decoder_outputs = outputs.decoder_outputs
    decoder_len = outputs.decoder_len
    if hps.beam_width == 0:
      predicted_ids = decoder_outputs.sample_id
      lengths = outputs.decoder_len
    else:
      predicted_ids = decoder_outputs.predicted_ids[Ellipsis, 0]
      lengths = outputs.decoder_len[Ellipsis, 0]

    predictions = {
        "outputs": predicted_ids,
        "lengths": lengths - 1,
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

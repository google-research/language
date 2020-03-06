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
"""Utilities for generating model function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.xsp.model import adam_weight_decay
from language.xsp.model import bert_utils
from language.xsp.model import constants
from language.xsp.model import decode_utils
from language.xsp.model import embeddings
from language.xsp.model import load_from_checkpoint
from language.xsp.model import loss
from language.xsp.model import metrics
from language.xsp.model import tpu_utils
from language.xsp.model import transformer
import tensorflow.compat.v1 as tf


def _compute_loss(logits, decode_steps, target_len, weights, output_vocab_size,
                  model_config):
  """Computes loss given batch of model outputs and labels."""
  logit_target_ids = decode_utils.get_extended_indices(decode_steps,
                                                       output_vocab_size,
                                                       model_config)
  batch_loss = loss.sequence_loss(
      logits=logits,
      targets=logit_target_ids,
      sequence_length=target_len,
      weights=weights)

  # Calculate the average log perplexity.
  return tf.reduce_sum(batch_loss.losses) / batch_loss.total_steps


def build_model_fn(model_config,
                   output_vocab_filepath,
                   use_tpu=False,
                   beam_size=1):
  """Builds model function based on model_config."""

  def model_fn(features, labels, mode, params=None):
    """Model function for use with tf.learn.Estimator."""
    del params  # unused. model_fn is batch-size agnostic.

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    pretrained_variable_names = None
    scaffold_fn = None

    bert_config = bert_utils.get_bert_config(
        model_config.model_parameters.pretrained_bert_dir,
        reinitialize_type_embeddings=model_config.model_parameters
        .use_segment_ids)

    input_embeddings = embeddings.get_input_embeddings(
        model_config,
        bert_config,
        features,
        is_training,
        use_one_hot_embeddings=use_tpu)

    source_len = tf.to_int32(features[constants.SOURCE_LEN_KEY])

    output_embeddings_table = embeddings.get_output_vocab_embeddings_table(
        model_config, output_vocab_filepath)

    output_vocab_size = embeddings.get_output_vocab_size(output_vocab_filepath)

    # For inference, just compute the inference predictions and return.
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = transformer.infer(
          model_config,
          input_embeddings,
          source_len,
          output_vocab_size,
          output_embeddings_table,
          mode,
          input_copy_mask=features[constants.COPIABLE_INPUT_KEY],
          beam_size=beam_size)

      if use_tpu:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tpu_utils.rewire_summary_calls(use_tpu):
      # Get training predictions.
      train_decode_steps = decode_utils.decode_steps_from_labels(
          labels, trim_end_symbol=True)
      logits, predictions = transformer.train(
          model_config,
          input_embeddings,
          source_len,
          output_vocab_size,
          output_embeddings_table,
          train_decode_steps,
          mode,
          input_copy_mask=features[constants.COPIABLE_INPUT_KEY])

      # Calculate loss.
      weights = labels[constants.WEIGHT_KEY]
      loss_decode_steps = decode_utils.decode_steps_from_labels(
          labels, trim_start_symbol=True)

      # Account for removed start symbol.
      target_len = tf.to_int32(labels[constants.TARGET_LEN_KEY])
      target_len -= 1

      batch_loss = _compute_loss(logits, loss_decode_steps, target_len, weights,
                                 output_vocab_size, model_config)

      if mode == tf.estimator.ModeKeys.TRAIN:
        pretrained_variable_names, scaffold_fn = load_from_checkpoint.init_model_from_checkpoint(
            model_config.model_parameters.pretrained_bert_dir,
            use_tpu=use_tpu,
            checkpoint_file="bert_model.ckpt",
            reinitialize_type_embeddings=model_config.model_parameters
            .use_segment_ids)
        train_op = adam_weight_decay.build_train_op_with_pretraining(
            batch_loss, model_config, pretrained_variable_names, use_tpu)

        if use_tpu:
          return tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=batch_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)
        else:
          return tf.estimator.EstimatorSpec(
              mode=mode, loss=batch_loss, train_op=train_op)

    eval_metrics = metrics.create_metrics_ops(
        labels=labels, predictions=predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=batch_loss, eval_metric_ops=eval_metrics)

  return model_fn

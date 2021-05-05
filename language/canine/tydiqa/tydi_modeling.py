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
"""Generic BERT-like compute graph implemented in TensorFlow.

CANINE-specific instantiation appears briefly at the bottom of this file.
"""

import abc
import collections

from absl import logging
from language.canine import bert_modeling  # For utility functions.
from language.canine import bert_optimization
from language.canine import modeling as canine_modeling
from language.canine.tydiqa import data
import tensorflow.compat.v1 as tf


class GenericModelBuilder(metaclass=abc.ABCMeta):
  """Class for constructing a TyDi QA model based on a BERT-like model.

  You can directly extend this class to change the model behavior, for example
  by overriding `create_encoder_model` to return a class with a different
  encoder architecture.
  """

  @abc.abstractmethod
  def create_encoder_model(self, is_training, input_ids, input_mask,
                           segment_ids):
    raise NotImplementedError()

  def create_classification_model(self, encoder_model):
    """Creates a classification model."""
    # Get the logits for the start and end predictions.
    final_hidden = encoder_model.get_sequence_output()

    batch_size, seq_length, hidden_size = (
        bert_modeling.get_shape_list(final_hidden, expected_rank=3))

    output_weights = tf.get_variable(
        "cls/tydi/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/tydi/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    start_logits, end_logits = tf.unstack(logits, axis=0)

    # Get the logits for the answer type prediction.
    answer_type_output_layer = encoder_model.get_pooled_output()
    answer_type_hidden_size = bert_modeling.get_shape_list(
        answer_type_output_layer)[-1]

    num_answer_types = len(data.AnswerType)
    answer_type_output_weights = tf.get_variable(
        "answer_type_output_weights",
        [num_answer_types, answer_type_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    answer_type_output_bias = tf.get_variable(
        "answer_type_output_bias", [num_answer_types],
        initializer=tf.zeros_initializer())

    answer_type_logits = tf.matmul(
        answer_type_output_layer, answer_type_output_weights, transpose_b=True)
    answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                        answer_type_output_bias)

    return start_logits, end_logits, answer_type_logits

  def model_fn_builder(self, init_checkpoint, learning_rate, num_train_steps,
                       num_warmup_steps, use_tpu):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
      """The `model_fn` for TPUEstimator."""

      del labels, params  # Unused.

      logging.info("*** Features ***")
      for name in sorted(features.keys()):
        logging.info("  name = %s, shape = %s", name, features[name].shape)

      unique_ids = features["unique_ids"]
      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]

      is_training = (mode == tf.estimator.ModeKeys.TRAIN)

      encoder_model = self.create_encoder_model(is_training, input_ids,
                                                input_mask, segment_ids)
      start_logits, end_logits, answer_type_logits = (
          self.create_classification_model(encoder_model))

      tvars = tf.trainable_variables()

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        assignment_map, initialized_variable_names = (
            bert_modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint))
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)

      output_spec = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        seq_length = bert_modeling.get_shape_list(input_ids)[1]

        # Computes the loss for positions.
        def compute_loss(logits, positions):
          one_hot_positions = (
              tf.one_hot(positions, depth=seq_length, dtype=tf.float32))
          log_probs = tf.nn.log_softmax(logits, axis=-1)
          loss = -tf.reduce_mean(
              tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
          return loss

        # Computes the loss for labels.
        def compute_label_loss(logits, labels):
          one_hot_labels = (
              tf.one_hot(labels, depth=len(data.AnswerType), dtype=tf.float32))
          log_probs = tf.nn.log_softmax(logits, axis=-1)
          loss = -tf.reduce_mean(
              tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
          return loss

        start_positions = features["start_positions"]
        end_positions = features["end_positions"]
        answer_types = features["answer_types"]

        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)

        answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

        total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

        train_op = bert_optimization.create_optimizer(total_loss, learning_rate,
                                                      num_train_steps,
                                                      num_warmup_steps, use_tpu)

        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "unique_ids": unique_ids,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "answer_type_logits": answer_type_logits,
        }
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        raise ValueError(f"Only TRAIN and PREDICT modes are supported: {mode}")
      return output_spec
    return model_fn


class CanineModelBuilder(GenericModelBuilder):
  """Class for constructing a TyDi QA model based on CANINE."""

  def __init__(self, model_config: canine_modeling.CanineModelConfig):
    self.model_config = model_config

  def create_encoder_model(self, is_training, input_ids, input_mask,
                           segment_ids):
    return canine_modeling.CanineModel(
        config=self.model_config,
        atom_input_ids=input_ids,
        atom_input_mask=input_mask,
        atom_segment_ids=segment_ids,
        is_training=is_training)


# This represents the raw predictions coming out of the neural model.
RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])

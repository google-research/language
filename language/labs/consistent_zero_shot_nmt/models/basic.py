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
"""Basic NMT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from language.labs.consistent_zero_shot_nmt.modules import decoders
from language.labs.consistent_zero_shot_nmt.modules import encoders
from language.labs.consistent_zero_shot_nmt.utils import common_utils as U
from language.labs.consistent_zero_shot_nmt.utils import model_utils
from language.labs.consistent_zero_shot_nmt.utils import t2t_tweaks  # pylint: disable=unused-import

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicMultilingualNmt(t2t_model.T2TModel):
  """Basic multilingual NMT model that follows T2T API.

  Reference: https://arxiv.org/abs/1611.04558.
  """

  def bottom(self, features):
    """Transforms features to feed into body.

    Ensures that all language tags are transformed using input modality.

    Args:
      features: dict of str to Tensor. The tensors contain token ids.

    Returns:
      transformed_features: dict of same key-value pairs as features. The value
        Tensors are newly transformed (i.e., embeddings).
    """
    if not self._problem_hparams:
      t2t_model.log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
      return features

    transformed_features = collections.OrderedDict()

    # Transform inputs.
    feature_name = "inputs"
    modality_obj = self._problem_hparams.modality[feature_name]
    with tf.variable_scope(modality_obj.name, reuse=False) as vs:
      self._add_variable_scope(modality_obj.name, vs)
      t2t_model.log_info("Transforming feature '%s' with %s.bottom",
                         feature_name,
                         modality_obj.name)
      transformed_features[feature_name] = modality_obj.bottom_simple(
          features[feature_name], "input_emb", reuse=False)

    # Transform tags (using same modality as for the inputs).
    for feature_name in ["all_tags", "input_tags", "target_tags"]:
      if feature_name not in features:
        tf.logging.warning("Missing feature %s - ignoring." % feature_name)
        continue
      with tf.variable_scope(modality_obj.name, reuse=True):
        t2t_model.log_info("Transforming feature '%s' with %s.bottom_simple",
                           feature_name,
                           modality_obj.name)
        transformed_features[feature_name] = modality_obj.bottom_simple(
            features[feature_name], "input_emb", reuse=True)

    # Transform targets.
    feature_name = "targets"
    modality_obj = self._problem_hparams.modality[feature_name]
    with tf.variable_scope(modality_obj.name, reuse=False) as vs:
      self._add_variable_scope(modality_obj.name, vs)
      t2t_model.log_info("Transforming feature '%s' with %s.bottom_simple",
                         feature_name,
                         modality_obj.name)
      transformed_features[feature_name] = modality_obj.bottom_simple(
          features[feature_name], "shared", reuse=False)

    for key in features:
      if key not in transformed_features:
        # For features without a modality, we pass them along as is
        transformed_features[key] = features[key]
      else:
        # Other features get passed along with the "raw" suffix
        transformed_features[key + "_raw"] = features[key]

    return transformed_features

  def _preprocess(self, features):
    """Preprocesses features for multilingual translation."""
    inputs = features["inputs"]
    targets = features["targets"]
    target_tags = features["target_tags"]

    # Expand target tags to beam width, if necessary.
    if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
      # <float32> [batch_size * beam_width, 1, 1, emb_size].
      beam_width = self._hparams.beam_width
      target_tags = tf.tile(target_tags, [beam_width, 1, 1, 1])

    # Add target tags to the input sequences.
    # <float32> [batch_size, seq_len + 1, 1, emb_size].
    inputs = tf.concat([target_tags, inputs], axis=1)

    # Compute length of the input sequences.
    inputs_length = common_layers.length_from_embedding(inputs)
    inputs = common_layers.flatten4d3d(inputs)

    # Preprocess targets.
    targets = common_layers.shift_right(targets)
    # Add 1 to account for the padding added to the left from shift_right.
    targets_length = common_layers.length_from_embedding(targets) + 1
    targets = common_layers.flatten4d3d(targets)

    return inputs, inputs_length, targets, targets_length

  def _top_single(self, body_output, target_modality, features):
    """Top transformation that ensures correct reuse of target embeddings."""
    t2t_model.log_info(
        "Transforming body output with %s.top", target_modality.name)

    # Get target embeddings.
    target_modality = self._problem_hparams.modality["targets"]
    target_modality_scope = self._variable_scopes[target_modality.name]
    target_embeddings = model_utils.get_embeddings(
        modality=target_modality,
        outer_scope=target_modality_scope,
        inner_scope="shared")
    target_vocab_size = target_modality._vocab_size  # pylint: disable=protected-access

    # Preprocess body output.
    last_only = (
        target_modality.top_is_pointwise and
        self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
        not self.hparams.force_full_predict)
    if last_only:
      # Take body outputs for the last position only.
      if "decode_loop_step" not in features:
        body_output = tf.expand_dims(body_output[:, -1, :, :], axis=[1])
      else:
        body_output_shape = body_output.shape.as_list()
        body_output = tf.slice(
            body_output, [0, features["decode_loop_step"][0], 0, 0], [
                body_output_shape[0], 1, body_output_shape[2],
                body_output_shape[3]
            ])

    # Build logits.
    logits = model_utils.build_logits(
        sequences=body_output,
        embeddings=target_embeddings,
        vocab_size=target_vocab_size)
    return logits

  def body(self, features):
    # Preprocess features.
    inputs, inputs_length, targets, targets_length = self._preprocess(features)

    # Encode.
    encoder = encoders.get(self._hparams.encoder_type)
    enc_outputs = encoder(
        inputs=inputs,
        inputs_length=inputs_length,
        mode=self._hparams.mode,
        hparams=self._hparams)

    # Get target embeddings.
    target_modality = self._problem_hparams.modality["targets"]
    target_modality_scope = self._variable_scopes[target_modality.name]
    target_embeddings = model_utils.get_embeddings(
        modality=target_modality,
        outer_scope=target_modality_scope,
        inner_scope="shared")

    # Decode.
    decoder = decoders.get(self._hparams.decoder_type)
    decoder = decoder(
        embeddings=target_embeddings,
        inputs=targets,
        inputs_length=targets_length,
        hiddens=enc_outputs.outputs,
        hiddens_length=inputs_length,
        enc_state=enc_outputs.final_state,
        mode=self._hparams.mode,
        hparams=self._hparams)
    dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

    return tf.expand_dims(dec_outputs.rnn_output, axis=2)


# ------------------------------------------------------------------------------
# Hparams
# ------------------------------------------------------------------------------


def base_nmt():
  """Base hparams for LSTM-based NMT models."""
  hparams = common_hparams.basic_params1()

  # Architecture.
  hparams.shared_embedding_and_softmax_weights = True
  hparams.daisy_chain_variables = False
  hparams.batch_size = 1024
  hparams.hidden_size = 512
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.2
  hparams.add_hparam("rnn_unit_type", "lstm")
  hparams.add_hparam("rnn_forget_bias", 1.0)
  hparams.add_hparam("pass_hidden_state", True)

  # Optimizer.
  # Adafactor uses less memory than Adam.
  # switch to Adafactor with its recommended learning rate scheme.
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000

  # Decoder.
  hparams.add_hparam("beam_width", 4)
  hparams.add_hparam("decode_length", 50)
  hparams.add_hparam("decoder_type", "basic")
  hparams.add_hparam("decoder_continuous", False)

  # Training.
  hparams.add_hparam("scheduled_training", False)
  hparams.add_hparam("scheduled_sampling_probability", 0.2)
  hparams.add_hparam("scheduled_mixing_concentration", 10.)

  return hparams


def base_att(hparams):
  """Adds base attention hparams."""
  hparams.add_hparam("attention_layer_size", hparams.hidden_size)
  hparams.add_hparam("output_attention", True)
  hparams.add_hparam("num_heads", 1)
  return hparams


## Bidirectional architectures.


def base_nmt_bilstm(hparams):
  """Base hparams for LSTM-based NMT models with bidirectional encoders."""
  ## Encoder.
  hparams.add_hparam("encoder_type", U.ENC_BI)
  hparams.add_hparam("enc_num_layers", 2)
  hparams.add_hparam("enc_num_residual_layers", 1)

  ## Decoder.
  hparams.add_hparam("dec_num_layers", 4)
  hparams.add_hparam("dec_num_residual_layers", 3)

  return hparams


@registry.register_hparams
def basic_nmt_bilstm_bahdanau_att():
  """Hparams for LSTM with bahdanau attention."""
  hparams = base_nmt_bilstm(base_att(base_nmt()))
  hparams.add_hparam("attention_gnmt", False)
  hparams.add_hparam("attention_mechanism", "bahdanau")
  return hparams


@registry.register_hparams
def basic_nmt_bilstm_luong_att():
  """Hparams for LSTM with luong attention."""
  hparams = base_nmt_bilstm(base_att(base_nmt()))
  hparams.add_hparam("attention_gnmt", False)
  hparams.add_hparam("attention_mechanism", "luong")
  return hparams


@registry.register_hparams
def basic_nmt_bilstm_bahdanau_att_multi():
  """Hparams for LSTM with bahdanau attention."""
  hparams = basic_nmt_bilstm_bahdanau_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def basic_nmt_bilstm_luong_att_multi():
  """Hparams for LSTM with luong attention."""
  hparams = basic_nmt_bilstm_luong_att()
  hparams.num_heads = 4
  return hparams


## GNMT architectures.


def base_nmt_gnmt_lstm(hparams):
  """Base hparams for LSTM-based GNMT models."""
  ## Adjust attention.
  hparams.output_attention = False

  ## Encoder.
  hparams.add_hparam("encoder_type", U.ENC_GNMT)
  hparams.add_hparam("enc_num_layers", 3)
  hparams.add_hparam("enc_num_residual_layers", 1)

  ## Decoder.
  hparams.add_hparam("dec_num_layers", 3)
  hparams.add_hparam("dec_num_residual_layers", 2)

  return hparams


@registry.register_hparams
def basic_gnmt_bahdanau_att():
  """Hparams for LSTM with bahdanau attention."""
  hparams = base_nmt_gnmt_lstm(base_att(base_nmt()))
  hparams.add_hparam("attention_gnmt", True)
  hparams.add_hparam("attention_mechanism", "bahdanau")
  return hparams


@registry.register_hparams
def basic_gnmt_luong_att():
  """Hparams for LSTM with luong attention."""
  hparams = base_nmt_gnmt_lstm(base_att(base_nmt()))
  hparams.add_hparam("attention_gnmt", True)
  hparams.add_hparam("attention_mechanism", "luong")
  return hparams


@registry.register_hparams
def basic_gnmt_bahdanau_att_multi():
  """Hparams for LSTM with bahdanau attention."""
  hparams = basic_gnmt_bahdanau_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def basic_gnmt_luong_att_multi():
  """Hparams for LSTM with luong attention."""
  hparams = basic_gnmt_luong_att()
  hparams.num_heads = 4
  return hparams

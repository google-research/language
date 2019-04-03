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
"""Multilingual NMT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from language.labs.consistent_zero_shot_nmt.data_generators import translate_multilingual
from language.labs.consistent_zero_shot_nmt.models import basic
from language.labs.consistent_zero_shot_nmt.models import losses
from language.labs.consistent_zero_shot_nmt.modules import decoders
from language.labs.consistent_zero_shot_nmt.modules import encoders
from language.labs.consistent_zero_shot_nmt.modules import language_models
from language.labs.consistent_zero_shot_nmt.utils import model_utils
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry

import tensorflow as tf


__all__ = [
    "AgreementMultilingualNmt",
    "AgreementMultilingualNmtLm",
]


@registry.register_model
class AgreementMultilingualNmt(basic.BasicMultilingualNmt):
  """Multilingual NMT model that encourages agreement between submodels.

  TODO(alshedivat): Conditional graph building got tricky. Refactor.
  """

  def _build_inputs_and_targets(
      self, from_seqs=None, from_tags=None, to_seqs=None, to_tags=None):
    """Given from and to sequences and tags, construct inputs and targets."""
    if from_seqs is not None:
      inputs = from_seqs
      inputs_length = common_layers.length_from_embedding(inputs)
      if to_tags is not None:
        # Add to-tags to the inputs and adjust lengths.
        # <float32> [batch_size, seq_len + 1, 1, emb_size].
        inputs = tf.concat([to_tags, inputs], axis=1)
        inputs_length = inputs_length + 1
      inputs = common_layers.flatten4d3d(inputs)
    else:
      inputs = None
      inputs_length = None

    if to_seqs is not None:
      # Shift to-sequences to form targets.
      # <float32> [batch_size, seq_len, 1, emb_size].
      targets = common_layers.shift_right(to_seqs)
      # Add 1 to account for the padding added to the left from shift_right.
      targets_length = common_layers.length_from_embedding(targets) + 1
      targets = common_layers.flatten4d3d(targets)
    else:
      targets = None
      targets_length = None

    return (inputs, inputs_length), (targets, targets_length)

  def _preprocess(self, features):
    """Preprocesses features for multilingual translation."""
    seqs, tags = {}, {}

    if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
      seqs["src"] = features["inputs"]
      seqs["tgt"] = features["targets"]
      seqs["aux"] = None
      tags["src"] = features["input_tags"]
      tags["tgt"] = features["target_tags"]
      tags["aux"] = None

      # Construct a tensor of auxiliary tags.
      batch_size = common_layers.shape_list(features["all_tags"])[0]
      num_all_tags = common_layers.shape_list(features["all_tags"])[1]
      # <float32> [num_all_tags, 1, emb_dim].
      all_tags = features["all_tags"][0]  # batch elements are identical.
      # <int32> [batch_size].
      aux_tag_index = tf.multinomial(
          tf.ones([1, num_all_tags]), batch_size,
          output_dtype=tf.int32)[0]
      # <float32> [batch_size, 1, 1, emb_dim].
      tags["aux"] = tf.expand_dims(tf.gather(all_tags, aux_tag_index), 1)

      from_domains = ["src", "src", "tgt"]
      to_domains = ["tgt", "aux", "aux"]
    else:
      seqs["src"] = features["inputs"]
      seqs["tgt"] = features["targets"]
      tags["src"] = None
      tags["tgt"] = features["target_tags"]

      # Expand target tags to beam width, if necessary.
      if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
        tags["tgt"] = tf.tile(tags["tgt"], [self._hparams.beam_width, 1, 1, 1])

      from_domains = ["src"]
      to_domains = ["tgt"]

    # Construct inputs and targets.
    inputs, targets = {}, {}
    for fd, td in zip(from_domains, to_domains):
      key = "%s>%s" % (fd, td)
      inputs[key], targets[key] = self._build_inputs_and_targets(
          seqs[fd], tags[fd], seqs[td], tags[td])

    return inputs, targets

  def _build_encoder_agreement_loss(self):
    """Builds an agreement loss that enforces consistency of the encodings.

    Returns:
      loss: <float32> [] for the agreement losses.

    Raises:
      ValueError: if loss_name is not in {"cosine", "entropy", "mse"}.
    """
    aux_keys = ["src>aux", "tgt>aux"]

    # Encode (if necessary).
    for key in aux_keys:
      if key not in self.enc_outputs:
        encode_func = self.get_encode_func(*self.inputs[key])
        self.enc_outputs[key] = encode_func()

    with tf.name_scope("enc_agreement_loss"):
      # Build loss.
      if self._hparams.enc_agreement_loss in {"cosine", "l2"}:
        if self._hparams.enc_agreement_pool:
          preproc_op_type = "max_pool"
        else:
          preproc_op_type = "truncate"
        enc_src, enc_tgt = model_utils.make_sequences_compatible(
            self.enc_outputs["src>aux"].outputs,
            self.enc_outputs["tgt>aux"].outputs,
            op_type=preproc_op_type)
        if self._hparams.enc_agreement_loss == "cosine":
          dist_fn = functools.partial(losses.cosine_distance, normalize=True)
        else:
          dist_fn = functools.partial(losses.l2_distance, normalize=True)
        aux_loss_fn = losses.DistanceLoss(dist_fn)
        aux_loss = aux_loss_fn(enc_src, enc_tgt)
      elif self._hparams.enc_agreement_loss in {"xatt_cosine", "xatt_l2"}:
        if self._hparams.enc_agreement_loss.endswith("cosine"):
          dist_fn = functools.partial(losses.cosine_distance, normalize=True)
        else:
          dist_fn = functools.partial(losses.l2_distance, normalize=True)
        aux_loss_fn = losses.CrossAttentionDistanceLoss(dist_fn=dist_fn)
        aux_loss = aux_loss_fn(self.enc_outputs["src>aux"].outputs,
                               self.enc_outputs["tgt>aux"].outputs)
      else:
        raise ValueError("Unknown auxiliary loss: %s." %
                         self._hparams.enc_agreement_loss)

      aux_loss = self._hparams.enc_agreement_coeff * aux_loss

    return aux_loss

  def _build_aux_sequences(self, target_embeddings, target_vocab_size,
                           central_lang_tag="<en>"):
    """Builds sequences in an auxiliary language."""
    aux_keys = ["src>aux", "tgt>aux"]

    # Determine which src and tgt sentences are central.
    central_lang_id = translate_multilingual.get_tag_id(central_lang_tag)
    self._is_central = {
        "src>aux": tf.squeeze(
            self._body_features["input_tags_raw"] == central_lang_id),
        "tgt>aux": tf.squeeze(
            self._body_features["target_tags_raw"] == central_lang_id)}

    for key in aux_keys:
      # Encode (if necessary).
      if key not in self.enc_outputs:
        encode_func = self.get_encode_func(*self.inputs[key])
        self.enc_outputs[key] = encode_func()

      # Decode (if necessary).
      if key not in self.dec_outputs:
        # Prepare for decoding.
        target_seqs, target_lens = self.targets[key]
        hiddens = self.enc_outputs[key].outputs
        hiddens_length = self.inputs[key][1]
        enc_state = self.enc_outputs[key].final_state
        decoder_hparams = tf.contrib.training.HParams(auxiliary=True)
        # Decode.
        decode_func = self.get_decode_func(
            target_embeddings,
            target_seqs, target_lens,
            hiddens, hiddens_length,
            enc_state,
            mode=self._hparams.mode,
            decoder_hparams=decoder_hparams,
            decoder_iterations=self._hparams.aux_decode_length)
        self.dec_outputs[key] = decode_func()
        # Compute logits.
        self.dec_outputs[key]["logits"] = model_utils.build_logits(
            sequences=tf.expand_dims(
                self.dec_outputs[key]["rnn_output"], axis=2),
            embeddings=target_embeddings,
            vocab_size=target_vocab_size)
        # Protect central directions from the gradients.
        for element in self.dec_outputs[key]:
          self.dec_outputs[key][element] = tf.where(
              self._is_central[key],
              tf.stop_gradient(self.dec_outputs[key][element]),
              self.dec_outputs[key][element])

    return aux_keys

  def _build_decoder_agreement_loss(self, central_lang_tag="<en>"):
    """Builds an agreement loss that enforces consistency of the decodings.

    Args:
      central_lang_tag: A string with the tag of the central language.
        A ``central'' language (usually English) is the one that has parallel
        data with all other languages. It is used to protect supervised
        directions from gradients coming from auxiliary losses.

    Returns:
      loss: <float32> [] for the agreement losses.
    """
    # Get target embeddigns and vocab size.
    target_modality = self._problem_hparams.modality["targets"]
    target_modality_scope = self._variable_scopes[target_modality.name]
    target_embeddings = model_utils.get_embeddings(
        modality=target_modality,
        outer_scope=target_modality_scope,
        inner_scope="shared")
    target_vocab_size = target_modality._vocab_size  # pylint: disable=protected-access

    # Build auxiliary sequences (if necessary).
    aux_keys = self._build_aux_sequences(
        target_embeddings, target_vocab_size,
        central_lang_tag=central_lang_tag)

    # Build loss.
    aux_loss = 0.
    with tf.name_scope("dec_agreement_loss"):
      for key1, key2 in zip(aux_keys, aux_keys[::-1]):
        # Prepare for decoding.
        targets = self.dec_outputs[key2]["rnn_output"]
        targets_length = self.dec_outputs[key2]["length"]
        shifted_targets = common_layers.shift_right_3d(targets)
        hiddens = self.enc_outputs[key1].outputs
        hiddens_length = self.inputs[key1][1]
        enc_state = self.enc_outputs[key1].final_state
        # Decode.
        decode_func = self.get_decode_func(
            target_embeddings,
            shifted_targets, targets_length,
            hiddens, hiddens_length,
            enc_state,
            mode=tf.estimator.ModeKeys.PREDICT,
            decoder_iterations=self._hparams.aux_decode_length)
        aux_dec_outputs = decode_func()
        # Compute logits (protect central directions from the gradients).
        aux_logits_1 = model_utils.build_logits(
            sequences=tf.expand_dims(
                aux_dec_outputs["rnn_output"], axis=2),
            embeddings=target_embeddings,
            vocab_size=target_vocab_size)
        aux_logits_1 = tf.where(
            self._is_central[key1],
            tf.stop_gradient(aux_logits_1),
            aux_logits_1)
        # Compute KL loss.
        logits = tf.squeeze(aux_logits_1, axis=2)
        if self._hparams.dec_agreement_loss_sparse:
          target_ids = self.dec_outputs[key2]["sample_id"]
          aux_loss = aux_loss + losses.CrossEntropyLoss(sparse=True)(
              logits, target_ids, targets_length)
        else:
          aux_logits_2 = tf.squeeze(self.dec_outputs[key2]["logits"], axis=2)
          target_probs = tf.nn.softmax(aux_logits_2, axis=-1)
          aux_loss = aux_loss + losses.CrossEntropyLoss(sparse=False)(
              logits, target_probs, targets_length)

    aux_loss = self._hparams.dec_agreement_coeff * aux_loss

    return aux_loss

  def get_encode_func(self, inputs, inputs_length):
    def encode_func():
      """A closure that builds encoder outputs."""
      return self.encoder(
          inputs=inputs,
          inputs_length=inputs_length,
          mode=self._hparams.mode,
          hparams=self._hparams,
          reuse=tf.AUTO_REUSE)
    return encode_func

  def get_decode_func(self, embeddings,
                      inputs, inputs_length,
                      hiddens, hiddens_length,
                      enc_state,
                      mode=None,
                      decoder_hparams=None,
                      impute_finished=False,
                      decoder_iterations=None):
    def decode_func():
      """A closure that builds decoder outputs."""
      dec_outputs, _, dec_lengths = tf.contrib.seq2seq.dynamic_decode(
          decoder=self.decoder(
              embeddings=embeddings,
              inputs=inputs,
              inputs_length=inputs_length,
              hiddens=hiddens,
              hiddens_length=hiddens_length,
              enc_state=enc_state,
              mode=mode,
              hparams=self._hparams,
              decoder_hparams=decoder_hparams,
              reuse=tf.AUTO_REUSE),
          impute_finished=impute_finished,
          maximum_iterations=decoder_iterations)
      return {
          "rnn_output": dec_outputs.rnn_output,
          "sample_id": dec_outputs.sample_id,
          "length": dec_lengths}
    return decode_func

  def body(self, features):
    # Save a reference to the features to access in other methods.
    self._body_features = features

    # Preprocess features.
    self.inputs, self.targets = self._preprocess(features)

    # Ensure auxiliary parts of the graph are built when necessary.
    batch_size = common_layers.shape_list(features["inputs"])[0]
    global_step = model_utils.get_global_step(self._hparams)

    # Encode (src>tgt).
    key = "src>tgt"
    self.enc_outputs = {}
    self.encoder = encoders.get(self._hparams.encoder_type)
    encode_func = self.get_encode_func(*self.inputs[key])
    self.enc_outputs[key] = encode_func()

    # Get target embeddings.
    target_modality = self._problem_hparams.modality["targets"]
    target_modality_scope = self._variable_scopes[target_modality.name]
    target_embeddings = model_utils.get_embeddings(
        modality=target_modality,
        outer_scope=target_modality_scope,
        inner_scope="shared")

    # Decode (src>tgt).
    key = "src>tgt"
    self.decoders = {}
    self.dec_outputs = {}
    self.decoder = decoders.get(self._hparams.decoder_type)
    # Prepare for decoding.
    target_seqs, target_lens = self.targets[key]
    hiddens = self.enc_outputs[key].outputs
    hiddens_length = self.inputs[key][1]
    enc_state = self.enc_outputs[key].final_state
    # Decode.
    decode_func = self.get_decode_func(
        target_embeddings,
        target_seqs, target_lens,
        hiddens, hiddens_length,
        enc_state,
        mode=self._hparams.mode)
    self.dec_outputs[key] = decode_func()
    outputs = tf.expand_dims(self.dec_outputs[key]["rnn_output"], axis=2)

    # Construct agreement losses.
    aux_losses = {}
    if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
      if self._hparams.enc_agreement_coeff > 0:
        aux_losses["agreement_enc"] = tf.cond(
            global_step > self._hparams.enc_agreement_enable_step,
            self._build_encoder_agreement_loss,
            lambda: tf.zeros([batch_size]))
      if self._hparams.dec_agreement_coeff > 0:
        aux_losses["agreement_dec"] = tf.cond(
            global_step > self._hparams.dec_agreement_enable_step,
            self._build_decoder_agreement_loss,
            lambda: tf.zeros([batch_size]))

    return outputs, aux_losses


@registry.register_model
class AgreementMultilingualNmtLm(AgreementMultilingualNmt):
  """Multilingual NMT model that encourages agreement between submodels.

  The model additionally trains and stores a multilingual LM. Similar to NMT,
  multilinguality of LM is enabled through language tags.

  There are two possible training modes:
    (1) LM training, where only parameters of the LM are updated.
    (2) NMT training, where LM is frozen and used to define an auxiliary loss.

  Full graphs for NMT and LM share variable spaces for easy warm-starting.
  """

  def _build_lm_inputs(self, features):
    """Builds inputs and targets for LM training."""
    targets = features["targets"]
    target_tags = features["target_tags"]

    if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
      target_tags = tf.tile(target_tags, [self._hparams.beam_width, 1, 1, 1])

    # Construct LM inputs.
    inputs = common_layers.shift_right(targets, pad_value=target_tags)
    inputs_length = common_layers.length_from_embedding(targets) + 1
    inputs = common_layers.flatten4d3d(inputs)

    return inputs, inputs_length

  def _build_decoder_lm_loss(self, central_lang_tag="<en>"):
    """Builds LM loss on the auxiliary decodings."""

    # Get target embeddigns and vocab size.
    target_modality = self._problem_hparams.modality["targets"]
    target_modality_scope = self._variable_scopes[target_modality.name]
    target_embeddings = model_utils.get_embeddings(
        modality=target_modality,
        outer_scope=target_modality_scope,
        inner_scope="shared")
    target_vocab_size = target_modality._vocab_size  # pylint: disable=protected-access

    # Build auxiliary sequences (if necessary).
    aux_keys = self._build_aux_sequences(
        target_embeddings, target_vocab_size,
        central_lang_tag=central_lang_tag)

    # Make sure LM loss does not affect embeddings.
    target_embeddings = tf.stop_gradient(target_embeddings)

    # Build loss.
    aux_loss = 0.
    with tf.name_scope("aux_lm_loss"):
      for key in aux_keys:
        dec_outputs = tf.expand_dims(
            self.dec_outputs[key]["rnn_output"], axis=2)
        dec_output_tags = tf.expand_dims(
            self.inputs[key][0][:, :1], axis=2)
        dec_lengths = self.dec_outputs[key]["length"]
        # Preprocess LM features.
        lm_features = {
            "targets": dec_outputs,
            "target_tags": dec_output_tags}
        inputs, inputs_length = self._build_lm_inputs(lm_features)
        # Build LM (with frozen weights in PREDICT mode).
        lm_outputs = self.language_model(
            inputs=inputs,
            inputs_length=inputs_length,
            mode=tf.estimator.ModeKeys.PREDICT,
            hparams=self._hparams,
            trainable=False,
            reuse=tf.AUTO_REUSE)
        # Compute logits.
        lm_logits = model_utils.build_logits(
            sequences=tf.expand_dims(lm_outputs, axis=2),
            embeddings=target_embeddings,
            vocab_size=target_vocab_size)
        # Compute decoder probabilities.
        dec_logits = model_utils.build_logits(
            sequences=dec_outputs,
            embeddings=target_embeddings,
            vocab_size=target_vocab_size)
        dec_probs = tf.nn.softmax(dec_logits, axis=-1)
        # Compute cross-entropy loss.
        aux_loss = aux_loss + losses.CrossEntropyLoss(sparse=False)(
            lm_logits, dec_probs, dec_lengths)

    aux_loss = self._hparams.lm_loss_coeff * aux_loss

    return aux_loss

  def body(self, features):
    if self._hparams.lm_do_train:
      inputs, inputs_length = self._build_lm_inputs(features)
      self.language_model = language_models.get(self._hparams.lm_type)
      lm_outputs = self.language_model(
          inputs=inputs,
          inputs_length=inputs_length,
          mode=self._hparams.mode,
          hparams=self._hparams)
      outputs = tf.expand_dims(lm_outputs, axis=2)
      aux_losses = {}
    else:
      # Build the NMT graph.
      nmt_body = super(AgreementMultilingualNmtLm, self).body
      outputs, aux_losses = nmt_body(features)
      # Build LM auxiliary losses.
      if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
        if self._hparams.lm_loss_coeff > 0:
          self.language_model = language_models.get(self._hparams.lm_type)
          # Make LM loss non-zero after the specified number of steps.
          batch_size = common_layers.shape_list(features["inputs"])[0]
          global_step = model_utils.get_global_step(self._hparams)
          aux_losses["language_model"] = tf.cond(
              global_step > self._hparams.lm_loss_enable_step,
              self._build_decoder_lm_loss,
              lambda: tf.zeros([batch_size]))

    return outputs, aux_losses


# ------------------------------------------------------------------------------
# Hparams
# ------------------------------------------------------------------------------


def base_agreement(hparams):
  """Adds base hparams for AgreementMultilingualNmt."""
  # Encoder agreement.
  hparams.add_hparam("enc_agreement_coeff", 0.0001)
  hparams.add_hparam("enc_agreement_loss", "cosine")
  hparams.add_hparam("enc_agreement_pool", True)
  hparams.add_hparam("enc_agreement_enable_step", 100000)

  # Decoder agreement.
  hparams.add_hparam("aux_decode_length", 40)
  hparams.add_hparam("dec_agreement_coeff", 0.001)
  hparams.add_hparam("dec_agreement_loss_sparse", False)
  hparams.add_hparam("dec_agreement_enable_step", 100000)

  return hparams


def base_lm(hparams):
  """Adds base hparams for LM."""

  # Language model.
  hparams.add_hparam("lm_type", "left2right")
  hparams.add_hparam("lm_num_layers", 2)
  hparams.add_hparam("lm_num_residual_layers", 1)

  # Language model training and loss.
  hparams.add_hparam("lm_do_train", False)
  hparams.add_hparam("lm_loss_coeff", 0.001)
  hparams.add_hparam("lm_loss_enable_step", 400000)

  return hparams


## Bidirectional architectures.


@registry.register_hparams
def ag_nmt_bilstm_bahdanau_att():
  """Hparams for LSTM with bahdanau attention."""
  hparams = basic.basic_nmt_bilstm_bahdanau_att()
  hparams = base_agreement(hparams)
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_bahdanau_att_lm():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_nmt_bilstm_bahdanau_att()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_luong_att():
  """Hparams for LSTM with luong attention."""
  hparams = basic.basic_nmt_bilstm_luong_att()
  hparams = base_agreement(hparams)
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_luong_att_lm():
  """Hparams for LSTM with luong attention."""
  hparams = ag_nmt_bilstm_luong_att()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_bahdanau_att_multi():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_nmt_bilstm_bahdanau_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_bahdanau_att_multi_lm():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_nmt_bilstm_bahdanau_att_multi()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_luong_att_multi():
  """Hparams for LSTM with luong attention."""
  hparams = ag_nmt_bilstm_luong_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def ag_nmt_bilstm_luong_att_multi_lm():
  """Hparams for LSTM with luong attention."""
  hparams = ag_nmt_bilstm_luong_att_multi()
  hparams = base_lm(hparams)
  return hparams


## GNMT architectures.


@registry.register_hparams
def ag_gnmt_bahdanau_att():
  """Hparams for LSTM with bahdanau attention."""
  hparams = basic.basic_gnmt_bahdanau_att()
  hparams = base_agreement(hparams)
  return hparams


@registry.register_hparams
def ag_gnmt_bahdanau_att_lm():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_gnmt_bahdanau_att()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_gnmt_luong_att():
  """Hparams for LSTM with luong attention."""
  hparams = basic.basic_gnmt_luong_att()
  hparams = base_agreement(hparams)
  return hparams


@registry.register_hparams
def ag_gnmt_luong_att_lm():
  """Hparams for LSTM with luong attention."""
  hparams = ag_gnmt_luong_att()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_gnmt_bahdanau_att_multi():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_gnmt_bahdanau_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def ag_gnmt_bahdanau_att_multi_lm():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_gnmt_bahdanau_att_multi()
  hparams = base_lm(hparams)
  return hparams


@registry.register_hparams
def ag_gnmt_luong_att_multi():
  """Hparams for LSTM with luong attention."""
  hparams = ag_gnmt_luong_att()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def ag_gnmt_luong_att_multi_lm():
  """Hparams for LSTM with bahdanau attention."""
  hparams = ag_gnmt_luong_att_multi()
  hparams = base_lm(hparams)
  return hparams

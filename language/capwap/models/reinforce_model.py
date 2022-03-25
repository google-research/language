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
"""Captioning model with reinforcement learning on QA pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.tpu import tpu_summaries
from bert import optimization
from language.capwap import datasets
from language.capwap.utils import checkpoint_utils
from language.capwap.utils import reward_utils
from language.capwap.utils import tensor_utils
from language.capwap.utils import transformer_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub


def model_fn(features, labels, mode, params, vocab):
  """Model function that satisfies the Estimator API.

  Args:
    features: Dictionary of model input tensors.
    labels: Ununsed.
    mode: A tf.estimator.ModeKeys value.
    params: Dictionary of model parameters.
    vocab: A utils.text_utils.Vocab instance.

  Returns:
    spec: A tf.estimator.TPUEstimatorSpec.
  """
  del labels

  # ----------------------------------------------------------------------------
  # INITIALIZATION.
  # ----------------------------------------------------------------------------

  # Update model config from the pre-trained checkpoint.
  model = transformer_utils.TransformerModel(
      config=transformer_utils.TransformerConfig.from_dict(params),
      is_training=(mode == tf_estimator.ModeKeys.TRAIN))

  # Initialize QA model.
  rc_model = hub.Module(params["rc_model"])

  # image_features: [batch_size, num_regions, feature_size]
  # image_positions: [batch_size, num_regions]
  # image_mask: [batch_size, num_regions]
  image_features = features["object_features"].features
  image_positions = features["object_features"].positions
  image_mask = features["object_features"].mask

  # Expand mask by 1 to account for the leading [IMG] token.
  # [batch_size, num_regions + 1]
  batch_size = tensor_utils.shape(image_mask, 0)
  input_mask = tf.pad(image_mask, [[0, 0], [1, 0]], constant_values=1)

  # Encode the image and store the cached transformer values.
  # [batch_size, num_regions + 1, num_layers, num_heads, head_size]
  _, input_cache = model.compute_image_transformer(
      input_ids=tf.fill([batch_size, 1], vocab.t2i(vocab.IMG)),
      input_image=image_features,
      input_image_mask=input_mask,
      input_positions=image_positions)

  # ----------------------------------------------------------------------------
  # TRAINING
  # ----------------------------------------------------------------------------

  if mode == tf_estimator.ModeKeys.TRAIN:
    # MIXER-style training objective consists of two parts:
    #   1) Policy gradient on rewarded rollouts.
    #   2) MLE regularization on references.
    # The full loss is L_total = L_pg + L_mle.

    # Step 1: Policy gradient.
    # Compute and score policy rollouts (multiple per image).
    rollouts = reward_utils.compute_rollouts(
        model=model,
        rc_model=rc_model,
        features=features,
        encoder_cache=input_cache,
        encoder_cache_mask=input_mask,
        vocab=vocab,
        params=params)

    # Using a self-critical baseline, R'(y) = R(y) - b where b = argmax p(y|x),
    # sample a single rollout with non-zero reward.
    rollout, reward = reward_utils.sample_from_rollouts(
        rollouts=rollouts,
        baseline=rollouts.rewards[params["reward"]][:, 0],
        reward_type=params["reward"])

    # Compute the probablity of the rollout (back-propable).
    # [batch_size, decode_length, input_length + decode_length]
    rollout_attention_mask = transformer_utils.compute_attention_mask(
        token_mask=rollout.mask[:, :-1], input_mask=input_mask)

    # [batch_size, decode_length, vocab_size]
    rollout_emb, _ = model.compute_transformer(
        input_ids=rollout.token_ids[:, :-1],
        input_segment_id=rollout.segment_ids[:, :-1],
        input_positions=rollout.positions[:, :-1],
        attention_mask=rollout_attention_mask,
        input_cache=input_cache,
        reuse=tf.AUTO_REUSE)

    # [batch_size, decode_length, vocab_size]
    rollout_logits = model.compute_logits(rollout_emb, reuse=tf.AUTO_REUSE)

    # Compute the RL loss, -R(y) * log p(y|x)
    # Some elements in this batch are MLE only, mask those out from the loss.
    rollout_mask = tf.cast(rollout.mask[:, 1:], tf.float32)
    pg_mask = tf.equal(features["input_type"], datasets.DatasetTypes.VQA)
    rollout_mask *= tf.expand_dims(tf.cast(pg_mask, tf.float32), 1)
    rl_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=rollout.token_ids[:, 1:],
        logits=rollout_logits,
        weights=tf.expand_dims(reward, 1) * rollout_mask,
        reduction=tf.losses.Reduction.SUM)
    rl_loss = tf.math.divide_no_nan(rl_loss, tf.reduce_sum(rollout_mask))

    # Step 2: MLE on references.
    # [batch_size, decode_length, input_length + decode_length]
    reference_attention_mask = transformer_utils.compute_attention_mask(
        token_mask=features["token_inputs"].mask, input_mask=input_mask)

    # [batch_size, decode_length, hidden_size]
    target_emb, _ = model.compute_transformer(
        input_ids=features["token_inputs"].token_ids,
        input_segment_id=features["token_inputs"].segment_ids,
        input_positions=features["token_inputs"].positions,
        attention_mask=reference_attention_mask,
        input_cache=input_cache,
        reuse=tf.AUTO_REUSE)

    # [batch_size, decode_length, vocab_size]
    target_logits = model.compute_logits(target_emb, reuse=tf.AUTO_REUSE)

    # Compute the MLE objective (cross-entropy loss).
    weights = features["token_outputs"].mask
    ref_mask = tf.equal(features["input_type"], datasets.DatasetTypes.REFERENCE)
    weights *= tf.expand_dims(tf.cast(ref_mask, tf.int32), 1)
    reference_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=features["token_outputs"].token_ids,
        logits=target_logits,
        weights=weights)

    # Add both losses together.
    loss = rl_loss + reference_loss

    # BERT-style optimization with linear warmp.
    train_op = optimization.create_optimizer(
        loss=loss,
        init_lr=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=params["num_warmup_steps"],
        use_tpu=params.get("use_tpu"))

    # Book-keeping.
    summaries = tpu_summaries.TpuSummaries(params["model_dir"])
    summaries.scalar("loss", loss)

    # Check what percentage of examples have non-zero reward.
    total_vqa = tf.reduce_sum(tf.cast(pg_mask, tf.float32))
    nonzero = tf.cast(tf.not_equal(reward, 0), tf.float32)
    nonzero *= tf.cast(pg_mask, tf.float32)
    total_nonzero = tf.reduce_sum(nonzero)
    summaries.scalar("density", tf.div_no_nan(total_nonzero, total_vqa))

    # Total (non-normalized) reward.
    reward = rollouts.rewards[params["reward"]][:, 0]
    reward *= tf.cast(pg_mask, tf.float32)
    total_reward = tf.reduce_sum(reward)
    summaries.scalar("reward", tf.div_no_nan(total_reward, total_vqa))
    host_call = summaries.get_host_call()
  else:
    loss = None
    train_op = None
    host_call = None

  # ----------------------------------------------------------------------------
  # TESTING.
  # ----------------------------------------------------------------------------

  if mode == tf_estimator.ModeKeys.PREDICT:
    decode_output = transformer_utils.beam_search_decode(
        model=model,
        encoder_cache=input_cache,
        encoder_cache_mask=input_mask,
        start_id=vocab.t2i(vocab.CLS),
        stop_id=vocab.t2i(vocab.SEP),
        segment_id=0,
        num_steps=params["decode_length"],
        beam_size=params["beam_size"],
        alpha=params["beam_length_penalty"],
        reuse=tf.AUTO_REUSE)
    predictions = dict(
        image_id=features.get("image_id", -1),
        question_id=features.get("question_id", -1),
        token_ids=decode_output.token_ids[:, :, 1:])
  else:
    predictions = None

  # ----------------------------------------------------------------------------
  # WARM-START.
  # ----------------------------------------------------------------------------

  # Initialize from pretrained model.
  def scaffold_fn():
    """Init op run on host."""
    checkpoint = params["base_model"]
    if params["warm_start_path"]:
      checkpoint = params["warm_start_path"]
    if checkpoint:
      checkpoint_utils.init_from_checkpoint(checkpoint)
    return tf.train.Scaffold()

  return tf_estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions=predictions,
      scaffold_fn=scaffold_fn,
      host_call=host_call,
  )

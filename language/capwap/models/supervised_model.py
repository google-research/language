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
"""Baseline captioning model with MLE objective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.tpu import tpu_summaries
from bert import optimization
from language.capwap.utils import checkpoint_utils
from language.capwap.utils import tensor_utils
from language.capwap.utils import transformer_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


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

  model = transformer_utils.TransformerModel(
      config=transformer_utils.TransformerConfig.from_dict(params),
      is_training=(mode == tf_estimator.ModeKeys.TRAIN))

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

  if params.get("conditional_decoding"):
    # Add additional (text) conditioning information to the input cache.
    # The conditioning information gets to see the image information.
    # The new input consists of both the image and the extra encoded text.
    # This is used for the LEARN function of Alg. 1 in the paper.

    # [batch_size, num_regions + condition_length + 1]
    input_mask = tf.concat([input_mask, features["condition_inputs"].mask], 1)

    # [batch_size, condition_length, num_layers, num_heads, head_size]
    _, condition_cache = model.compute_transformer(
        input_ids=features["condition_inputs"].token_ids,
        input_segment_id=features["condition_inputs"].segment_ids,
        input_positions=features["condition_inputs"].positions,
        attention_mask=tf.expand_dims(input_mask, 1),
        input_cache=input_cache,
        reuse=tf.AUTO_REUSE,
        conditional=True)

    # [batch_size, input_length, num_layers, num_heads, head_size]
    input_cache = transformer_utils.TransformerCache(
        keys=tf.concat([input_cache.keys, condition_cache.keys], 1),
        values=tf.concat([input_cache.values, condition_cache.values], 1))

  # ----------------------------------------------------------------------------
  # TRAINING
  # ----------------------------------------------------------------------------

  if mode == tf_estimator.ModeKeys.TRAIN:
    # During training, apply forced decoding with a diagonal attention mask.
    # [batch_size, caption_length - 1, input_length + caption_length - 1]
    attention_mask = transformer_utils.compute_attention_mask(
        token_mask=features["token_inputs"].mask, input_mask=input_mask)

    # [batch_size, caption_length - 1, hidden_size]
    target_emb, _ = model.compute_transformer(
        input_ids=features["token_inputs"].token_ids,
        input_segment_id=features["token_inputs"].segment_ids,
        input_positions=features["token_inputs"].positions,
        attention_mask=attention_mask,
        input_cache=input_cache,
        reuse=tf.AUTO_REUSE)

    # [batch_size, caption_length - 1, vocab_size]
    target_logits = model.compute_logits(target_emb, reuse=tf.AUTO_REUSE)

    # Compute the MLE objective (cross-entropy loss).
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=features["token_outputs"].token_ids,
        logits=target_logits,
        weights=features["token_outputs"].mask)

    # BERT-style optimization with linear warmp.
    train_op = optimization.create_optimizer(
        loss=loss,
        init_lr=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=params["num_warmup_steps"],
        use_tpu=params.get("use_tpu"))

    summaries = tpu_summaries.TpuSummaries(params["model_dir"])
    summaries.scalar("loss", loss)
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
    checkpoint = params.get("warm_start_path")
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

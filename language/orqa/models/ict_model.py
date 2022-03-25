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
"""Inverse Cloze Task model."""
import functools

from bert import optimization
from language.common.utils import tensor_utils
from language.common.utils import tpu_utils
from language.orqa.datasets import ict_dataset
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub


def module_fn(is_training, params):
  """Module function."""
  input_ids = tf.placeholder(tf.int32, [None, None], "input_ids")
  input_mask = tf.placeholder(tf.int32, [None, None], "input_mask")
  segment_ids = tf.placeholder(tf.int32, [None, None], "segment_ids")

  bert_module = hub.Module(
      params["bert_hub_module_path"],
      tags={"train"} if is_training else {},
      trainable=True)

  output_layer = bert_module(
      inputs=dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids),
      signature="tokens",
      as_dict=True)["pooled_output"]
  projected_emb = tf.layers.dense(output_layer, params["projection_size"])
  projected_emb = tf.keras.layers.LayerNormalization(axis=-1)(projected_emb)
  if is_training:
    projected_emb = tf.nn.dropout(projected_emb, rate=0.1)

  hub.add_signature(
      name="projected",
      inputs=dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids),
      outputs=projected_emb)

  hub.add_signature(
      name="tokenization_info",
      inputs={},
      outputs=bert_module(signature="tokenization_info", as_dict=True))


def create_ict_module(params, mode):
  """Create hub module."""
  tags_and_args = []
  for is_training in (True, False):
    tags = set()
    if is_training:
      tags.add("train")
    tags_and_args.append((tags, dict(is_training=is_training)))
  ict_module_spec = hub.create_module_spec(
      functools.partial(module_fn, params=params),
      tags_and_args=tags_and_args)
  ict_module = hub.Module(
      ict_module_spec,
      tags={"train"} if mode == tf_estimator.ModeKeys.TRAIN else {},
      trainable=True)
  hub.register_module_for_export(ict_module, "ict")
  return ict_module


def model_fn(features, labels, mode, params):
  """Model function."""
  del labels

  # [local_batch_size, block_seq_len]
  block_ids = features["block_ids"]
  block_mask = features["block_mask"]
  block_segment_ids = features["block_segment_ids"]

  # [local_batch_size, query_seq_len]
  query_ids = features["query_ids"]
  query_mask = features["query_mask"]

  local_batch_size = tensor_utils.shape(block_ids, 0)
  tf.logging.info("Model batch size: %d", local_batch_size)

  ict_module = create_ict_module(params, mode)

  query_emb = ict_module(
      inputs=dict(
          input_ids=query_ids,
          input_mask=query_mask,
          segment_ids=tf.zeros_like(query_ids)),
      signature="projected")
  block_emb = ict_module(
      inputs=dict(
          input_ids=block_ids,
          input_mask=block_mask,
          segment_ids=block_segment_ids),
      signature="projected")

  if params["use_tpu"]:
    # [global_batch_size, hidden_size]
    block_emb = tpu_utils.cross_shard_concat(block_emb)

    # [global_batch_size, local_batch_size]
    labels = tpu_utils.cross_shard_pad(tf.eye(local_batch_size))

    # [local_batch_size]
    labels = tf.argmax(labels, 0)
  else:
    # [local_batch_size]
    labels = tf.range(local_batch_size)

  tf.logging.info("Global batch size: %s", tensor_utils.shape(block_emb, 0))

  # [batch_size, global_batch_size]
  logits = tf.matmul(query_emb, block_emb, transpose_b=True)

  # []
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  train_op = optimization.create_optimizer(
      loss=loss,
      init_lr=params["learning_rate"],
      num_train_steps=params["num_train_steps"],
      num_warmup_steps=min(10000, max(100, int(params["num_train_steps"]/10))),
      use_tpu=params["use_tpu"] if "use_tpu" in params else False)

  predictions = tf.argmax(logits, -1)

  metric_args = [query_mask, block_mask, labels, predictions,
                 features["mask_query"]]
  def metric_fn(query_mask, block_mask, labels, predictions, mask_query):
    masked_accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predictions,
        weights=mask_query)
    unmasked_accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predictions,
        weights=tf.logical_not(mask_query))
    return dict(
        query_non_padding=tf.metrics.mean(query_mask),
        block_non_padding=tf.metrics.mean(block_mask),
        actual_mask_ratio=tf.metrics.mean(mask_query),
        masked_accuracy=masked_accuracy,
        unmasked_accuracy=unmasked_accuracy)

  if params["use_tpu"]:
    return tf_estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metrics=(metric_fn, metric_args))
  else:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metric_fn(*metric_args),
        predictions=predictions)


def input_fn(params, is_train):
  """An input function satisfying the tf.estimator API."""
  dataset = ict_dataset.get_dataset(
      examples_path=params["examples_path"],
      mask_rate=params["mask_rate"],
      bert_hub_module_path=params["bert_hub_module_path"],
      query_seq_len=params["query_seq_len"],
      block_seq_len=params["block_seq_len"],
      num_block_records=params["num_block_records"],
      num_input_threads=params["num_input_threads"])
  batch_size = params["batch_size"] if is_train else params["eval_batch_size"]
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def exporter():
  """Create exporters."""
  serving_input_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(
      features=dict(
          block_ids=tf.placeholder(tf.int32, [None, None]),
          block_mask=tf.placeholder(tf.int32, [None, None]),
          block_segment_ids=tf.placeholder(tf.int32, [None, None]),
          query_ids=tf.placeholder(tf.int32, [None, None]),
          query_mask=tf.placeholder(tf.int32, [None, None]),
          mask_query=tf.placeholder(tf.bool, [None])),
      default_batch_size=8)
  return hub.LatestModuleExporter("tf_hub", serving_input_fn, exports_to_keep=1)

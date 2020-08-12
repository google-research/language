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
r"""Implementation of the Retrieval Augmented Masked Language Model."""


import collections


from bert import optimization
from language.common.utils import exporters
from language.common.utils import nest_utils
from language.common.utils import tensor_utils
from language.realm import featurization
from language.realm import preprocessing
from tensorflow.compat import v1 as tf
import tensorflow_hub as hub


def model_fn(features, labels, mode, params):
  """Model function."""
  del labels

  # ==============================
  # Input features
  # ==============================
  # [batch_size, query_seq_len]
  query_inputs = features["query_inputs"]

  # [batch_size, num_candidates, candidate_seq_len]
  candidate_inputs = features["candidate_inputs"]

  # [batch_size, num_candidates, query_seq_len + candidate_seq_len]
  joint_inputs = features["joint_inputs"]

  # [batch_size, num_masks]
  mlm_targets = features["mlm_targets"]
  mlm_positions = features["mlm_positions"]
  mlm_mask = features["mlm_mask"]

  # ==============================
  # Create modules.
  # ==============================
  bert_module = hub.Module(
      spec=params["bert_hub_module_handle"],
      name="bert",
      tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
      trainable=True)
  hub.register_module_for_export(bert_module, "bert")

  embedder_module = hub.Module(
      spec=params["embedder_hub_module_handle"],
      name="embedder",
      tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
      trainable=True)
  hub.register_module_for_export(embedder_module, "embedder")

  # ==============================
  # Retrieve.
  # ==============================
  # [batch_size, projected_size]
  query_emb = embedder_module(
      inputs=dict(
          input_ids=query_inputs.token_ids,
          input_mask=query_inputs.mask,
          segment_ids=query_inputs.segment_ids),
      signature="projected")

  # [batch_size * num_candidates, candidate_seq_len]
  flat_candidate_inputs, unflatten = flatten_bert_inputs(
      candidate_inputs)

  # [batch_size * num_candidates, projected_size]
  flat_candidate_emb = embedder_module(
      inputs=dict(
          input_ids=flat_candidate_inputs.token_ids,
          input_mask=flat_candidate_inputs.mask,
          segment_ids=flat_candidate_inputs.segment_ids),
      signature="projected")

  # [batch_size, num_candidates, projected_size]
  unflattened_candidate_emb = unflatten(flat_candidate_emb)

  # [batch_size, num_candidates]
  retrieval_score = tf.einsum("BD,BND->BN", query_emb,
                              unflattened_candidate_emb)

  # ==============================
  # Read.
  # ==============================
  # [batch_size * num_candidates, query_seq_len + candidate_seq_len]
  flat_joint_inputs, unflatten = flatten_bert_inputs(joint_inputs)

  # [batch_size * num_candidates, num_masks]
  flat_mlm_positions, _ = tensor_utils.flatten(
      tf.tile(
          tf.expand_dims(mlm_positions, 1), [1, params["num_candidates"], 1]))

  batch_size, num_masks = tensor_utils.shape(mlm_targets)

  # [batch_size * num_candidates, query_seq_len + candidates_seq_len]
  flat_joint_bert_outputs = bert_module(
      inputs=dict(
          input_ids=flat_joint_inputs.token_ids,
          input_mask=flat_joint_inputs.mask,
          segment_ids=flat_joint_inputs.segment_ids,
          mlm_positions=flat_mlm_positions),
      signature="mlm",
      as_dict=True)

  # [batch_size, num_candidates]
  candidate_score = retrieval_score

  # [batch_size, num_candidates]
  candidate_log_probs = tf.math.log_softmax(candidate_score)

  # ==============================
  # Compute marginal log-likelihood.
  # ==============================
  # [batch_size * num_candidates, num_masks]
  flat_mlm_logits = flat_joint_bert_outputs["mlm_logits"]

  # [batch_size, num_candidates, num_masks, vocab_size]
  mlm_logits = tf.reshape(
      flat_mlm_logits, [batch_size, params["num_candidates"], num_masks, -1])
  mlm_log_probs = tf.math.log_softmax(mlm_logits)

  # [batch_size, num_candidates, num_masks]
  tiled_mlm_targets = tf.tile(
      tf.expand_dims(mlm_targets, 1), [1, params["num_candidates"], 1])

  # [batch_size, num_candidates, num_masks, 1]
  tiled_mlm_targets = tf.expand_dims(tiled_mlm_targets, -1)

  # [batch_size, num_candidates, num_masks, 1]
  gold_log_probs = tf.batch_gather(mlm_log_probs, tiled_mlm_targets)

  # [batch_size, num_candidates, num_masks]
  gold_log_probs = tf.squeeze(gold_log_probs, -1)

  # [batch_size, num_candidates, num_masks]
  joint_gold_log_probs = (
      tf.expand_dims(candidate_log_probs, -1) + gold_log_probs)

  # [batch_size, num_masks]
  marginal_gold_log_probs = tf.reduce_logsumexp(joint_gold_log_probs, 1)

  # [batch_size, num_masks]
  float_mlm_mask = tf.cast(mlm_mask, tf.float32)

  # []
  loss = -tf.div_no_nan(
      tf.reduce_sum(marginal_gold_log_probs * float_mlm_mask),
      tf.reduce_sum(float_mlm_mask))

  # ==============================
  # Optimization
  # ==============================
  num_warmup_steps = min(10000, max(100, int(params["num_train_steps"] / 10)))
  train_op = optimization.create_optimizer(
      loss=loss,
      init_lr=params["learning_rate"],
      num_train_steps=params["num_train_steps"],
      num_warmup_steps=num_warmup_steps,
      use_tpu=params["use_tpu"])

  # ==============================
  # Evaluation
  # ==============================
  eval_metric_ops = None if params["use_tpu"] else dict()
  if mode != tf.estimator.ModeKeys.PREDICT:
    # [batch_size, num_masks]
    retrieval_utility = marginal_gold_log_probs - gold_log_probs[:, 0]
    retrieval_utility *= tf.cast(features["mlm_mask"], tf.float32)

    # []
    retrieval_utility = tf.div_no_nan(
        tf.reduce_sum(retrieval_utility), tf.reduce_sum(float_mlm_mask))
    add_mean_metric("retrieval_utility", retrieval_utility, eval_metric_ops)

    has_timestamp = tf.cast(
        tf.greater(features["export_timestamp"], 0), tf.float64)
    off_policy_delay_secs = (
        tf.timestamp() - tf.cast(features["export_timestamp"], tf.float64))
    off_policy_delay_mins = off_policy_delay_secs / 60.0
    off_policy_delay_mins *= tf.cast(has_timestamp, tf.float64)

    add_mean_metric("off_policy_delay_mins", off_policy_delay_mins,
                    eval_metric_ops)

  # Create empty predictions to avoid errors when running in prediction mode.
  predictions = dict()

  if params["use_tpu"]:
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions)
  else:
    if eval_metric_ops is not None:
      # Make sure the eval metrics are updated during training so that we get
      # quick feedback from tensorboard summaries when debugging locally.
      with tf.control_dependencies([u for _, u in eval_metric_ops.values()]):
        loss = tf.identity(loss)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        predictions=predictions)


def flatten_bert_inputs(bert_inputs):
  """Flatten all tensors in a BertInput and also return the inverse."""
  flat_token_ids, unflatten = tensor_utils.flatten(bert_inputs.token_ids)
  flat_mask, _ = tensor_utils.flatten(bert_inputs.mask)
  flat_segment_ids, _ = tensor_utils.flatten(bert_inputs.segment_ids)
  flat_bert_inputs = featurization.BertInputs(
      token_ids=flat_token_ids, mask=flat_mask, segment_ids=flat_segment_ids)
  return flat_bert_inputs, unflatten


def add_mean_metric(name, values, eval_metric_ops):
  """Add the mean of the values Tensor as a metric."""
  if eval_metric_ops is not None:
    v, u = tf.metrics.mean(values)
    # This line is required to make the statistics show up on Tensorboard.
    tf.summary.scalar(name, v)
    eval_metric_ops[name] = (v, u)


def load_featurizer(params):
  tokenizer = featurization.Tokenizer(
      vocab_path=params["vocab_path"],
      do_lower_case=params["do_lower_case"])

  return featurization.Featurizer(
      query_seq_len=params["query_seq_len"],
      candidate_seq_len=params["candidate_seq_len"],
      num_candidates=params["num_candidates"],
      max_masks=params["max_masks"],
      tokenizer=tokenizer)


def input_fn(params, is_train):
  """Input_fn satisfying TF Estimator spec."""
  featurizer = load_featurizer(params)

  if is_train:
    preprocessing_servers = params["train_preprocessing_servers"]
  else:
    preprocessing_servers = params["eval_preprocessing_servers"]

  # Preprocessing the raw text has already happened in the servers.
  dataset = get_dynamic_dataset(
      preprocessing_servers=preprocessing_servers,
      featurizer=featurizer,
      num_input_threads=params["num_input_threads"])

  if is_train:
    # The preprocessors already repeat the data.
    dataset = dataset.shuffle(10000)

  batch_size = params["batch_size"] if is_train else params["eval_batch_size"]
  dataset = dataset.batch(batch_size, drop_remainder=True)

  dataset = dataset.prefetch(32)

  return dataset


def get_exporters(params):
  """Create a collection of exporters."""

  def serving_input_fn():
    """Dummy serving input function.

    Returns:
      dummy: A dummy tuple that looks like a ServingInputReceiver as far as
             LatestModuleExporter is concerned.

    This doesn't need to return a real ServingInputReceiver since
    LatestModuleExporter only cares that you can reconstruct the prediction
    graph using some placeholder features.
    """
    featurizer = load_featurizer(params)
    batched_structure = tf.nest.map_structure(
        lambda ph: tf.expand_dims(ph, 0),
        featurizer.query_and_docs_feature_structure)
    return collections.namedtuple("Dummy", ["features"])(batched_structure)

  return [
      exporters.LatestExporterWithSteps("tf_hub", serving_input_fn),
      exporters.BestModuleExporter(
          "tf_hub_best", serving_input_fn,
          exporters.metric_compare_fn("retrieval_utility",
                                      lambda best, current: current > best))
  ]


def get_dynamic_dataset(preprocessing_servers,
                        featurizer,
                        num_input_threads):
  """Get dataset generated dynamically by a preprocessing job."""

  def _parse_example(serialized_example):
    return nest_utils.tf_example_to_structure(
        serialized_example, featurizer.query_and_docs_feature_structure)

  dataset = preprocessing.dataset_from_preprocessors(
      preprocessing_servers=preprocessing_servers,
      rpcs_per_tf_op=20,
  )
  return dataset.map(_parse_example, num_parallel_calls=num_input_threads)

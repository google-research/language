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
# coding=utf-8
"""Run BERT on SQuAD under Phrase-Indexed QA setting."""

import collections
import copy
import json
import math
import os
import re

from absl import flags
from bert import modeling
from bert import optimization
from bert import tokenization
from language.labs.drkit import evaluate
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("pretrain_data_dir", None,
                    "Directory containing pretraining files.")

flags.DEFINE_string("pretrain_tfrecord_file", None,
                    "A processed training TFRecords file.")

flags.DEFINE_string("eval_train_tfrecord_file", None,
                    "A processed eval training TFRecords file.")

flags.DEFINE_string("eval_eval_tfrecord_file", None,
                    "A processed eval eval TFRecords file.")

flags.DEFINE_string("eval_eval_features_file", None,
                    "A processed eval features file.")

flags.DEFINE_string("eval_eval_gt_file", None,
                    "Original eval file for evaluation.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 192,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 64,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 48,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_pretrain", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("pretrain_batch_size", 64,
                     "Total batch size for training.")

flags.DEFINE_integer("eval_train_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_integer("eval_predict_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_float(
    "normalize_emb", None, "Fixed norm to normalize document embeddings to. "
    "If None or 0.0, no normalization is done.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_pretrain_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("num_eval_train_epochs", 0.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 15,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_string(
    "doc_layers_to_use", "-1",
    "Comma-separated list of layer representations to use as the fixed "
    "document representation.")

flags.DEFINE_string(
    "doc_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`doc_layers`.")

flags.DEFINE_string(
    "qry_layers_to_use", "4",
    "Comma-separated list of layer representations to use as the fixed "
    "query representation.")

flags.DEFINE_string(
    "qry_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`qry_layers`.")

flags.DEFINE_integer(
    "projection_dim", 200, "Number of dimensions to project embeddings to. "
    "Set to None to use full dimensions.")

flags.DEFINE_boolean(
    "use_best_ckpt_for_predict", False,
    "If True, loads the best_model checkpoint in model_dir, "
    "instead of the latest one.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("num_eval_examples", None,
                     "Number of evaluation examples.")

flags.DEFINE_integer("random_seed", 1, "Random seed for reproducibility.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", True,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class BertModel(modeling.BertModel):
  """See modeling.BertModel."""

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None,
               reuse=False):
    """Constructor for BertModel which adds an option to reuse variables.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".
      reuse: (optional) if True, reuse previously initialized variables.

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = modeling.get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert", reuse=reuse):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.embedding_table) = (
            modeling.embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings))

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = modeling.embedding_postprocessor(
            input_tensor=self.word_embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = modeling.create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = modeling.transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(
                config.initializer_range))


class QAConfig:
  """Hyperparameters for the QA model."""

  def __init__(self, doc_layers_to_use, doc_aggregation_fn, qry_layers_to_use,
               qry_aggregation_fn, projection_dim, normalize_emb, share_bert,
               exclude_scopes):
    self.doc_layers_to_use = [int(vv) for vv in doc_layers_to_use.split(",")]
    self.doc_aggregation_fn = doc_aggregation_fn
    self.qry_layers_to_use = [int(vv) for vv in qry_layers_to_use.split(",")]
    self.qry_aggregation_fn = qry_aggregation_fn
    self.projection_dim = projection_dim
    self.normalize_emb = normalize_emb
    self.share_bert = share_bert
    self.exclude_scopes = exclude_scopes


def _get_bert_embeddings(model, layers_to_use, aggregation_fn, name="bert"):
  """Extract embeddings from BERT model."""
  all_hidden = model.get_all_encoder_layers()
  layers_hidden = [all_hidden[i] for i in layers_to_use]
  hidden_shapes = [
      modeling.get_shape_list(hid, expected_rank=3) for hid in all_hidden
  ]

  if len(layers_hidden) == 1:
    hidden_emb = layers_hidden[0]
    hidden_size = hidden_shapes[0][2]
  elif aggregation_fn == "concat":
    hidden_emb = tf.concat(layers_hidden, 2)
    hidden_size = sum([hidden_shapes[i][2] for i in layers_to_use])
  elif aggregation_fn == "average":
    hidden_size = hidden_shapes[0][2]
    assert all([shape[2] == hidden_size for shape in hidden_shapes
               ]), hidden_shapes
    hidden_emb = tf.add_n(layers_hidden) / len(layers_hidden)
  elif aggregation_fn == "attention":
    hidden_size = hidden_shapes[0][2]
    mixing_weights = tf.get_variable(
        name + "/mixing/weights", [len(layers_hidden)],
        initializer=tf.zeros_initializer())
    mixing_scores = tf.nn.softmax(mixing_weights)
    hidden_emb = tf.tensordot(
        tf.stack(layers_hidden, axis=-1), mixing_scores, [[-1], [0]])
  else:
    raise ValueError("Unrecognized aggregation function %s." % aggregation_fn)

  return hidden_emb, hidden_size


def create_model(bert_config, qa_config, is_training, doc_input_ids,
                 doc_input_mask, doc_segment_ids, qry_input_ids, qry_input_mask,
                 qry_segment_ids, use_one_hot_embeddings):
  """Creates a classification model."""
  tf.random.set_random_seed(FLAGS.random_seed)

  # document embedding
  doc_scope = "bert" if qa_config.share_bert else "doc_bert"
  doc_model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=doc_input_ids,
      input_mask=doc_input_mask,
      token_type_ids=doc_segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=doc_scope)

  doc_hidden, _ = _get_bert_embeddings(
      doc_model,
      qa_config.doc_layers_to_use,
      qa_config.doc_aggregation_fn,
      name="doc")

  # question shared encoder
  qry_scope = "bert" if qa_config.share_bert else "qry_bert"
  qry_model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=qry_input_ids,
      input_mask=qry_input_mask,
      token_type_ids=qry_segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=qry_scope,
      reuse=qa_config.share_bert)
  qry_seq_emb, _ = _get_bert_embeddings(
      qry_model,
      qa_config.qry_layers_to_use,
      qa_config.qry_aggregation_fn,
      name="qry")

  na_emb = tf.get_variable(
      "noanswer_emb", [1, bert_config.hidden_size],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer())
  # initializer=tf.zeros_initializer())

  if qa_config.projection_dim is not None:
    assert 2 * qa_config.projection_dim <= bert_config.hidden_size
    doc_hidden_st = doc_hidden[:, :, :qa_config.projection_dim]
    doc_hidden_en = doc_hidden[:, :, qa_config.projection_dim:2 *
                               qa_config.projection_dim]
    doc_hidden = doc_hidden[:, :, :2 * qa_config.projection_dim]
    qry_start_emb = qry_seq_emb[:, 0, :qa_config.projection_dim]
    qry_end_emb = qry_seq_emb[:, 0, qa_config.projection_dim:2 *
                              qa_config.projection_dim]
    na_emb_st = na_emb[:, :qa_config.projection_dim]
    na_emb_en = na_emb[:, qa_config.projection_dim:2 * qa_config.projection_dim]

  def _inner_logits(na, emb, qry):
    """Returns logits computed using inner product of qry and embeddings."""
    na_logit = tf.matmul(qry, na, transpose_b=True)
    logits = tf.reduce_sum(tf.expand_dims(qry, 1) * emb[:, 1:, :], 2)
    return tf.concat([na_logit, logits], axis=1)

  rel_start_logits = _inner_logits(na_emb_st, doc_hidden_st, qry_start_emb)
  rel_end_logits = _inner_logits(na_emb_en, doc_hidden_en, qry_end_emb)

  return rel_start_logits, rel_end_logits, doc_hidden, qry_start_emb, qry_end_emb


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     exclude_scopes):
  """Creates an optimizer training op, optionally excluding BERT vars."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf_estimator.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  exclude_vars = []
  if exclude_scopes:
    for sc in exclude_scopes:
      exclude_vars += list(
          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sc))
    # bert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bert")
    tvars = [vv for vv in tvars if vv not in exclude_vars]

  tf.logging.info("Training the following variables:")
  for vv in tvars:
    tf.logging.info(vv.name)

  tf.logging.info("NOT Training the following variables:")
  for vv in exclude_vars:
    tf.logging.info(vv.name)

  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


def get_assignment_map_from_checkpoint(tvars,
                                       init_checkpoint,
                                       my_scope,
                                       ckpt_scope,
                                       load_only_bert=False):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    if my_scope is not None:
      if name.startswith(my_scope):
        name = name[len(my_scope):]
      else:
        continue
    if load_only_bert and ("bert" not in name):
      continue
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if ckpt_scope is not None:
      if name.startswith(ckpt_scope):
        name = name[len(ckpt_scope):]
      else:
        continue
    if name not in name_to_variable:
      continue
    assignment_map[ckpt_scope + name] = name_to_variable[name]
    initialized_variable_names[my_scope + name] = 1
    initialized_variable_names[my_scope + name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def model_fn_builder(bert_config,
                     qa_config,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     summary_obj=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    unique_ids = features["unique_ids"]
    doc_input_ids = features["doc_input_ids"]
    doc_input_mask = features["doc_input_mask"]
    doc_segment_ids = features["doc_segment_ids"]
    qry_input_ids = features["qry_input_ids"]
    qry_input_mask = features["qry_input_mask"]
    qry_segment_ids = features["qry_segment_ids"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    start_logits, end_logits, doc_hidden, qry_start, qry_end = create_model(
        bert_config=bert_config,
        qa_config=qa_config,
        is_training=is_training,
        doc_input_ids=doc_input_ids,
        doc_input_mask=doc_input_mask,
        doc_segment_ids=doc_segment_ids,
        qry_input_ids=qry_input_ids,
        qry_input_mask=qry_input_mask,
        qry_segment_ids=qry_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      if qa_config.share_bert:
        (assignment_map,
         initialized_variable_names) = get_assignment_map_from_checkpoint(
             tvars, init_checkpoint, "bert/", "bert/")
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      else:
        (assignment_map_doc,
         initialized_variable_names_doc) = get_assignment_map_from_checkpoint(
             tvars, init_checkpoint, "doc_bert/", "bert/")
        (assignment_map_qry,
         initialized_variable_names_qry) = get_assignment_map_from_checkpoint(
             tvars, init_checkpoint, "qry_bert/", "bert/")
        initialized_variable_names = initialized_variable_names_doc
        initialized_variable_names.update(initialized_variable_names_qry)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map_doc)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map_qry)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map_doc)
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map_qry)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(doc_input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2.0

      if summary_obj is not None:
        summary_obj.scalar("Total Loss", tf.expand_dims(total_loss, 0))

      train_op = create_optimizer(
          total_loss,
          learning_rate,
          num_train_steps,
          num_warmup_steps,
          use_tpu,
          exclude_scopes=qa_config.exclude_scopes)

      host_call = summary_obj.get_host_call() if summary_obj else None
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "doc_features": doc_hidden,
          "qry_st_features": qry_start,
          "qry_en_features": qry_end,
      }
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_files, seq_length, qry_length, is_training,
                     drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "doc_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "doc_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "doc_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "qry_input_ids": tf.FixedLenFeature([qry_length], tf.int64),
      "qry_input_mask": tf.FixedLenFeature([qry_length], tf.int64),
      "qry_segment_ids": tf.FixedLenFeature([qry_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.Dataset.from_tensor_slices(input_files).interleave(
        lambda x: tf.data.TFRecordDataset(x),  # pylint: disable=unnecessary-lambda
        cycle_length=len(input_files),
        block_length=1)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s", output_prediction_file)
  tf.logging.info("Writing nbest to: %s", output_nbest_file)

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.doc_tokens):
            continue
          if end_index >= len(feature.doc_tokens):
            continue
          if start_index not in feature.doc_token_to_orig_map:
            continue
          if end_index not in feature.doc_token_to_orig_map:
            continue
          if not feature.doc_token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index],
              ))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit,
          ))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.doc_tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.doc_token_to_orig_map[pred.start_index]
        orig_doc_end = feature.doc_token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
          ))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
    assert len(nbest) >= 1

    # if we didn't include the empty option in the n-best, include it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    if not best_non_null_entry:
      best_non_null_entry = _NbestPrediction(
          text="empty", start_logit=0.0, end_logit=0.0)

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info("Unable to find text: '%s' in '%s'", pred_text, orig_text)
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def train(tfrecord_filenames, estimator, num_train_steps):
  """Run one training loop over given TFRecords file."""
  train_input_fn = input_fn_builder(
      input_files=tfrecord_filenames,
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


def single_eval(eval_tfrecord_filename, estimator, eval_examples, eval_features,
                ckpt_path, output_prediction_file, output_nbest_file,
                output_null_log_odds_file):
  """Run one evaluation using given checkpoint."""
  tf.logging.info("***** Running predictions using %s *****", ckpt_path)
  tf.logging.info("  Num orig examples = %d", len(eval_examples))
  tf.logging.info("  Num split examples = %d", len(eval_features))

  all_results = []

  predict_input_fn = input_fn_builder(
      input_files=[eval_tfrecord_filename],
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      is_training=False,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_results = []
  for result in estimator.predict(
      predict_input_fn, yield_single_examples=True, checkpoint_path=ckpt_path):
    if len(all_results) % 1000 == 0:
      tf.logging.info("Processing example: %d", len(all_results))
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits,
        ))

  write_predictions(eval_examples, eval_features, all_results,
                    FLAGS.n_best_size, FLAGS.max_answer_length,
                    FLAGS.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file)


def _copy_model(in_path, out_path):
  """Copy model checkpoint for future use."""
  tf.logging.info("Copying checkpoint from %s to %s.", in_path, out_path)
  tf.gfile.Copy(
      in_path + ".data-00000-of-00001",
      out_path + ".data-00000-of-00001",
      overwrite=True)
  tf.gfile.Copy(in_path + ".index", out_path + ".index", overwrite=True)
  tf.gfile.Copy(in_path + ".meta", out_path + ".meta", overwrite=True)


class Example:
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position,)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position,)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible,)
    return s


class InputFeatures:
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               qas_id,
               example_index,
               doc_span_index,
               doc_tokens,
               doc_token_to_orig_map,
               doc_token_is_max_context,
               doc_input_ids,
               doc_input_mask,
               doc_segment_ids,
               qry_tokens,
               qry_input_ids,
               qry_input_mask,
               qry_segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.qas_id = qas_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.doc_tokens = doc_tokens
    self.doc_token_to_orig_map = {
        int(k): v for k, v in doc_token_to_orig_map.items()
    }
    self.doc_token_is_max_context = {
        int(k): v for k, v in doc_token_is_max_context.items()
    }
    self.doc_input_ids = doc_input_ids
    self.doc_input_mask = doc_input_mask
    self.doc_segment_ids = doc_segment_ids
    self.qry_tokens = qry_tokens
    self.qry_input_ids = qry_input_ids
    self.qry_input_mask = qry_input_mask
    self.qry_segment_ids = qry_segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  qa_config = QAConfig(
      doc_layers_to_use=FLAGS.doc_layers_to_use,
      doc_aggregation_fn=FLAGS.doc_aggregation_fn,
      qry_layers_to_use=FLAGS.qry_layers_to_use,
      qry_aggregation_fn=FLAGS.qry_aggregation_fn,
      projection_dim=FLAGS.projection_dim,
      normalize_emb=FLAGS.normalize_emb,
      share_bert=True,
      exclude_scopes=[])

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2

  json.dump(
      tf.app.flags.FLAGS.flag_values_dict(),
      tf.gfile.Open(os.path.join(FLAGS.output_dir, "pretrain_flags.json"), "w"))

  if FLAGS.do_pretrain:
    run_config = tf_estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf_estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    summary_obj = None

    train_files = [
        os.path.join(FLAGS.pretrain_data_dir, f + ".tf_record")
        for f in FLAGS.pretrain_tfrecord_file.split(":")
    ]
    num_examples = 1e10
    for f in train_files:
      my_examples = sum(1 for _ in tf.python_io.tf_record_iterator(f))
      if my_examples < num_examples:
        num_examples = my_examples
    num_train_steps = len(train_files) * int(
        num_examples / FLAGS.pretrain_batch_size * FLAGS.num_pretrain_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    qa_config.share_bert = True
    model_fn = model_fn_builder(
        bert_config=bert_config,
        qa_config=qa_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        summary_obj=summary_obj)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf_estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.pretrain_batch_size,
        predict_batch_size=FLAGS.pretrain_batch_size)

    tf.logging.info("***** Running pretraining *****")
    tf.logging.info("  Num orig examples = %d", num_examples)
    tf.logging.info("  Batch size = %d", FLAGS.pretrain_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train(train_files, estimator, num_train_steps)

  if FLAGS.do_eval:
    if not tf.gfile.Exists(os.path.join(FLAGS.output_dir, "eval_log")):
      tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, "eval_log"))
    event_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.output_dir, "eval_log"))
    init_ckpt_path = tf.train.latest_checkpoint(FLAGS.output_dir)
    if init_ckpt_path is None:
      return
    ckpt_number = int(init_ckpt_path.rsplit("-", 1)[1])
    my_output_dir = os.path.join(FLAGS.output_dir, "eval", str(ckpt_number))
    tf.gfile.MakeDirs(my_output_dir)

    run_config = tf_estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=my_output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf_estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    eval_train_files = FLAGS.eval_train_tfrecord_file.split(",")
    num_examples = 0
    for f in eval_train_files:
      num_examples += sum(1 for _ in tf.python_io.tf_record_iterator(f))
    num_eval_train_steps = int(num_examples / FLAGS.eval_train_batch_size *
                               FLAGS.num_eval_train_epochs)
    num_warmup_steps = int(num_eval_train_steps * FLAGS.warmup_proportion)

    qa_config.share_bert = False
    qa_config.exclude_scopes = "doc_bert"
    model_fn = model_fn_builder(
        bert_config=bert_config,
        qa_config=qa_config,
        init_checkpoint=init_ckpt_path,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_eval_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        summary_obj=None)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf_estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.eval_train_batch_size,
        predict_batch_size=FLAGS.eval_predict_batch_size)

    tf.logging.info("***** Running eval training *****")
    tf.logging.info("  Num orig examples = %d", num_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_train_batch_size)
    tf.logging.info("  Num steps = %d", num_eval_train_steps)
    if num_eval_train_steps > 0:
      train(eval_train_files, estimator, num_eval_train_steps)

    # Test
    ckpt_path = tf.train.latest_checkpoint(my_output_dir)
    eval_eval_files = FLAGS.eval_eval_tfrecord_file.split(",")
    eval_eval_jsons = FLAGS.eval_eval_features_file.split(",")
    eval_eval_gts = FLAGS.eval_eval_gt_file.split(",")
    assert len(eval_eval_files) == len(eval_eval_jsons), (
        FLAGS.eval_eval_tfrecord_file, FLAGS.eval_eval_features_file)
    for eval_file, eval_json, eval_gt in zip(eval_eval_files, eval_eval_jsons,
                                             eval_eval_gts):
      base_name = os.path.splitext(os.path.basename(eval_file))[0]
      eval_examples, eval_features = json.load(tf.gfile.Open(eval_json))
      eval_examples = [Example(**ee) for ee in eval_examples]
      eval_features = [InputFeatures(**ff) for ff in eval_features]
      num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(eval_file))
      output_prediction_file = os.path.join(my_output_dir,
                                            base_name + "_predictions.json")
      output_nbest_file = os.path.join(my_output_dir, base_name + "_nbest.json")
      output_null_log_odds_file = os.path.join(
          my_output_dir, base_name + "_null_log_odds.json")
      output_metric_file = os.path.join(my_output_dir,
                                        base_name + "_metrics.json")
      single_eval(eval_file, estimator, eval_examples, eval_features, ckpt_path,
                  output_prediction_file, output_nbest_file,
                  output_null_log_odds_file)
      metrics = evaluate.mrqa_eval_fn(eval_gt, output_prediction_file)
      json.dump(metrics, tf.gfile.Open(output_metric_file, "w"))
      for k, v in metrics.items():
        tf.logging.info("%s: %.3f", k, v)
        my_summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
        event_writer.add_summary(my_summary, global_step=ckpt_number)


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

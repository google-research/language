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
"""Models and experiments for ranking a set of passages with respect to query.

Includes dual encoder, full cross-attention, and other models.
"""

import os
import re

from absl import app
from absl import flags
from bert import modeling
from bert import optimization
from language.common.utils import tensor_utils
from language.common.utils import tpu_utils
from language.multivec.models import checkpoint_utils
from language.multivec.models.metrics import rank_metrics
from language.multivec.models.metrics import RawResult
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

FLAGS = flags.FLAGS
ismasked = "masked"
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("eval_name", "train", "Eval set to evaluate on.")

flags.DEFINE_bool("layer_norm", False, "use layer norm or l2")

flags.DEFINE_bool(
    "factored_model", True, "If true train Dual encoder, otherwise "
    "train cross-attention.")

flags.DEFINE_string(
    "file_pattern", "*", "A pattern to define training etc files"
    "e.g. empty or .(.*)")

flags.DEFINE_integer(
    "num_candidates", 8,
    "The number of canidates packed into single feature vectors for ranking. ")

flags.DEFINE_string(
    "output_base_dir", None,
    "The base of the output directory where the model checkpoints"
    "will be written.")

flags.DEFINE_string("ckpt_pattern", None,
                    "Glob pattern to get a list of ckpt to evaluate.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_query", 64,
    "The maximum total query input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_vec_query", 1, "The number of output query vectors."
    " M > 1 means the first M BERT output vectors")

flags.DEFINE_integer(
    "num_vec_passage", 1, "The number of output passage vectors."
    " M > 1 means the first M BERT output vectors")

flags.DEFINE_float("dropout", 0.9, "Dropout rate")

flags.DEFINE_integer("projection_size", 0,
                     "whether to project BERT embedding to lower dimension ")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("global_step", "0",
                     "The global step used for a file name.")

flags.DEFINE_string("mode", "continuous_eval",
                    "To train, eval, or continuous eval.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-06,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "The maximum number of checkpoints to keep.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

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

flags.DEFINE_string("tpu_job_name", None, "[Optional] Name of TPU job.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("data_dir", None,
                    "The  directory where the saved tf examples should be.")

flags.DEFINE_string("output_dir", None, "Final output directory.")


def get_file_names(root, file_pattern):
  glob = root + file_pattern + ".tf_record"
  tf.logging.info("glob pattern:" + glob)
  print("Checking glob " + glob)
  files = tf.io.gfile.glob(glob)
  return files


def _num_records(tf_record_files):
  c = 0
  for f in tf_record_files:
    for _ in tf.python_io.tf_record_iterator(f):
      c = c + 1
  return c


def file_based_input_fn_builder_files(input_files, seq_length, seq_length_query,
                                      num_candidates, is_training,
                                      drop_remainder):
  if not FLAGS.factored_model:
    return file_based_input_fn_builder_single_files(input_files, seq_length,
                                                    num_candidates, is_training,
                                                    drop_remainder)
  else:
    seq_length_passage = seq_length - seq_length_query
    return file_based_input_fn_builder_dual_files(input_files, seq_length_query,
                                                  seq_length_passage,
                                                  num_candidates, is_training,
                                                  drop_remainder)


def file_based_input_fn_builder(input_file, seq_length, seq_length_query,
                                num_candidates, is_training, drop_remainder):
  input_files = [input_file]
  return file_based_input_fn_builder_files(input_files, seq_length,
                                           seq_length_query, num_candidates,
                                           is_training, drop_remainder)


def file_based_input_fn_builder_dual_files(input_files,
                                           seq_length_query,
                                           seq_length_passage,
                                           num_candidates,
                                           is_training,
                                           drop_remainder,
                                           num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "q_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "cand_nums":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "input_ids_1":
          tf.FixedLenFeature([seq_length_query], tf.int64),
      "input_mask_1":
          tf.FixedLenFeature([seq_length_query], tf.int64),
      "segment_ids_1":
          tf.FixedLenFeature([seq_length_query], tf.int64),
      "input_masks_2":
          tf.FixedLenFeature([seq_length_passage * num_candidates], tf.int64),
      "segment_ids_2":
          tf.FixedLenFeature([seq_length_passage * num_candidates], tf.int64),
      "input_ids_2":
          tf.FixedLenFeature([seq_length_passage * num_candidates], tf.int64),
      "label_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "scores":
          tf.FixedLenFeature([num_candidates], tf.float32),
  }

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
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))
      cycle_length = min(num_cpu_threads, len(input_files))
      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)

    else:
      d = tf.data.TFRecordDataset(input_files)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def file_based_input_fn_builder_single(input_file, seq_length, num_candidates,
                                       is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "q_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "cand_nums":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "input_ids":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "input_mask":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "label_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
  }

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
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def file_based_input_fn_builder_single_files(input_files,
                                             seq_length,
                                             num_candidates,
                                             is_training,
                                             drop_remainder,
                                             num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "q_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "cand_nums":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "input_ids":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "input_mask":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([seq_length * num_candidates], tf.int64),
      "label_ids":
          tf.FixedLenFeature([num_candidates], tf.int64),
      "scores":
          tf.FixedLenFeature([num_candidates], tf.float32),
  }

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
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))
      cycle_length = min(num_cpu_threads, len(input_files))
      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)

    else:
      d = tf.data.TFRecordDataset(input_files)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_ca_model(bert_config, is_training, input_ids, input_mask,
                    segment_ids, num_candidates, sequence_length, labels,
                    use_one_hot_embeddings):
  """Creates a ranking model using cross attention representations."""

  input_ids = tf.reshape(input_ids, [-1, sequence_length])
  segment_ids = tf.reshape(segment_ids, [-1, sequence_length])
  input_mask = tf.reshape(input_mask, [-1, sequence_length])
  labels = tf.dtypes.cast(labels, tf.float32)
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1]

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))

  with tf.variable_scope("loss"):
    if is_training:
      output_layer = tf.nn.dropout(output_layer, keep_prob=FLAGS.dropout)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.reshape(logits, [-1, num_candidates])
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, logits)


def get_multi_vectors(model, input_masks, num_vec):
  """Get multi-vector representations.

  Args:
    model: BERT model
    input_masks: [batch_size, sequence_length]
    num_vec: scalar

  Returns:
    output_layer: [batch_size, num_vec, hidden_size],
    input_mask: [batch_size, num_vec]
  """
  output_layer = model.get_sequence_output()
  input_masks_ = input_masks
  input_masks = tf.expand_dims(tf.cast(input_masks, tf.float32), axis=2)
  output_layer = tf.multiply(output_layer, input_masks)
  return output_layer[:, :num_vec, :], tf.cast(input_masks_[:, :num_vec],
                                               tf.bool)


def encode_block(bert_config, input_ids, input_masks, segment_ids,
                 use_one_hot_embeddings, num_vec, is_training):
  """Encode text and get multi-vector representations."""
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_masks,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert")

  emb_dim = bert_config.hidden_size
  output_layer, mask = get_multi_vectors(model, input_masks, num_vec)
  # [batch_size, num_vec, hidden_size], [batch_size, num_vec]

  output_layer.set_shape([None, None, emb_dim])

  if FLAGS.projection_size > 0:
    with tf.variable_scope("projected_layer", reuse=tf.AUTO_REUSE):
      output_layer = tf.layers.dense(output_layer, FLAGS.projection_size)

    emb_dim = FLAGS.projection_size
    output_layer.set_shape([None, None, emb_dim])

  if FLAGS.layer_norm:
    output_layer = modeling.layer_norm(output_layer)
  else:
    output_layer = tf.math.l2_normalize(output_layer, axis=-1)
  return output_layer, mask


def create_de_model(bert_config, is_training, input_ids_1, input_mask_1,
                    segment_ids_1, input_ids_2, input_masks_2, segment_ids_2,
                    num_candidates, labels, use_one_hot_embeddings):
  """Creates a ranking model using cosine and dual encoder representations."""

  sequence_length_query = FLAGS.max_seq_length_query
  sequence_length_passage = FLAGS.max_seq_length - FLAGS.max_seq_length_query

  input_ids_1 = tf.reshape(input_ids_1, [-1, sequence_length_query])
  segment_ids_1 = tf.reshape(segment_ids_1, [-1, sequence_length_query])
  input_masks_1 = tf.reshape(input_mask_1, [-1, sequence_length_query])
  batch_size = tf.shape(input_masks_1)[0]

  input_ids_2 = tf.reshape(input_ids_2, [-1, sequence_length_passage])
  segment_ids_2 = tf.reshape(segment_ids_2, [-1, sequence_length_passage])
  input_masks_2 = tf.reshape(input_masks_2, [-1, sequence_length_passage])

  # [batch_size, num_candidates]
  labels = tf.dtypes.cast(labels, tf.float32)

  # [batch_size, num_vec_query, hidden_size], [batch_size, num_vec_query]
  output_layer_1, mask_1 = encode_block(bert_config, input_ids_1, input_masks_1,
                                        segment_ids_1, use_one_hot_embeddings,
                                        FLAGS.num_vec_query, is_training)

  output_layer_2, mask_2 = encode_block(bert_config, input_ids_2, input_masks_2,
                                        segment_ids_2, use_one_hot_embeddings,
                                        FLAGS.num_vec_passage, is_training)

  label_mask = tf.expand_dims(tf.eye(batch_size), axis=2)
  label_mask = tf.tile(label_mask, [1, 1, num_candidates])
  label_mask = tf.reshape(label_mask, [batch_size, -1])
  label_mask = tf.cast(label_mask, tf.float32)

  labels = tf.tile(labels, [1, batch_size])
  labels = tf.multiply(labels, label_mask)
  output_layer_2_logits = tf.reshape(
      output_layer_2, [batch_size, num_candidates, FLAGS.num_vec_passage, -1])
  mask_2_logits = tf.reshape(
      mask_2, [batch_size, num_candidates, FLAGS.num_vec_passage])
  mask_logits = tf.einsum("BQ,BCP->BCQP", tf.cast(mask_1, tf.float32),
                          tf.cast(mask_2_logits, tf.float32))

  logits = tf.einsum("BQH,BCPH->BCQP", output_layer_1, output_layer_2_logits)
  logits = tf.multiply(logits, mask_logits)
  logits = tf.reduce_max(logits, axis=-1)
  logits = tf.reduce_sum(logits, axis=-1)

  if FLAGS.use_tpu and is_training:
    num_shards = tpu_utils.num_tpu_shards()
    output_layer_2 = tpu_utils.cross_shard_concat(output_layer_2)
    mask_2 = tpu_utils.cross_shard_concat(tf.cast(mask_2, tf.float32))
    mask_2 = tf.cast(mask_2, tf.bool)
    labels = tpu_utils.cross_shard_pad(labels)
    tf.logging.info("Global batch size: %s", tensor_utils.shape(labels, 0))
    tf.logging.info("Num shards: %s", num_shards)
    tf.logging.info("Number of candidates in batch: %s",
                    tensor_utils.shape(output_layer_2, 0))
    labels = tf.reshape(labels, [num_shards, batch_size, -1])
    labels = tf.transpose(labels, perm=[1, 0, 2])
    labels = tf.reshape(labels, [batch_size, -1])

  with tf.variable_scope("loss"):
    if is_training:
      output_layer_1 = tf.nn.dropout(output_layer_1, keep_prob=FLAGS.dropout)
      output_layer_2 = tf.nn.dropout(output_layer_2, keep_prob=FLAGS.dropout)
    cosine_similarity = tf.einsum("AQH,BPH->ABQP", output_layer_1,
                                  output_layer_2)
    mask = tf.cast(
        tf.logical_and(
            tf.expand_dims(tf.expand_dims(mask_1, 2), 1),
            tf.expand_dims(tf.expand_dims(mask_2, 1), 0)), tf.float32)
    cosine_similarity = tf.multiply(cosine_similarity, mask)
    cosine_similarity = tf.reduce_max(cosine_similarity, axis=-1)
    cosine_similarity = tf.reduce_sum(cosine_similarity, axis=-1)
    per_example_loss = tf.losses.softmax_cross_entropy(labels,
                                                       cosine_similarity)

    return (per_example_loss, logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  if not FLAGS.factored_model:
    return model_fn_builder_single(bert_config, init_checkpoint, learning_rate,
                                   num_train_steps, num_warmup_steps, use_tpu,
                                   use_one_hot_embeddings)
  else:
    return model_fn_builder_dual(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings)


def model_fn_builder_dual(bert_config, init_checkpoint, learning_rate,
                          num_train_steps, num_warmup_steps, use_tpu,
                          use_one_hot_embeddings):
  """Returns model_fn for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del params, labels
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    q_ids = features["q_ids"]
    cand_nums = features["cand_nums"]
    input_ids_1 = features["input_ids_1"]
    input_mask_1 = features["input_mask_1"]
    segment_ids_1 = features["segment_ids_1"]
    input_ids_2 = features["input_ids_2"]
    input_masks_2 = features["input_masks_2"]
    segment_ids_2 = features["segment_ids_2"]
    label_ids = features["label_ids"]
    prior_scores = features["scores"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    (total_loss, logits) =\
        create_de_model(
            bert_config, is_training, input_ids_1, input_mask_1, segment_ids_1,
            input_ids_2, input_masks_2, segment_ids_2,
            FLAGS.num_candidates,
            label_ids,
            use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          tf.logging.info("init from " + init_checkpoint)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("init from " + init_checkpoint)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "q_ids": q_ids,
          "cand_nums": cand_nums,
          "probabilities": logits,
          "label_ids": label_ids,
          "scores": prior_scores,
      }
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def model_fn_builder_single(bert_config, init_checkpoint, learning_rate,
                            num_train_steps, num_warmup_steps, use_tpu,
                            use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del params, labels
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    q_ids = features["q_ids"]
    cand_nums = features["cand_nums"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    prior_scores = features["scores"]
    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    (total_loss, logits) = create_ca_model(bert_config, is_training, input_ids,
                                           input_mask, segment_ids,
                                           FLAGS.num_candidates,
                                           FLAGS.max_seq_length, label_ids,
                                           use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "q_ids": q_ids,
          "cand_nums": cand_nums,
          "probabilities": logits,
          "label_ids": label_ids,
          "scores": prior_scores,
      }
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def run_dev_eval(estimator, evalset, output_dir):
  """Run evaluation on eval set."""
  factored_str = "True"
  if not FLAGS.factored_model:
    factored_str = "False"
  data_dir = os.path.join(FLAGS.data_dir, factored_str)
  prefix = evalset
  eval_files = get_file_names(
      os.path.join(data_dir, prefix), FLAGS.file_pattern)

  tf.logging.info("Evaluating on " + " ".join(eval_files))
  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Num examples = %d", _num_records(eval_files))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  # This tells the estimator to run through the entire set.
  eval_cnt = _num_records(eval_files)
  print(os.path.join(data_dir, prefix))
  tf.logging.info("Number of records in evaluation:" + str(eval_cnt))

  eval_drop_remainder = True if FLAGS.use_tpu else False
  eval_input_fn = file_based_input_fn_builder_files(
      input_files=eval_files,
      seq_length=FLAGS.max_seq_length,
      seq_length_query=FLAGS.max_seq_length_query,
      num_candidates=FLAGS.num_candidates,
      is_training=False,
      drop_remainder=eval_drop_remainder)
  if FLAGS.mode == "eval":
    # When FLAGS.mode == "eval", will run evaluation on FLAGS.init_checkpoint
    global_step = get_ckpt_number(FLAGS.init_checkpoint)
    FLAGS.global_step = global_step
    print("Running model eval on ckpt " + FLAGS.init_checkpoint)
    evaluate(FLAGS.init_checkpoint, eval_input_fn, estimator, evalset)

  elif FLAGS.mode == "continuous_eval":
    # writer = tf.summary.FileWriter(output_dir, flush_secs=20)
    # get a list of all checkpoints and eval on them if they still exist
    # then wait for new chekcpoints until the end of training and evaluate
    checkpoint_path = output_dir
    print(checkpoint_path)
    checkpoint_list = []
    try:
      checkpoint_list.extend(
          tf.train.get_checkpoint_state(
              checkpoint_path).all_model_checkpoint_paths)
    except tf.errors.NotFoundError:
      tf.logging.info("WARNING. Did not find checkpoints.")
    tf.logging.info("checkpoint list:" + "_".join(checkpoint_list))

    for ckpt in tf.train.checkpoints_iterator(output_dir, timeout=200 * 60):
      try:
        global_step = get_ckpt_number(ckpt)
        FLAGS.global_step = global_step
        evaluate(ckpt, eval_input_fn, estimator, evalset)
      except tf.errors.NotFoundError:
        tf.logging.error("Checkpoint path '%s' no longer exists.", ckpt)

  else:
    checkpoint_list = FLAGS.ckpt_pattern.split(",")
    tf.logging.info("Found " + FLAGS.ckpt_pattern)
    for ckpt in checkpoint_list:
      try:
        tf.logging.info("Found " + ckpt)
        global_step = get_ckpt_number(ckpt)
        FLAGS.global_step = global_step
        evaluate(ckpt, eval_input_fn, estimator, evalset)
      except tf.errors.NotFoundError:
        tf.logging.info("WARNING. Lost checkpoint. ")


def get_ckpt_number(ckpt):
  pattern = re.compile("model.ckpt-[0-9]+")
  pattern_match = pattern.search(ckpt)
  if pattern_match is None:
    return "best_checkpoint"
  else:
    return int(pattern_match.group().replace("model.ckpt-", ""))


def evaluate(checkpoint, eval_input_fn, estimator, eval_name):
  """Run evaluation."""
  try:
    eval_results = get_eval_dictionary(
        estimator, checkpoint, eval_input_fn, eval_name=eval_name)
    if FLAGS.mode == "continuous_eval":
      # need to write summaries manually
      log_dir = os.path.join(FLAGS.output_dir, "eval_" + eval_name)
      writer = tf.summary.FileWriter(log_dir, flush_secs=20)
      global_step = get_ckpt_number(checkpoint)
      if global_step != "best_checkpoint":
        for k, v in eval_results.items():
          summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
          writer.add_summary(summary, global_step)
      writer.close()
    if eval_name != "train":
      checkpoint_utils.save_checkpoint_if_best(
          eval_results["mrr_10"], checkpoint,
          os.path.join(estimator.model_dir, eval_name + "best_checkpoint_mrr"))
  except tf.errors.NotFoundError:
    tf.logging.error("Checkpoint path '%s' no longer exists.", checkpoint)


def get_eval_dictionary(estimator, checkpoint, eval_input_fn, eval_name=None):
  """run inference and get eval results."""
  eval_results = dict()
  all_results = []
  n_proc = 0
  results = estimator.predict(
      input_fn=eval_input_fn, checkpoint_path=checkpoint)
  for result in results:
    q_ids = result["q_ids"]
    cand_nums = result["cand_nums"]
    probs = result["probabilities"]
    label_ids = result["label_ids"]
    scores = result["scores"]
    cand_index = 0
    n_proc = n_proc + 1
    while cand_index < len(q_ids):
      q_id = q_ids[cand_index]
      cand_num = cand_nums[cand_index]
      prob = probs[cand_index]
      label_id = label_ids[cand_index]
      score = scores[cand_index]
      cand_index = cand_index + 1
      all_results.append(
          RawResult(
              q_id=q_id,
              cand_num=cand_num,
              prob=prob,
              label_id=label_id,
              prior_score=score))
  eval_results = rank_metrics(
      all_results,
      FLAGS.output_dir,
      FLAGS.global_step,
      eval_name,
      save_raw=True)
  tf.logging.info("Eval results: %s", eval_results)

  return eval_results


def get_output_dir(bert_config, output_base_dir):
  """Appends hyperparameter info to output_base_dir."""

  layer_norm = "L2"

  if FLAGS.layer_norm:
    layer_norm = "layernorm"

  if not FLAGS.factored_model:
    items = [
        "msl%d" % FLAGS.max_seq_length,
        "nl%d" % bert_config.num_hidden_layers,
        "ah%d" % bert_config.num_attention_heads,
        "hs%d" % bert_config.hidden_size,
        "lr%.0e" % FLAGS.learning_rate,
        "warmup%.2f" % FLAGS.warmup_proportion,
        "bs%d" % FLAGS.train_batch_size,
        "ne%d" % FLAGS.num_train_epochs,
        "dropout%.1f" % FLAGS.dropout,
        "CrossAttention",
    ]
  else:
    items = [
        "msl%d" % FLAGS.max_seq_length,
        "nl%d" % bert_config.num_hidden_layers,
        "ah%d" % bert_config.num_attention_heads,
        "hs%d" % bert_config.hidden_size,
        "lr%.0e" % FLAGS.learning_rate,
        "warmup%.2f" % FLAGS.warmup_proportion,
        "bs%d" % FLAGS.train_batch_size,
        "ne%d" % FLAGS.num_train_epochs,
        "dropout%.1f" % FLAGS.dropout,
        "proj%d" % FLAGS.projection_size,
        layer_norm,
        "q%d" % FLAGS.num_vec_query,
        "d%d" % FLAGS.num_vec_passage,
    ]

  model_name = "_".join([item for item in items if item])
  output_dir = os.path.join(output_base_dir, model_name)
  print("Write models to " + output_dir)
  return output_dir


def main(_):

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  factored_str = "True"
  if not FLAGS.factored_model:
    factored_str = "False"
  data_dir = os.path.join(FLAGS.data_dir, factored_str)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if not FLAGS.output_base_dir:
    raise ValueError("--output_base_dir must be specified.")
  output_dir = get_output_dir(bert_config, FLAGS.output_base_dir)
  print("Output dir %s", output_dir)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  FLAGS.output_dir = output_dir

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host,
          tpu_job_name=FLAGS.tpu_job_name))
  num_train_steps = 0
  num_warmup_steps = 0
  prefix = ""
  if FLAGS.mode == "eval":
    train_files = []
  else:
    train_files = get_file_names(
        os.path.join(data_dir, prefix + "train"), FLAGS.file_pattern)
  tf.logging.info("Loading training data from: " + " ".join(train_files))
  print("Looked for files in " + data_dir + " and found " +
        " ".join(train_files))
  train_examples_cnt = _num_records(train_files)
  tf.logging.info("Number of training records: " + str(train_examples_cnt))
  num_train_steps = int(train_examples_cnt / FLAGS.train_batch_size *
                        FLAGS.num_train_epochs)

  if FLAGS.do_train:
    print("Training is on")
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf_estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    print("Training is on")
    prefix = ""
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", train_examples_cnt)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder_files(
        input_files=train_files,
        seq_length=FLAGS.max_seq_length,
        seq_length_query=FLAGS.max_seq_length_query,
        num_candidates=FLAGS.num_candidates,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  if FLAGS.do_eval:
    print("Evaluation is on")
    run_dev_eval(estimator, FLAGS.eval_name, FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)

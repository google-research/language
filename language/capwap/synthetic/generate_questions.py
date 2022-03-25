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
"""Question generation training and (greedy) inference.

This program expects precomputed tf.Examples for training and prediction.
Input tf.Examples are expected to contain:
  `inputs_ids`: `seq_length` word piece ids, e.g.
                [CLS] [Q] q1 q2 q3 ... [SEQ] c1 c2 c3 ... [SEP]
  `segment_ids`: `seq_length` token type ids,
                  0 -> question, 1 -> context, 2 -> answer.
  `input_mask`: `seq_length` 0/1 integers indicating real tokens vs. padding.

This code is borrowed from the paper:
Chris Alberti, Daniel Andor, Emily Pitler, Jacob Devlin, and Michael Collins.
2019. Synthetic QA Corpora Generation with Roundtrip Consistency. In ACL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from bert import modeling
from bert import optimization
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

flags = tf.flags

FLAGS = flags.FLAGS

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

## Required parameters
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "bert_config_file", os.path.join(DATA_DIR, "qgen_model/bert_config.json"),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file",
                    os.path.join(DATA_DIR, "qgen_model/uncased_vocab.txt"),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint",
    os.path.join(DATA_DIR, "qgen_model/question_generation/model.ckpt"),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "train_precomputed_file", None,
    "Precomputed tf records for training. This can be provided in place of "
    "`train_file`. If this is provided, then --train_num_precomputed should "
    "be set to approximately the number of precomputed training tf examples.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string(
    "predict_precomputed_file", None,
    "Precomputed tf records for preditions. This can be provided in place of "
    "`predict_file`, but bookkeeping information necessary for the "
    "evaluation is missing from precomputed features so raw logits will "
    "be written to .npy file instead.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 20,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", True,
                  "Whether to compute predictions on the predict file.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("random_seed", 0, "Seed to use for shuffling the data.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("tpu_job_name", "tpu_worker", "Name of tpu worker.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("session_master", "local",
                       "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", None,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# **MONKEY PATCH** to directly provide BERT with the attention mask.
modeling.create_attention_mask_from_input_mask = (
    lambda from_tensor, to_mask: to_mask)


def make_attention_mask(batch_size, query_length, seq_length):
  """Returns an attention mask for question generation.

  Args:
    batch_size: number of windows per batch.
    query_length: length of question including [CLS], [Q] and [SEP].
    seq_length: total number of word pieces in window.

  Returns:
    input_mask: mask used to zero the question token ids at inference time.
    attention_mask: mask used at training and inference time to make the
      representations of question tokens causal.

  The attention mask should be [batch_size, from_seq_length, to_seq_length].
  Every training example will have the same attention mask and it's:
               to:
              [CLS]   [Q]   q1    q2    ...  [SEP]  c1    c2    c3   ...
        [CLS]    1     1     0     0     0     1     1     1     1    1
  from:  [Q]     1     1     0     0     0     1     1     1     1    1
         q1      1     1     1     0     0     1     1     1     1    1
         q2      1     1     1     1     0     1     1     1     1    1
         ...     1     1     1     1     1     1     1     1     1    1
        [SEP]    1     1     0     0     0     1     1     1     1    1
         c1      1     1     0     0     0     1     1     1     1    1
         c2      1     1     0     0     0     1     1     1     1    1
         c3      1     1     0     0     0     1     1     1     1    1
         ...     1     1     0     0     0     1     1     1     1    1
  """
  # TODO(chrisalberti): add test for this.
  input_mask = np.reshape(
      np.array(
          [1, 1] + [0] * (query_length - 2) + [1] * (seq_length - query_length),
          dtype=np.int64), [1, 1, seq_length])
  attention_mask = np.reshape(input_mask, [1, 1, seq_length]) * np.ones(
      [1, seq_length, 1])  # Uses broadcasting.
  for f in range(1, query_length):
    attention_mask[0, f, 1:f + 1] = 1
  attention_mask = np.ones([batch_size, 1, 1]) * attention_mask  # Broadcasting.
  return input_mask, attention_mask


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings, scope):
  """Creates a QGen model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope)

  tvars = tf.trainable_variables()
  word_embeddings = [
      v for v in tvars if v.name == "bert/embeddings/word_embeddings:0"
  ][0]
  final_hidden = model.get_sequence_output()
  query_length = FLAGS.max_query_length

  # Get the logits for the word predictions.
  word_logits = tf.einsum(
      "aij,kj->aik",
      final_hidden[:, 1:query_length - 1, :],
      word_embeddings,
      name="word_logits")
  return word_logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    seq_length = modeling.get_shape_list(input_ids)[1]
    query_length = FLAGS.max_query_length
    batch_size = params["batch_size"]

    _, attention_mask = make_attention_mask(batch_size, query_length,
                                            seq_length)

    with tf.variable_scope("bert") as scope:
      word_logits = create_model(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=attention_mask,
          segment_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings,
          scope=scope)

    if not is_training:
      with tf.variable_scope("bert", reuse=True) as scope:
        output_ids = input_ids
        word_id = tf.argmax(word_logits, axis=2, output_type=tf.int32)

        # This operation implements: output_ids[:, 2] = word_id[:, 0]
        word_id = tf.pad(word_id, [[0, 0], [2, seq_length - query_length]])
        output_ids = input_ids + word_id * tf.one_hot(
            2, seq_length, dtype=tf.int32)

        def body(i, ids):
          """A decoding step."""
          word_logits = create_model(
              bert_config=bert_config,
              is_training=is_training,
              input_ids=ids,
              input_mask=attention_mask,
              segment_ids=segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=scope)

          word_id = tf.argmax(word_logits, axis=2, output_type=tf.int32)

          # This operation implements: output_ids[:, 1 + i] = word_id[:, i - 1]
          word_id = tf.pad(word_id, [[0, 0], [2, seq_length - query_length]])
          return [
              i + 1,
              ids + word_id * tf.one_hot(i + 1, seq_length, dtype=tf.int32)
          ]

        i0 = tf.constant(2)
        c = lambda i, _: i < query_length - 1
        _, output_ids = tf.while_loop(c, body, loop_vars=[i0, output_ids])

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
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
      # Computes the loss for word prediction.
      loss = tf.losses.sparse_softmax_cross_entropy(
          input_ids[:, 2:query_length],
          word_logits,
          reduction=tf.losses.Reduction.MEAN)

      train_op = optimization.create_optimizer(loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": tf.identity(unique_ids),
          "input_ids": output_ids,
          "segment_ids": tf.minimum(segment_ids, 1),
          "input_mask": tf.to_int32(tf.not_equal(output_ids, 0)),
          "start_positions": tf.identity(features["start_positions"]),
          "end_positions": tf.identity(features["end_positions"]),
          "answer_types": tf.identity(features["answer_types"])
      }
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  vocab = tf.gfile.GFile(FLAGS.vocab_file).read().split("\n")
  cls_id = vocab.index("[CLS]")
  q_id = vocab.index("[Q]")

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "start_positions": tf.FixedLenFeature([], tf.int64),
      "end_positions": tf.FixedLenFeature([], tf.int64),
      "answer_types": tf.FixedLenFeature([], tf.int64),
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

    if not is_training:
      # This zeroes out the question portion of the input and then adde back
      # the first two tokens.
      example["input_ids"] = example["input_ids"] * example["segment_ids"]
      example["input_ids"] = (
          example["input_ids"] +
          cls_id * tf.one_hot(0, seq_length, dtype=tf.int32) +
          q_id * tf.one_hot(1, seq_length, dtype=tf.int32))

    example["segment_ids"] = (
        example["segment_ids"] + tf.to_int32(
            tf.logical_and(
                tf.range(seq_length) >= example["start_positions"],
                tf.range(seq_length) <= example["end_positions"])))

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


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_precomputed_file:
      raise ValueError("If `do_train` is True, then "
                       "`train_precomputed_file` must be specified.")

  if FLAGS.train_precomputed_file:
    if not FLAGS.train_num_precomputed:
      raise ValueError("If `train_precomputed_file` is specified, then "
                       "`train_num_precomputed` must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_precomputed_file:
      raise ValueError("If `do_predict` is True, then "
                       "`predict_precomputed_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


RawResult = collections.namedtuple("RawResult", ["unique_id", "output_ids"])


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.session_master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_features = FLAGS.train_num_precomputed
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf_estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training on precomputed features *****")
    tf.logging.info("  Num split examples = %d", num_train_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_filename = FLAGS.train_precomputed_file
    train_input_fn = input_fn_builder(
        input_file=train_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    num_eval_features = len(
        list(tf.python_io.tf_record_iterator(FLAGS.predict_precomputed_file)))
    eval_filename = FLAGS.predict_precomputed_file

    tf.logging.info("***** Running predictions on precomputed features *****")
    tf.logging.info("  Num split examples = %d", num_eval_features)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    processed_examples = 0
    output_file = os.path.join(FLAGS.output_dir, "predicted-tfrecords")

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    with tf.python_io.TFRecordWriter(output_file) as writer:
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if processed_examples % 1000 == 0:
          tf.logging.info("Processing example: %d" % processed_examples)
        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([result["unique_ids"]])
        features["input_ids"] = create_int_feature(result["input_ids"])
        features["input_mask"] = create_int_feature(result["input_mask"])
        features["segment_ids"] = create_int_feature(result["segment_ids"])
        features["start_positions"] = create_int_feature(
            [result["start_positions"]])
        features["end_positions"] = create_int_feature(
            [result["end_positions"]])
        features["answer_types"] = create_int_feature([result["answer_types"]])
        writer.write(
            tf.train.Example(features=tf.train.Features(
                feature=features)).SerializeToString())
        processed_examples += 1


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  tf.disable_v2_behavior()
  tf.app.run()

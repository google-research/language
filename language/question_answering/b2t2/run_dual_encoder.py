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
"""Dual Encoder model for VCR.

This program expects precomputed tf.Examples for training and prediction.
Input tf.Examples are expected to contain:
  `inputs_ids`: `seq_length` word piece ids, e.g.
                [CLS] c1 c2 c3 c4 ... [SEP] m1 m2 m3 ... [SEP]
  `segment_ids`: `seq_length` token type ids, 0 -> caption, 1 -> memory.
  `input_mask`: `seq_length` 0/1 integers indicating real tokens vs. padding.
  `image_vector`: float vector with visual features, e.g. from resnet.
  `label`: 1 or 0, whether the image vector matches the caption.
  `img_id`: unique identifier for this image.
  `annot_id`: unique identifier for this annotation.
  `choice_id`: unique identifier for this choice of answer and rationale.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from bert import modeling
from bert import optimization
import tensorflow as tf
import tensorflow_hub as hub

flags = tf.flags

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
    "`predict_file`, but bookkeeping information necessary for the COQA "
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
    "image_vector_dim", 2048,
    "Dimensionality of the image vector.")

flags.DEFINE_bool(
    "ignore_image", False,
    "Whether to ignore image information when answering a question.")

flags.DEFINE_bool("trainable_resnet", False,
                  "Whether to finetune resnet during training or freeze it.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False,
                  "Whether to compute predictions on the predict file.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_string("output_pred_file", "predicted-tfrecords",
                    "File to write output predictions to in output dir.")

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

tf.flags.DEFINE_string("master", "local", "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 image_vector, use_one_hot_embeddings, scope):
  """Creates a model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope)

  if FLAGS.ignore_image:
    logit = tf.layers.dense(
        model.get_pooled_output(), 1, activation=tf.tanh,
        kernel_initializer=
        modeling.create_initializer(bert_config.initializer_range))
    logit = tf.squeeze(logit, axis=1)
  else:
    logit = tf.einsum("ij,ij->i", tf.layers.dense(
        image_vector,
        bert_config.hidden_size,
        activation=tf.tanh,
        kernel_initializer=
        modeling.create_initializer(bert_config.initializer_range)),
                      model.get_pooled_output(),
                      name="inner")

  return tf.stack([-logit, logit], axis=1)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    label = features["label"]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    module = hub.Module(
        "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3",
        trainable=FLAGS.trainable_resnet)
    batch_size = params["batch_size"]
    height, width = hub.get_expected_image_size(module)
    image_vector = module(
        tf.reshape(features["image"], [batch_size, height, width, 3]))

    with tf.variable_scope("bert") as scope:
      logits = create_model(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=features["input_ids"],
          input_mask=features["input_mask"],
          segment_ids=features["segment_ids"],
          image_vector=image_vector,
          use_one_hot_embeddings=use_one_hot_embeddings,
          scope=scope)

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
    if mode == tf.estimator.ModeKeys.TRAIN:
      loss = tf.losses.sparse_softmax_cross_entropy(
          label, logits, reduction=tf.losses.Reduction.MEAN)

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "img_id": features["img_id"],
          "annot_id": features["annot_id"],
          "choice_id": features["choice_id"],
          "label": label,
          "output_logits": logits
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      # VCR:
      "img_id": tf.FixedLenFeature([], tf.int64),
      "annot_id": tf.FixedLenFeature([], tf.int64),
      "choice_id": tf.FixedLenFeature([], tf.int64),

      # Common:
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "image": tf.FixedLenFeature([], tf.string),
      "label": tf.FixedLenFeature([], tf.int64),
  }

  height = 224
  width = 224

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # Convert image to float tensor.
    image = example["image"]
    image_decoded = tf.image.decode_jpeg(image, channels=3)
    image_decoded.set_shape([None, None, 3])
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_image_with_pad(image_float, height, width)
    example["image"] = tf.reshape(image_resized, [height, width, 3])

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
    d = tf.data.Dataset.list_files(input_file, shuffle=False)
    d = d.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=is_training))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_features = FLAGS.train_num_precomputed
    num_train_steps = int(
        num_train_features / FLAGS.train_batch_size * FLAGS.num_train_epochs)
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
  estimator = tf.contrib.tpu.TPUEstimator(
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
    tf.logging.info("***** Running predictions on precomputed features *****")
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    eval_filename = FLAGS.predict_precomputed_file
    predict_input_fn = input_fn_builder(
        input_file=eval_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    def create_int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    def create_float_feature(values):
      return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    processed_examples = 0
    output_file = os.path.join(FLAGS.output_dir, FLAGS.output_pred_file)
    tf.logging.info("Writing results to: %s", output_file)
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if processed_examples % 1000 == 0:
          tf.logging.info("Processing example: %d" % processed_examples)
        features = collections.OrderedDict()
        features["img_id"] = create_int_feature([result["img_id"]])
        features["annot_id"] = create_int_feature([result["annot_id"]])
        features["choice_id"] = create_int_feature([result["choice_id"]])
        features["label"] = create_int_feature([result["label"]])
        features["output_logits"] = create_float_feature(
            result["output_logits"])
        writer.write(tf.train.Example(
            features=tf.train.Features(feature=features)).SerializeToString())
        processed_examples += 1


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

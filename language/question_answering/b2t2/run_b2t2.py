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
"""Full B2T2 for VCR.

This program expects precomputed tf.Examples for training and prediction.
Input tf.Examples are expected to contain:
  `inputs_ids`: `seq_length` word piece ids, e.g.
                [CLS] c1 c2 c3 c4 ... [SEP] m1 m2 m3 ... [SEP]
  `segment_ids`: `seq_length` token type ids, 0 -> caption, 1 -> memory.
  `input_mask`: `seq_length` 0/1 integers indicating real tokens vs. padding.
  `image`: float vector with visual features, e.g. from resnet.
  `label`: 1 or 0, whether the image vector matches the caption.
  `img_id`: unique identifier for this image.
  `annot_id`: unique identifier for this annotation.
  `choice_id`: unique identifier for this choice of answer and rationale.

If --use_bboxes is true, then these fields are also expected:
  `bbox_pos`: a `FLAGS.max_num_bboxes` x 4 int matrix, containing
       the first `FLAGS.max_num_bboxes` bounding boxes referenced in the
       tokens. Each row contains offset_height, offset_width,
       target_height and target_width of the bounding box.
  `bbox_idx`: an int vector of length `seq_length`, containing
       the index of the bounding box relevant to each token, or -1 if no
       bounding box is referenced.

If --mask_lm_loss is true, then these fields are also expected:
  `masked_lm_positions`: positions at which to predict masked wordpieces.
  `masked_lm_ids`: which to wordpiece to predict.
  `masked_lm_weights`: 1.0 for real predictions, 0.0 for padding.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os

from bert import modeling
from bert import optimization
from bert import run_pretraining
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import tpu as contrib_tpu

flags = tf.flags

FLAGS = flags.FLAGS

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

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("negative_loss", True, "True to predict negative captions.")

flags.DEFINE_bool("mask_lm_loss", False, "Whether to use a mask LM loss.")

flags.DEFINE_string("output_pred_file", "predicted-tfrecords",
                    "File to write output predictions to in output dir.")

flags.DEFINE_integer("image_seq_length", 1, "Number image embedding vectors.")

flags.DEFINE_integer(
    "image_vector_dim", 2048,
    "Dimensionality of the image vector.")

flags.DEFINE_bool("use_bboxes", False,
                  "Whether to embed bounding boxes input them to the model.")

flags.DEFINE_integer("max_num_bboxes", 4,
                     "Maximum number of bounding boxes to consider.")

flags.DEFINE_integer("num_output_labels", 2, "Number of output labels.")

flags.DEFINE_bool("use_bbox_position", False,
                  "True to embed bounding box center position.")

flags.DEFINE_bool("trainable_resnet", False,
                  "Whether to finetune resnet during training or freeze it.")

flags.DEFINE_bool("has_choice_id", True,
                  "Whether the input tf.Examples contain `choice_id`.")

flags.DEFINE_bool("do_output_logits", True,
                  "Whether to output all logits or just the best label.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("random_seed", 0, "Seed to use for shuffling the data.")

# Images and patches are resized to these dimensions before feature extraction.
IMG_HEIGHT = 224
IMG_WIDTH = 224


class B2T2Model(modeling.BertModel):
  """A BERT model with visual tokens."""

  def __init__(self,
               config,
               is_training,
               input_ids,
               image_embeddings,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for a visually grounded BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      image_embeddings: float32 Tensor of shape [batch_size, seq_length, depth].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".
    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    text_input_shape = modeling.get_shape_list(input_ids, expected_rank=2)
    batch_size = text_input_shape[0]
    text_seq_length = text_input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, text_seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(
          shape=[batch_size, text_seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output,
         self.embedding_table) = modeling.embedding_lookup(
             input_ids=input_ids,
             vocab_size=config.vocab_size,
             embedding_size=config.hidden_size,
             initializer_range=config.initializer_range,
             word_embedding_name="word_embeddings",
             use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = modeling.embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        # Add image embeddings the rest of the input embeddings.
        self.embedding_output += tf.layers.dense(
            image_embeddings,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(
                config.initializer_range))

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = modeling.create_attention_mask_from_input_mask(
            self.embedding_output, input_mask)

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
            kernel_initializer=
            modeling.create_initializer(config.initializer_range))


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

    # First extract a global image vector.
    image_vector = module(
        tf.reshape(features["image"], [batch_size, height, width, 3]),
        signature="image_feature_vector",
        as_dict=True)["default"]
    image_vector = tf.reshape(image_vector,
                              [batch_size, 1, FLAGS.image_vector_dim])

    # The global image vector is added in the position of the [IMAGE] token,
    # which comes right after the [CLS] token.
    image_embeddings = tf.concat([
        tf.zeros([batch_size, 1, FLAGS.image_vector_dim]), image_vector,
        tf.zeros([batch_size, FLAGS.max_seq_length - 2, FLAGS.image_vector_dim])
    ],
                                 axis=1)

    if FLAGS.use_bboxes:
      # Then extract an image vector for each of the bounding boxes (at most
      # FLAGS.max_num_bboxes).
      boxes_vectors = module(
          tf.reshape(features["bboxes"],
                     [batch_size * FLAGS.max_num_bboxes, height, width, 3]),
          signature="image_feature_vector",
          as_dict=True)["default"]
      boxes_vectors = tf.reshape(
          boxes_vectors,
          [batch_size, FLAGS.max_num_bboxes, FLAGS.image_vector_dim])

      if FLAGS.use_bbox_position:
        tf.logging.info("Embedding bbox position with 56 positions.")
        # Position embedding for bbox location.
        def _make_position_embedding_table(scope):
          # 56 is 224 / 4
          return tf.get_variable(
              scope, [56, FLAGS.image_vector_dim // 4],
              initializer=tf.truncated_normal_initializer(stddev=0.02))

        position_x_embeds = _make_position_embedding_table(
            "position_x_embeddings")
        position_y_embeds = _make_position_embedding_table(
            "position_y_embeddings")

        # bbox_pos features are top left corner in image, and height and width.
        bbox_pos = features["bbox_pos"] // 4
        y1 = tf.one_hot(bbox_pos[:, :, 0], 56)
        x1 = tf.one_hot(bbox_pos[:, :, 1], 56)
        y2 = tf.one_hot(bbox_pos[:, :, 0] + bbox_pos[:, :, 2], 56)
        x2 = tf.one_hot(bbox_pos[:, :, 1] + bbox_pos[:, :, 3], 56)

        bbox_x_embeds = tf.einsum("bixc,cd->bixd", tf.stack([x1, x2], axis=2),
                                  position_x_embeds)
        bbox_y_embeds = tf.einsum("biyc,cd->biyd", tf.stack([y1, y2], axis=2),
                                  position_y_embeds)
        # [batch_size, max_num_bboxes, image_vector_size]
        bbox_pos_embeds = tf.concat([
            tf.reshape(bbox_x_embeds, [batch_size, FLAGS.max_num_bboxes, -1]),
            tf.reshape(bbox_y_embeds, [batch_size, FLAGS.max_num_bboxes, -1])
        ], -1)
        boxes_vectors += bbox_pos_embeds

      # Now place the image vectors of each bounding box in the position of each
      # special token that references that bounding box. The letters in the
      # einsum mean the following:
      #   b: batch element index
      #   t: token index
      #   i: bounding box index
      #   d: depth of image representation
      image_embeddings += tf.einsum(
          "bti,bid->btd", tf.one_hot(features["bbox_idx"],
                                     FLAGS.max_num_bboxes), boxes_vectors)

    with tf.variable_scope("bert") as scope:
      # This is just like a regular BERT model, but the given image embeddings
      # are added to the input word piece embeddings.
      model = B2T2Model(
          config=bert_config,
          is_training=is_training,
          input_ids=features["input_ids"],
          image_embeddings=image_embeddings,
          input_mask=features["input_mask"],
          token_type_ids=features["segment_ids"],
          use_one_hot_embeddings=use_one_hot_embeddings,
          scope=scope)
      output_weights = tf.get_variable(
          "output_weights", [FLAGS.num_output_labels, bert_config.hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable(
          "output_bias", [FLAGS.num_output_labels],
          initializer=tf.zeros_initializer())
      logits = tf.matmul(
          model.get_pooled_output(), output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

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
      loss = 0.0
      if FLAGS.negative_loss:
        tf.logging.info("Using negative loss.")
        loss += tf.losses.sparse_softmax_cross_entropy(
            label, logits, reduction=tf.losses.Reduction.MEAN)

      if FLAGS.mask_lm_loss:
        tf.logging.info("Using mask LM loss.")
        # Don't use mask LM for negative captions.
        masked_lm_label_weights = features["masked_lm_weights"]
        masked_lm_label_weights *= tf.expand_dims(
            tf.cast(label, tf.float32), -1)
        (masked_lm_loss, _, _) = run_pretraining.get_masked_lm_output(
            bert_config, model.get_sequence_output(),
            model.get_embedding_table(), features["masked_lm_positions"],
            features["masked_lm_ids"], masked_lm_label_weights)
        loss += masked_lm_loss

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          # Image, annotation and choice identifiers.
          "img_id": features["img_id"],
          "annot_id": features["annot_id"],

          # Gold label.
          "label": label,
      }

      if FLAGS.mask_lm_loss:
        # We don't care about masked_lm_weights for prediction.
        _, _, masked_lm_log_probs = run_pretraining.get_masked_lm_output(
            bert_config, model.get_sequence_output(),
            model.get_embedding_table(), features["masked_lm_positions"],
            features["masked_lm_ids"], features["masked_lm_weights"])
        # [batch_size * max_preds_per_seq]
        masked_lm_preds = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        batch_size, max_preds_per_seq = modeling.get_shape_list(
            features["masked_lm_positions"])
        masked_lm_preds = tf.reshape(
            masked_lm_preds, [batch_size, max_preds_per_seq])
        predictions.update({
            "input_ids": features["input_ids"],
            "masked_lm_ids": features["masked_lm_ids"],
            "masked_lm_preds": masked_lm_preds
        })

      if FLAGS.has_choice_id:
        predictions["choice_id"] = features["choice_id"]

      if FLAGS.do_output_logits:
        predictions["output_logits"] = logits
      else:
        predictions["output_label"] = tf.argmax(logits, axis=-1)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      # Image, annotation and choice ids.
      "img_id": tf.FixedLenFeature([], tf.int64),
      "annot_id": tf.FixedLenFeature([], tf.int64),

      # JPEG image.
      "image": tf.FixedLenFeature([], tf.string),

      # Text input.
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),

      # Label to predict for this example.
      "label": tf.FixedLenFeature([], tf.int64),
  }

  if FLAGS.has_choice_id:
    name_to_features["choice_id"] = tf.FixedLenFeature([], tf.int64)

  if FLAGS.use_bboxes:
    name_to_features.update({
        # Bounding boxes metadata.
        "bbox_pos": tf.FixedLenFeature([FLAGS.max_num_bboxes, 4], tf.int64),
        "bbox_idx": tf.FixedLenFeature([seq_length], tf.int64),
    })

  if FLAGS.mask_lm_loss:
    name_to_features.update({
        # Masked LM ids.
        "masked_lm_positions":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
    })

  def parse_bounding_box(height, width, image_float, box):
    """Decodes an image, resizes and scales RGB values to be in [0, 1]."""
    image_bbox = tf.image.crop_to_bounding_box(
        image_float,
        offset_height=box[0],
        offset_width=box[1],
        target_height=box[2],
        target_width=box[3])
    image_resized = tf.image.resize_image_with_pad(image_bbox, height, width)
    image = tf.reshape(image_resized, [height, width, 3])
    return image

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # Convert image to float tensor.
    image = example["image"]
    image_decoded = tf.image.decode_jpeg(image, channels=3)
    image_decoded.set_shape([None, None, 3])
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_image_with_pad(
        image_float, IMG_HEIGHT, IMG_WIDTH)
    example["image"] = tf.reshape(image_resized, [IMG_HEIGHT, IMG_WIDTH, 3])

    # Get bboxes.
    if FLAGS.use_bboxes:
      example["bbox_pos"] = tf.to_int32(example["bbox_pos"])
      bboxes = []
      for idx in range(FLAGS.max_num_bboxes):
        bboxes.append(
            parse_bounding_box(IMG_HEIGHT, IMG_WIDTH, image_float,
                               example["bbox_pos"][idx, :]))
      example["bboxes"] = tf.stack(bboxes)

      if FLAGS.use_bbox_position:
        # Resized bboxes.
        y, x, bbox_height, bbox_width = tf.unstack(example["bbox_pos"], axis=1)
        orig_height, orig_width = modeling.get_shape_list(image_float)[:2]
        example["bbox_pos"] = tf.cast(
            tf.stack([
                IMG_HEIGHT * y / orig_height,
                IMG_WIDTH * x / orig_width,
                IMG_HEIGHT * bbox_height / orig_height,
                IMG_WIDTH * bbox_width / orig_width
            ], 1),
            dtype=tf.int32)

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
    d = tf.data.Dataset.list_files(input_file, shuffle=True)
    d = d.apply(
        contrib_data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=is_training))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    d = d.prefetch(1)

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
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=contrib_tpu.TPUConfig(
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
  estimator = contrib_tpu.TPUEstimator(
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

        if FLAGS.has_choice_id:
          features["choice_id"] = create_int_feature([result["choice_id"]])

        features["label"] = create_int_feature([result["label"]])

        if FLAGS.do_output_logits:
          features["output_logits"] = create_float_feature(
              result["output_logits"])
        else:
          features["output_label"] = create_int_feature(
              [result["output_label"]])

        writer.write(tf.train.Example(
            features=tf.train.Features(feature=features)).SerializeToString())
        processed_examples += 1


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

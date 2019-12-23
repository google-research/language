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
"""BERT next sentence prediction / binary coherence finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from absl import app
from absl import flags
from bert import modeling
from bert import optimization
from bert import tokenization
from language.conpono.cpc import bilin_model_builder
from language.conpono.reconstruct import preprocess as ip
import tensorflow as tf


from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import lookup as contrib_lookup
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "eval_file", None,
    "The input data. Should be in tfrecord format ready to input to BERT.")

flags.DEFINE_string(
    "train_file", None,
    "The input data. Should be in tfrecord format ready to input to BERT.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_bool("include_mlm", False, "Whether to include MLM loss/objective")

flags.DEFINE_integer("num_choices", 32, "Number of negative samples + 1")

flags.DEFINE_integer("data_window_size", 5, "Number of documents to draw"
                     "negative samples from.")

flags.DEFINE_integer("data_window_shift", 2, "Shift windows by this many for"
                     "negative samples.")

flags.DEFINE_integer("max_sent_length", 70, "Number of tokens per sentence.")

flags.DEFINE_integer("max_para_length", 30, "Number of sentences per paragraph")

flags.DEFINE_integer("context_size", 4, "Number of sentences in the context")

flags.DEFINE_integer("margin", 1, "Eta value for margin.")

flags.DEFINE_float("mask_rate", 0.1, "Rate of masking for mlm.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("k_size", 4, "Size of k.")

flags.DEFINE_float(
    "dataset_one_weight", 0.5, "Weight of first dataset."
    "Weight of second dataset will be 1-x")

flags.DEFINE_float(
    "dataset_two_weight", 0, "Weight of second dataset."
    "Weight of second dataset will be 1-x")

flags.DEFINE_bool("include_context", False, "Whether to include context.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("train_data_size", 10000, "The number of examples in the"
                     "training data")

flags.DEFINE_integer("eval_data_size", -1, "The number of examples in the"
                     "validation data")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
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

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

_SEP_TOKEN = "[SEP]"


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


# pylint: disable=invalid-name
Outputs_And_Context = collections.namedtuple(
    "Outputs_And_Context",
    ["input_ids", "input_mask", "segment_ids", "label_types", "context"])
# pylint: enable=invalid-name


def pad_and_cut(tensor, max_len_scalar):
  end_padding = tf.constant([[0, max_len_scalar]])
  return tf.pad(tensor, end_padding)[:max_len_scalar]


def build_distractors(distractor_examples, context):
  """Create inputs with distractors."""

  CLS_ID = tf.constant([101], dtype=tf.int64)  # pylint: disable=invalid-name
  SEP_ID = tf.constant([102], dtype=tf.int64)  # pylint: disable=invalid-name

  bert_inputs = []
  input_masks = []
  segment_ids = []
  # for each distractor
  sample_size = int(
      (FLAGS.num_choices - FLAGS.k_size) / (FLAGS.data_window_size - 1))
  for example in distractor_examples:
    # randomly sample 7
    intermediate_examples_tensor = tf.reduce_sum(tf.abs(example), 1)
    examples_zero_vector = tf.zeros(shape=(1, 1), dtype=tf.int64)
    examples_bool_mask = tf.squeeze(
        tf.not_equal(intermediate_examples_tensor, examples_zero_vector))
    paragraph_len = tf.reduce_sum(tf.cast(examples_bool_mask, tf.int32))
    indices = tf.range(0, limit=paragraph_len, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)[:sample_size]

    # extend examples / targets
    distractor_cand = example
    distractor_cand_plus_one = distractor_cand[1:]
    distractor_cand_plus_two = distractor_cand[2:]

    # pad extensions
    paddings_one = tf.constant([[0, 1], [0, 0]])
    distractor_cand_plus_one = tf.pad(distractor_cand_plus_one, paddings_one)

    paddings_two = tf.constant([[0, 2], [0, 0]])
    distractor_cand_plus_two = tf.pad(distractor_cand_plus_two, paddings_two)

    distractor_cand_ext = tf.concat(
        [distractor_cand, distractor_cand_plus_one, distractor_cand_plus_two],
        axis=1)

    distractors = tf.gather(distractor_cand_ext, shuffled_indices)
    for i in range(sample_size):
      distractors_non_zero = tf.where(
          tf.not_equal(distractors[i], tf.zeros_like(distractors[i])))
      distractors_stripped = tf.gather_nd(distractors[i], distractors_non_zero)
      if FLAGS.include_context:
        segment_id = tf.concat([
            tf.zeros_like(CLS_ID, dtype=tf.int64),
            tf.zeros_like(context),
            tf.zeros_like(SEP_ID, dtype=tf.int64),
            tf.ones_like(distractors_stripped),
            tf.ones_like(SEP_ID, dtype=tf.int64)
        ],
                               axis=0)
      else:
        segment_id = tf.concat([
            tf.zeros_like(CLS_ID, dtype=tf.int64),
            tf.zeros_like(distractors_stripped),
            tf.zeros_like(SEP_ID, dtype=tf.int64)
        ],
                               axis=0)
      segment_id = pad_and_cut(segment_id, FLAGS.max_seq_length)
      segment_ids.append(segment_id)
      if FLAGS.include_context:
        new_input = tf.concat(
            [CLS_ID, context, SEP_ID, distractors_stripped, SEP_ID], axis=0)
      else:
        new_input = tf.concat([CLS_ID, distractors_stripped, SEP_ID], axis=0)

      input_mask = tf.ones_like(new_input)
      input_mask = pad_and_cut(input_mask, FLAGS.max_seq_length)
      input_masks.append(input_mask)
      padded_new_input = pad_and_cut(new_input, FLAGS.max_seq_length)
      bert_inputs.append(padded_new_input)

  bert_inputs = tf.stack(bert_inputs, axis=0)
  input_masks = tf.stack(input_masks, axis=0)
  segment_ids = tf.stack(segment_ids, axis=0)
  out = Outputs_And_Context(bert_inputs, input_masks, segment_ids, None, None)
  return out


def build_bert_inputs(example):
  """Convert example <Tensor [30, 70]> into bert inputs."""
  k_size = FLAGS.k_size

  CLS_ID = tf.constant([101], dtype=tf.int64)  # pylint: disable=invalid-name
  SEP_ID = tf.constant([102], dtype=tf.int64)  # pylint: disable=invalid-name
  max_len = tf.constant([FLAGS.max_para_length])
  context_size = tf.constant([FLAGS.context_size])

  intermediate_examples_tensor = tf.reduce_sum(tf.abs(example), 1)
  examples_zero_vector = tf.zeros(shape=(1, 1), dtype=tf.int64)
  examples_bool_mask = tf.squeeze(
      tf.not_equal(intermediate_examples_tensor, examples_zero_vector))
  paragraph_len = tf.reduce_sum(tf.cast(examples_bool_mask, tf.int32))

  start = tf.random.uniform([1],
                            0,
                            tf.reshape(paragraph_len, []) -
                            tf.reshape(context_size, []) + 1,
                            dtype=tf.int32)

  # Slice the document into the before, after and context.
  # Discard the zero padding.
  sizes = tf.squeeze(
      tf.concat([[
          start, context_size, paragraph_len - context_size - start,
          max_len - paragraph_len
      ]], 0))
  before, context, after, _ = tf.split(example, sizes, axis=0)

  # Gather the context removing zero padding at end of sentences.
  non_zeros = tf.where(tf.not_equal(context, tf.zeros_like(context)))
  context_gathered = tf.gather_nd(context, non_zeros)

  # Flip before so we select the 4 sentences closest to target
  before = tf.reverse(before, axis=[0])

  # pad both to longer than needed
  paddings = tf.constant([[0, 8], [0, 0]])
  before = tf.pad(before, paddings)
  after = tf.pad(after, paddings)

  # Extend targets to 3 sentences
  # pad both
  before_minus_one = before[1:][:k_size]
  before_minus_two = before[2:][:k_size]
  after_plus_one = after[1:][:k_size]
  after_plus_two = after[2:][:k_size]
  before = before[:k_size]
  after = after[:k_size]

  before = tf.concat([before_minus_two, before_minus_one, before], axis=1)
  after = tf.concat([after, after_plus_one, after_plus_two], axis=1)
  ############################################################################

  # These 8 sentences are the 8 surrounding targets. Some are padding.
  targets = tf.concat([before, after], axis=0)

  # Remove the padding from the sourrounding sentences
  # Eg. if context starts at beginning of paragraph, before is all padding
  intermediate_tensor = tf.reduce_sum(tf.abs(targets), 1)
  zero_vector = tf.zeros(shape=(1, 1), dtype=tf.int64)
  bool_mask = tf.squeeze(tf.not_equal(intermediate_tensor, zero_vector))
  bool_mask.set_shape([None])
  targets = tf.boolean_mask(targets, bool_mask)

  # Randomly select 4 targets
  # We will also select the label_types for each selected target
  indices = tf.range(0, limit=tf.shape(targets)[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)[:k_size]

  targets = tf.gather(targets, shuffled_indices)
  if k_size == 4:
    full_labels = tf.concat([tf.range(3, -1, -1), tf.range(4, 8)], axis=0)
  elif k_size == 3:
    full_labels = tf.concat([tf.range(2, -1, -1), tf.range(3, 6)], axis=0)
  elif k_size == 2:
    full_labels = tf.concat([tf.range(1, -1, -1), tf.range(2, 4)], axis=0)
  elif k_size == 1:
    full_labels = tf.concat([tf.range(0, -1, -1), tf.range(1, 2)], axis=0)
  label_types = tf.boolean_mask(full_labels, bool_mask)
  label_types = tf.gather(label_types, shuffled_indices)

  # create inputs
  bert_inputs = []
  input_masks = []
  segment_ids = []

  # make context
  ctx_segment_id = tf.concat([
      tf.zeros_like(CLS_ID, dtype=tf.int64),
      tf.zeros_like(context_gathered),
      tf.zeros_like(SEP_ID, dtype=tf.int64)
  ],
                             axis=0)
  ctx_segment_id = pad_and_cut(ctx_segment_id, FLAGS.max_seq_length)
  segment_ids.append(ctx_segment_id)

  new_ctx_input = tf.concat([CLS_ID, context_gathered, SEP_ID], axis=0)
  ctx_input_mask = tf.ones_like(new_ctx_input)
  ctx_input_mask = pad_and_cut(ctx_input_mask, FLAGS.max_seq_length)
  input_masks.append(ctx_input_mask)
  padded_new_ctx_input = pad_and_cut(new_ctx_input, FLAGS.max_seq_length)
  bert_inputs.append(padded_new_ctx_input)

  for i in range(k_size):
    target_non_zero = tf.where(
        tf.not_equal(targets[i], tf.zeros_like(targets[i])))
    targets_stripped = tf.gather_nd(targets[i], target_non_zero)
    if FLAGS.include_context:
      segment_id = tf.concat([
          tf.zeros_like(CLS_ID, dtype=tf.int64),
          tf.zeros_like(context_gathered),
          tf.zeros_like(SEP_ID, dtype=tf.int64),
          tf.ones_like(targets_stripped),
          tf.ones_like(SEP_ID, dtype=tf.int64)
      ],
                             axis=0)
    else:
      segment_id = tf.concat([
          tf.zeros_like(CLS_ID, dtype=tf.int64),
          tf.zeros_like(targets_stripped),
          tf.zeros_like(SEP_ID, dtype=tf.int64)
      ],
                             axis=0)
    segment_id = pad_and_cut(segment_id, FLAGS.max_seq_length)
    segment_ids.append(segment_id)
    if FLAGS.include_context:
      new_input = tf.concat(
          [CLS_ID, context_gathered, SEP_ID, targets_stripped, SEP_ID], axis=0)
    else:
      new_input = tf.concat([CLS_ID, targets_stripped, SEP_ID], axis=0)
    input_mask = tf.ones_like(new_input)
    input_mask = pad_and_cut(input_mask, FLAGS.max_seq_length)
    input_masks.append(input_mask)
    padded_new_input = pad_and_cut(new_input, FLAGS.max_seq_length)
    bert_inputs.append(padded_new_input)
  bert_inputs = tf.stack(bert_inputs, axis=0)
  input_masks = tf.stack(input_masks, axis=0)
  segment_ids = tf.stack(segment_ids, axis=0)

  out = Outputs_And_Context(bert_inputs, input_masks, segment_ids, label_types,
                            context_gathered)

  return out


def file_based_input_fn_builder(input_file, is_training, drop_remainder,
                                add_masking):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  input_file = input_file.split(",")

  expanded_files = []
  for infile in input_file:
    try:
      sharded_files = tf.io.gfile.glob(infile)
      expanded_files.append(sharded_files)
    except tf.errors.OpError:
      expanded_files.append(infile)

  name_to_features = {
      "sents":
          tf.FixedLenFeature([FLAGS.max_para_length * FLAGS.max_sent_length],
                             tf.int64)
  }

  def _decode_record(record, name_to_features, vocab_table):
    """Decodes a record to a TensorFlow example."""
    target_example = tf.parse_single_example(record[0], name_to_features)
    target_example = tf.reshape(target_example["sents"],
                                [FLAGS.max_para_length, FLAGS.max_sent_length])

    # This is an unfortunate hack but is necessary to get around a TF error.
    dist0 = tf.reshape(
        tf.parse_single_example(record[1], name_to_features)["sents"],
        [FLAGS.max_para_length, FLAGS.max_sent_length])
    dist1 = tf.reshape(
        tf.parse_single_example(record[2], name_to_features)["sents"],
        [FLAGS.max_para_length, FLAGS.max_sent_length])
    dist2 = tf.reshape(
        tf.parse_single_example(record[3], name_to_features)["sents"],
        [FLAGS.max_para_length, FLAGS.max_sent_length])
    dist3 = tf.reshape(
        tf.parse_single_example(record[4], name_to_features)["sents"],
        [FLAGS.max_para_length, FLAGS.max_sent_length])

    inputs_obj = build_bert_inputs(target_example)

    distractor_obj = build_distractors([dist0, dist1, dist2, dist3],
                                       inputs_obj.context)

    example = {}
    example["input_ids"] = tf.concat(
        [inputs_obj.input_ids, distractor_obj.input_ids], axis=0)
    example["input_mask"] = tf.concat(
        [inputs_obj.input_mask, distractor_obj.input_mask], axis=0)
    example["segment_ids"] = tf.concat(
        [inputs_obj.segment_ids, distractor_obj.segment_ids], axis=0)
    example["label_types"] = inputs_obj.label_types

    # Add masking:
    if add_masking:
      mask_rate = FLAGS.mask_rate
      max_predictions_per_seq = int(math.ceil(FLAGS.max_seq_length * mask_rate))
      cls_token = "[CLS]"
      sep_token = "[SEP]"
      mask_token = "[MASK]"
      # pad_token = "[PAD]"
      mask_blacklist = tf.constant([cls_token, sep_token])  # , pad_token])
      mask_blacklist_ids = tf.to_int32(vocab_table.lookup(mask_blacklist))
      mask_token_id = tf.to_int32(vocab_table.lookup(tf.constant(mask_token)))
      input_ids = tf.to_int32(example["input_ids"])

      def call_sample_mask_indices(x):
        return ip.sample_mask_indices(x, mask_rate, mask_blacklist_ids,
                                      max_predictions_per_seq)

      mask_indices = tf.map_fn(
          call_sample_mask_indices, input_ids, dtype=tf.int32)

      def call_get_target_tokens(x):
        input_len = tf.shape(input_ids)[-1]
        x_input_id = x[:input_len]
        x_mask_indices = x[input_len:]
        return ip.get_target_tokens_for_apply(x_input_id, x_mask_indices)

      map_input = tf.concat([input_ids, mask_indices], -1)
      target_token_ids = tf.map_fn(call_get_target_tokens, map_input)

      def call_apply_masking(x):
        input_len = tf.shape(input_ids)[-1]
        mask_idx_len = tf.shape(mask_indices)[-1]
        x_input_id = x[:input_len]
        x_mask_indices = x[input_len:input_len + mask_idx_len]
        x_target_token_ids = x[input_len + mask_idx_len:]
        return ip.apply_masking(x_input_id, x_target_token_ids, x_mask_indices,
                                mask_token_id, 1000)

      map_input2 = tf.concat([input_ids, mask_indices, target_token_ids], -1)
      token_ids_masked = tf.map_fn(call_apply_masking, tf.to_int64(map_input2))
      target_token_weights = tf.ones_like(target_token_ids, dtype=tf.float32)
      pad_targets = tf.where(
          tf.equal(target_token_ids, 0),
          tf.ones_like(target_token_ids, dtype=tf.float32),
          tf.zeros_like(target_token_ids, dtype=tf.float32))
      target_token_weights = target_token_weights - pad_targets
      example["target_token_weights"] = target_token_weights
      example["target_token_ids"] = target_token_ids
      example["input_ids"] = token_ids_masked
      example["mask_indices"] = mask_indices

      # Set shape explicitly for TPU
      example["target_token_weights"].set_shape(
          [FLAGS.num_choices + 1, max_predictions_per_seq])
      example["target_token_ids"].set_shape(
          [FLAGS.num_choices + 1, max_predictions_per_seq])
      example["mask_indices"].set_shape(
          [FLAGS.num_choices + 1, max_predictions_per_seq])

    # Set shape explicitly for TPU
    k_size = FLAGS.k_size
    example["input_ids"].set_shape(
        [FLAGS.num_choices + 1, FLAGS.max_seq_length])
    example["input_mask"].set_shape(
        [FLAGS.num_choices + 1, FLAGS.max_seq_length])
    example["segment_ids"].set_shape(
        [FLAGS.num_choices + 1, FLAGS.max_seq_length])
    example["label_types"].set_shape([k_size])

    example["label_ids"] = tf.scatter_nd(
        tf.reshape(example["label_types"], [k_size, 1]), tf.range(k_size),
        [k_size * 2])

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):  # pylint: disable=g-builtin-op
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    vocab_table = contrib_lookup.index_table_from_file(FLAGS.vocab_file)

    if len(expanded_files) == 1:
      d = tf.data.TFRecordDataset(expanded_files[0])
      if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=256)
    else:
      dataset_list = [
          tf.data.TFRecordDataset(expanded_files[i])
          for i in range(len(expanded_files))
      ]
      if is_training:
        dataset_list = [d.repeat() for d in dataset_list]
      dset_weights = [FLAGS.dataset_one_weight, 1 - FLAGS.dataset_one_weight]
      if FLAGS.dataset_two_weight != 0:
        dset_weights = [
            FLAGS.dataset_one_weight, FLAGS.dataset_two_weight,
            1 - FLAGS.dataset_one_weight + FLAGS.dataset_two_weight
        ]
      d = tf.data.experimental.sample_from_datasets(dataset_list, dset_weights)

      # Note that sample_from_datasets() inserts randomness into the training
      # An alternative would be to use choose_from_datasets() but then the
      # order must be stated explicitly which is less intitive for unbalanced
      # datasets. Example below:
      #
      # choice_dataset = tf.data.Dataset.range(len(dataset_list)).repeat()
      # d = tf.data.experimental.choose_from_datasets(dataset_list,
      #                                               choice_dataset)

      if is_training:
        d = d.shuffle(buffer_size=256)

    # The window size will be for selecting negative samples
    # It equals the number of documents to sample from -1
    d = d.apply(
        contrib_data.sliding_window_batch(
            window_size=FLAGS.data_window_size,
            window_shift=FLAGS.data_window_shift))
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features, vocab_table
                                         ),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_choices, add_masking):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = tf.reshape(features["input_ids"], [-1, FLAGS.max_seq_length])
    input_mask = tf.reshape(features["input_mask"], [-1, FLAGS.max_seq_length])
    segment_ids = tf.reshape(features["segment_ids"],
                             [-1, FLAGS.max_seq_length])

    label_types = features["label_types"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_real_example = tf.reduce_sum(
        tf.one_hot(label_types, FLAGS.k_size * 2), axis=1)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (cpc_loss, _, logits, probabilities) = bilin_model_builder.create_model(
        model, label_ids, label_types, num_choices, k_size=FLAGS.k_size)

    if add_masking:
      mask_rate = FLAGS.mask_rate  # search alternatives?
      max_predictions_per_seq = int(math.ceil(FLAGS.max_seq_length * mask_rate))
      masked_lm_positions = tf.reshape(features["mask_indices"],
                                       [-1, max_predictions_per_seq])
      masked_lm_ids = tf.reshape(features["target_token_ids"],
                                 [-1, max_predictions_per_seq])
      masked_lm_weights = tf.reshape(features["target_token_weights"],
                                     [-1, max_predictions_per_seq])
      (masked_lm_loss, _, _) = bilin_model_builder.get_masked_lm_output(
          bert_config, model.get_sequence_output(), model.get_embedding_table(),
          masked_lm_positions, masked_lm_ids, masked_lm_weights)
      total_loss = cpc_loss + masked_lm_loss
    else:
      total_loss = cpc_loss
      masked_lm_loss = tf.constant([0])

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

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(cpc_loss, mlm_loss, label_ids, logits, is_real_example):
        """Collect metrics for function."""

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        cpc_loss_metric = tf.metrics.mean(values=cpc_loss)
        mlm_loss_metric = tf.metrics.mean(values=mlm_loss)
        metric_dict = {
            "eval_accuracy": accuracy,
            "eval_cpc_loss": cpc_loss_metric,
            "eval_mlm_loss": mlm_loss_metric
        }
        for i in range(FLAGS.k_size * 2):
          metric_dict["acc" + str(i)] = tf.metrics.accuracy(
              labels=label_ids[:, i],
              predictions=predictions[:, i],
              weights=is_real_example[:, i])
        return metric_dict

      eval_metrics = (metric_fn, [
          cpc_loss, masked_lm_loss, label_ids, logits, is_real_example
      ])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train`, `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

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
    num_train_steps = int(FLAGS.train_data_size / FLAGS.train_batch_size)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      num_choices=FLAGS.num_choices,
      add_masking=FLAGS.include_mlm)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.train_file,
        is_training=True,
        drop_remainder=True,
        add_masking=FLAGS.include_mlm)
    estimator.train(input_fn=train_input_fn, steps=num_train_steps)

  if FLAGS.do_eval:
    # This tells the estimator to run through the entire set.
    if FLAGS.eval_data_size < 0:
      eval_steps = None
    else:
      eval_steps = int(FLAGS.eval_data_size / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    # Note that we are masking inputs for eval as well as training and this will
    # decrease eval performance
    eval_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.eval_file,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        add_masking=FLAGS.include_mlm)

    # checkpoints_iterator blocks until a new checkpoint appears.
    for ckpt in contrib_training.checkpoints_iterator(estimator.model_dir):
      try:
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        tf.logging.info("********** Eval results:*******\n")
        for key in sorted(result.keys()):
          tf.logging.info("%s = %s" % (key, str(result[key])))
      except tf.errors.NotFoundError:
        tf.logging.error("Checkpoint path '%s' no longer exists.", ckpt)


if __name__ == "__main__":
  flags.mark_flag_as_required("eval_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)

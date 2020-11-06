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
"""Run QA model with empty questions to generate answer candidates.

This code is adapted from the paper:
Chris Alberti, Daniel Andor, Emily Pitler, Jacob Devlin, and Michael Collins.
2019. Synthetic QA Corpora Generation with Roundtrip Consistency. In ACL.

Most irrelevant code has been stripped away (i.e., for training/preprocessing).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import functools
import itertools
import json
import os

from absl import app
from absl import flags
from bert import modeling
import numpy as np
import tensorflow.compat.v1 as tf


from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import layers as contrib_layers

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

## Required parameters
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("predict_precomputed_file", None,
                    "Precomputed tf records for preditions.")

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
    os.path.join(DATA_DIR, "qgen_model/answer_extraction/model.ckpt"),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("predict_precomputed_file", None,
                    "Precomputed tf records for preditions.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint",
    os.path.join(DATA_DIR, "qgen_model/answer_extraction/model.ckpt"),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_enum("compression_type", "", ["GZIP", ""],
                  "Compression type of input TFRecord files.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 10,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 10,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_enum("span_encoding", "concat-mlp", ["independent", "concat-mlp"],
                  "Different ways of modeling answer spans.")

flags.DEFINE_integer(
    "sample_nbest_list", 3,
    "If >0, we sample the predicted answer nbest list N times uniformly, "
    "otherwise we pick the top one.")

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

FLAGS = flags.FLAGS


class AnswerType(enum.IntEnum):
  """Type of answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  EXTRACTIVE = 3
  ABSTRACTIVE = 4


RELEVANT_FLAGS = [
    "n_best_size",
    "max_answer_length",
    "max_seq_length",
    "span_encoding",
]


class DataConfig(collections.namedtuple("DataConfig", RELEVANT_FLAGS)):

  @staticmethod
  def from_flags():
    relevant_flags_dict = {f: FLAGS[f].value for f in RELEVANT_FLAGS}
    return DataConfig(**relevant_flags_dict)


def compute_joint_mlp_logits(sequence, max_span_length):
  """Computes joint span (start, end) logits from sequence input."""
  batch_size, seq_length, hidden_size = modeling.get_shape_list(
      sequence, expected_rank=3)

  projection_size = hidden_size  # This seems to be a reasonable setting.

  with tf.variable_scope("joint_span"):
    projection = tf.layers.dense(
        sequence,
        projection_size * 2,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name="projection")
    start_projection, end_projection = tf.split(projection, 2, axis=-1)

    # 1. The start representations are tiled max_answer_length times.
    # TODO(danielandor): Use the mask to compute an optimal span list.
    starts = tf.reshape(start_projection,
                        [batch_size * seq_length, 1, projection_size])
    starts = tf.tile(starts, [1, max_span_length, 1])
    starts = tf.reshape(
        starts, [batch_size, seq_length * max_span_length, projection_size])

    # 2. To make the end representations, we compute band diagonal indices and
    #    perform a batched gather.
    seqs = tf.expand_dims(tf.range(seq_length), 1)
    offsets = tf.expand_dims(tf.range(max_span_length), 0)
    indices = seqs + offsets  # uses broadcasting
    indices.shape.assert_is_compatible_with((seq_length, max_span_length))
    indices = tf.reshape(indices, [1, seq_length * max_span_length])
    indices = tf.tile(indices, [batch_size, 1])
    indices = tf.minimum(indices, seq_length - 1)  # clips indices
    ends = tf.batch_gather(end_projection, indices)

    # 3. The final step adds the starts and ends.
    ends.shape.assert_is_compatible_with(starts.shape)
    inputs = starts + ends
    inputs = modeling.gelu(inputs)  # Bias is already in the projection.
    inputs = contrib_layers.layer_norm(inputs)
    start_logits = tf.layers.dense(
        inputs,
        1,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name="logits")
  return tf.reshape(start_logits, [batch_size, seq_length, max_span_length])


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 span_encoding, max_answer_length, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  if span_encoding == "independent":
    output_weights = tf.get_variable(
        "cls/coqa/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/coqa/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    start_logits, end_logits = tf.unstack(logits, axis=2)
  elif span_encoding == "concat-mlp":
    with tf.variable_scope("coqa"):
      if is_training:
        # The batch size can be variable during inference.
        final_hidden.shape.assert_is_compatible_with(
            (batch_size, seq_length, hidden_size))
      start_logits = compute_joint_mlp_logits(final_hidden, max_answer_length)
      start_logits = mask_joint_logits(input_mask, start_logits)
      end_logits = tf.zeros([batch_size], dtype=tf.float32)  # dummy
  else:
    raise ValueError("Unknown span_encoding: %s" % span_encoding)

  # Get the logits for the answer type prediction.
  # TODO(epitler): Try variants here.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, EXTRACTIVE, ABSTRACTIVE
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)
  return (start_logits, end_logits, answer_type_logits)


def mask_joint_logits(input_mask, start_end_logits):
  """Masks logits based on input mask and valid start/end combinations."""
  _, _, length = modeling.get_shape_list(start_end_logits, expected_rank=3)

  mask = tf.TensorArray(input_mask.dtype, size=length, dynamic_size=False)
  for i in range(length):
    mask = mask.write(i, input_mask)
    # The permitted span length is determined by the existing mask combined
    # with its being shifted up by one.
    input_mask = input_mask * tf.pad(input_mask[:, 1:], [[0, 0], [0, 1]])
  mask = mask.stack()
  mask = tf.transpose(mask, [1, 2, 0])
  mask.shape.assert_is_compatible_with(start_end_logits.shape)

  start_end_logits -= 1e6 * tf.cast(1 - mask, tf.float32)
  return start_end_logits


def model_fn_builder(bert_config, init_checkpoint, use_tpu, span_encoding,
                     max_answer_length, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del labels, params

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    start_logits, end_logits, answer_type_logits = create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        span_encoding=span_encoding,
        max_answer_length=max_answer_length,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    predictions = {
        "unique_ids": tf.identity(unique_ids),
        "start_logits": start_logits,
        "end_logits": end_logits,
        "answer_type_logits": answer_type_logits
    }

    # Input features need to be present in tf.Example output.
    predictions.update({
        "input_ids": tf.identity(input_ids),
        "input_mask": tf.identity(input_mask),
        "segment_ids": tf.identity(segment_ids),
        "start_positions": tf.identity(features["start_positions"]),
        "end_positions": tf.identity(features["end_positions"]),
        "answer_types": tf.identity(features["answer_types"])
    })

    output_spec = tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  # When generating tf.Examples we expect these features to be there.
  name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
  name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
  name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

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

    # Zero question because we are generating.
    example["input_ids"] = example["input_ids"] * example["segment_ids"]

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.Dataset.list_files(input_file, shuffle=False)
    d = d.apply(
        contrib_data.parallel_interleave(
            functools.partial(
                tf.data.TFRecordDataset,
                compression_type=FLAGS.compression_type),
            cycle_length=32,
            sloppy=is_training))
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


def make_estimator_from_flags(bert_config, init_checkpoint):
  """Builds TPUEstimator from args, FLAGS, and the model_fn_builder method."""
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.session_master,
      model_dir=FLAGS.output_dir,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          num_shards=FLAGS.num_tpu_cores,
          experimental_host_call_every_n_steps=1000,
          per_host_input_for_training=is_per_host))
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      span_encoding=FLAGS.span_encoding,
      max_answer_length=FLAGS.max_answer_length,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  return tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.predict_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)


RawPrediction = collections.namedtuple("RawPrediction", [
    "start_index", "end_index", "answer_type_index", "start_logit", "end_logit",
    "answer_type_logit"
])


def compute_extractive_predictions_using_results_only(result, span_encoding,
                                                      n_best_size,
                                                      max_answer_length):
  """Computes final extractive predictions based on logits."""
  prelim_predictions = []
  start_logits = [float(x) for x in result["start_logits"].flat]
  end_logits = [float(x) for x in result["end_logits"].flat]
  answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
  answer_type_index = AnswerType.EXTRACTIVE
  answer_type_logit = answer_type_logits[answer_type_index]

  if span_encoding == "independent":
    starts = get_best_indexes(start_logits, 50)
    ends = get_best_indexes(end_logits, 50)
    starts_ends = list(itertools.product(starts, ends))
    start_indexes, end_indexes = zip(*starts_ends)
  elif span_encoding == "concat-mlp":
    start_indexes, end_indexes = _get_best_joint_indexes(
        max_answer_length, start_logits, 50)
  else:
    raise ValueError("Unknown span_encoding: %s" % span_encoding)
  for start_index, end_index in zip(start_indexes, end_indexes):
    if start_index == 0 or end_index == 0:
      continue

    if end_index < start_index:
      continue
    length = end_index - start_index + 1
    if length > max_answer_length:
      continue

    if span_encoding == "independent":
      start_logit = start_logits[start_index]
      end_logit = end_logits[end_index]
    elif span_encoding == "concat-mlp":
      start_logit = start_logits[start_index * (max_answer_length - 1) +
                                 end_index]
      end_logit = 0
    else:
      raise ValueError("Unknown span_encoding: %s" % span_encoding)
    prelim_predictions.append(
        RawPrediction(
            start_index=start_index,
            end_index=end_index,
            answer_type_index=answer_type_index,
            start_logit=start_logit,
            end_logit=end_logit,
            answer_type_logit=answer_type_logit))

  prelim_predictions = sorted(
      prelim_predictions,
      # First sorts by the answer type classification, then uses the span
      # extent logits as the secondary key.
      key=lambda x: (x.answer_type_logit, x.start_logit + x.end_logit),
      reverse=True)

  nbest = prelim_predictions[:n_best_size]

  # In very rare edge cases we could have no valid predictions. So we
  # just create a nonce prediction in this case to avoid failure.
  if not nbest:
    nbest.append(
        RawPrediction(
            start_index=0,
            end_index=0,
            answer_type_index=AnswerType.UNKNOWN,
            start_logit=0,
            end_logit=0,
            answer_type_logit=0))

  assert len(nbest) >= 1
  return nbest


def get_best_indexes(logits, n_best_size):
  """Gets the indices of the n-best logits from a list."""
  indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
  return indices[:n_best_size]


def _get_best_joint_indexes(max_span_length, logits, n_best_size):
  """Gets the n-best start and end indices from flat logits."""
  indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
  indices = indices[:n_best_size]
  starts = [x // max_span_length for x in indices]
  offsets = [x % max_span_length for x in indices]
  starts, ends = zip(*[(s, s + offs) for s, offs in zip(starts, offsets)])
  return starts, ends


def pick_answer_from_nbest(nbest, result):
  """Picks from nbest uniformly at random, or return s."""
  # If no answer can be picked, default to a null span.
  answer = [
      RawPrediction(
          start_index=0,
          end_index=0,
          answer_type_index=AnswerType.UNKNOWN,
          start_logit=0,
          end_logit=0,
          answer_type_logit=0)
  ]

  if FLAGS.sample_nbest_list > 0:
    # Pick n predicted answers from the nbest list uniformly at random.
    candidate_answers = [t for t in nbest if t.start_index > 0]
    indices = list(range(len(candidate_answers)))
    np.random.shuffle(indices)
    if candidate_answers:
      answer = [candidate_answers[i] for i in indices[:FLAGS.sample_nbest_list]]
  else:
    # When not sampling an answer, we assume that this is being used
    # for roundtrip filtering, so we only output an extractive answer
    # if it matches the input extractive answer.
    t = nbest[0]
    if (result["start_positions"] == t.start_index and
        result["end_positions"] == t.end_index):
      answer = [t]

  return answer


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  data_config = DataConfig.from_flags()
  experiment_config_path = os.path.join(FLAGS.output_dir,
                                        "experiment_config.json")
  with tf.gfile.GFile(experiment_config_path, "w") as writer:
    json.dump(data_config._asdict(), writer, indent=4)

  estimator = make_estimator_from_flags(bert_config, FLAGS.init_checkpoint)

  num_eval_features = len(
      list(tf.python_io.tf_record_iterator(FLAGS.predict_precomputed_file)))
  eval_filename = FLAGS.predict_precomputed_file

  tf.logging.info("***** Running predictions on precomputed features *****")
  tf.logging.info("  Num split examples = %d", num_eval_features)
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  predict_input_fn = input_fn_builder(
      input_file=eval_filename,
      seq_length=data_config.max_seq_length,
      is_training=False,
      drop_remainder=False)

  output_file = os.path.join(FLAGS.output_dir, "predicted-tfrecords")

  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  with tf.python_io.TFRecordWriter(output_file) as writer:
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      nbest = compute_extractive_predictions_using_results_only(
          result, data_config.span_encoding, data_config.n_best_size,
          data_config.max_answer_length)
      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([int(result["unique_ids"])])
      features["input_ids"] = create_int_feature(result["input_ids"])
      features["input_mask"] = create_int_feature(result["input_mask"])
      features["segment_ids"] = create_int_feature(result["segment_ids"])
      for answer in pick_answer_from_nbest(nbest, result):
        features["start_positions"] = create_int_feature([answer.start_index])
        features["end_positions"] = create_int_feature([answer.end_index])
        features["answer_types"] = create_int_feature(
            [answer.answer_type_index])
        writer.write(
            tf.train.Example(features=tf.train.Features(
                feature=features)).SerializeToString())


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.disable_v2_behavior()
  app.run(main)

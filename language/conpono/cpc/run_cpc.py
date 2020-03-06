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

from absl import app
from absl import flags

from bert import modeling
from bert import optimization
from bert import tokenization
from language.conpono.cpc import model_builder
import tensorflow.compat.v1 as tf


from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
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

flags.DEFINE_integer("num_choices", 32, "Number of negative samples + 1")

flags.DEFINE_bool("add_lv2loss", False, "Whether to use the level 2 loss.")

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


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_choices):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  input_file = input_file.split(",")

  expanded_files = []
  for infile in input_file:
    try:
      sharded_files = tf.io.gfile.glob(infile)
      expanded_files.append(sharded_files)
    except tf.errors.OpError:
      expanded_files.append(infile)

  name_to_features = {}
  for i in range(50):
    name_to_features["input_ids" + str(i)] = tf.FixedLenFeature([seq_length],
                                                                tf.int64)
    name_to_features["input_mask" + str(i)] = tf.FixedLenFeature([seq_length],
                                                                 tf.int64)
    name_to_features["segment_ids" + str(i)] = tf.FixedLenFeature([seq_length],
                                                                  tf.int64)
  name_to_features["label_types"] = tf.FixedLenFeature([4], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # total of 32 examples
    # 4 labels / shuffled
    # random samples from 23 + 23 last for distractors
    num_sampled = 14
    same_doc_idxs = tf.random.shuffle(tf.range(4, 27))[:num_sampled]
    rand_doc_idxs = tf.random.shuffle(tf.range(27, 50))[:num_sampled]
    batch_indexes = tf.concat([tf.range(4), same_doc_idxs, rand_doc_idxs],
                              axis=0)
    batch_indexes = tf.random.shuffle(batch_indexes)
    # At this point, we have shuffled the indexes and sampled them such that
    # we still have the index of 4 targets, 14 sampled from the same doc
    # and 14 sampled from different docs. But these are just indexes.

    # Here we need to grab the inputs according to the indexes above
    # We stack all the inputs so we can gather on the matrix
    input_id_stack, input_mask_stack, segment_id_stack = [], [], []
    for i in range(50):
      input_id_stack.append(example["input_ids" + str(i)])
      input_mask_stack.append(example["input_mask" + str(i)])
      segment_id_stack.append(example["segment_ids" + str(i)])
    input_id_stack = tf.stack(input_id_stack)
    input_mask_stack = tf.stack(input_mask_stack)
    segment_id_stack = tf.stack(segment_id_stack)

    input_ids = tf.gather(input_id_stack, batch_indexes)
    input_masks = tf.gather(input_mask_stack, batch_indexes)
    segment_ids = tf.gather(segment_id_stack, batch_indexes)

    # Note that we override the name of the input (eg. input_ids5)
    # So we replace the input with the shuffled and sampled input
    # We only set num_choices of them since those will be used.
    for i in range(num_choices):
      example["input_ids" + str(i)] = input_ids[i]
      example["input_mask" + str(i)] = input_masks[i]
      example["segment_ids" + str(i)] = segment_ids[i]

    # Note that for inputs num_choices-50 will not be used so we must purge them
    for i in range(num_choices, 50):
      del example["input_ids" + str(i)]
      del example["input_mask" + str(i)]
      del example["segment_ids" + str(i)]

    label_idx = []
    for i in range(4):
      label_idx.append(tf.where(tf.equal(batch_indexes, tf.constant(i)))[0])
    label_idx = tf.reshape(tf.concat(label_idx, axis=0), [-1])
    label_idx = tf.scatter_nd(
        tf.reshape(example["label_types"], [4, 1]), label_idx, [8])
    example["labels"] = label_idx

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
      wiki_pct = 0.02222
      dset_weights = [wiki_pct, 1 - wiki_pct]
      d = tf.data.experimental.sample_from_datasets(dataset_list, dset_weights)
      # choice_dataset = tf.data.Dataset.range(len(dataset_list)).repeat()

      # d = tf.data.experimental.choose_from_datasets(dataset_list,
      # choice_dataset)
      if is_training:
        d = d.shuffle(buffer_size=256)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_choices):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = [features["input_ids" + str(i)] for i in range(num_choices)]
    input_mask = [features["input_mask" + str(i)] for i in range(num_choices)]
    segment_ids = [features["segment_ids" + str(i)] for i in range(num_choices)]
    label_ids = features["labels"]
    label_types = features["label_types"]

    seq_length = input_ids[0].shape[-1]
    input_ids = tf.reshape(tf.stack(input_ids, axis=1), [-1, seq_length])
    input_mask = tf.reshape(tf.stack(input_mask, axis=1), [-1, seq_length])
    segment_ids = tf.reshape(tf.stack(segment_ids, axis=1), [-1, seq_length])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_real_example = tf.reduce_sum(tf.one_hot(label_types, 8), axis=1)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (total_loss, per_example_loss, logits,
     probabilities) = model_builder.create_model(
         model, label_ids, label_types,
         FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size,
         num_choices, use_tpu, FLAGS.add_lv2loss)

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

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        """Collect metrics for function."""

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        metric_dict = {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }
        for i in range(8):
          metric_dict["acc" + str(i)] = tf.metrics.accuracy(
              labels=label_ids[:, i],
              predictions=predictions[:, i],
              weights=is_real_example[:, i])
        return metric_dict

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
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
      num_choices=FLAGS.num_choices)

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
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        num_choices=FLAGS.num_choices)
    estimator.train(input_fn=train_input_fn, steps=num_train_steps)

  if FLAGS.do_eval:
    # This tells the estimator to run through the entire set.
    if FLAGS.eval_data_size < 0:
      eval_steps = None
    else:
      eval_steps = int(FLAGS.eval_data_size / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        num_choices=FLAGS.num_choices)

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

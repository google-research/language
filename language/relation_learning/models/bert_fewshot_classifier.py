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
"""Few-shot relations classifier, based on BERT-like Transformers models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import tempfile
from absl import flags
from bert import modeling
from bert import tokenization
from language.relation_learning.data import fewrel
import tensorflow as tf

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input", None,
    "The input filename. Should contain the JSON file with FewRel examples.")

flags.DEFINE_string(
    "output", None,
    "The output filename. JSON formatted label ids for model predictions.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("checkpoint", None,
                    "Model checkpoint used for predictions.")

## Other parameters

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_enum(
    "fewshot_examples_combiner", "max",
    ["max", "mean", "logsumexp", "min", "sigmoid_mean"],
    "Reduction operation to use for combining scores for examples within the "
    "same class (when --fewshot_num_examples_per_class > 1). Currently "
    'supports "max", "mean" and "sigmoid_mean".')


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, guid, input_ids, input_mask, segment_ids):
    self.guid = guid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens = example.wordpieces
  segment_ids = [0] * len(tokens)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 10:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(tokens).encode("utf-8"))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  feature = InputFeatures(
      guid=example.guid,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)

  return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_string_feature(values):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
    return f

  def feature_list_from_input_feature(input_feature):
    features = collections.OrderedDict()
    features["guid"] = [create_string_feature([input_feature.guid])]
    features["input_ids"] = [create_int_feature(input_feature.input_ids)]
    features["input_mask"] = [create_int_feature(input_feature.input_mask)]
    features["segment_ids"] = [create_int_feature(input_feature.segment_ids)]
    return features

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    query_features = feature_list_from_input_feature(
        convert_single_example(ex_index, example.query, max_seq_length,
                               tokenizer))
    features_list_dict = {
        name: tf.train.FeatureList() for name in query_features.keys()
    }
    for name in query_features:
      features_list_dict[name].feature.extend(query_features[name])

    for _, test_set in example.sets.iteritems():
      for test_example in test_set:
        test_features = feature_list_from_input_feature(
            convert_single_example(ex_index, test_example, max_seq_length,
                                   tokenizer))
        for name in test_features:
          features_list_dict[name].feature.extend(test_features[name])

    sequence_example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=features_list_dict))
    writer.write(sequence_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, fewshot_num_classes,
                                fewshot_num_examples_per_class, drop_remainder):
  """Creates an `input_fn` closure to be passed to tf.Estimator."""

  # Add one for the 'query' example.
  fewshot_batch = fewshot_num_classes * fewshot_num_examples_per_class + 1
  name_to_features = {
      "input_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "guid": tf.FixedLenSequenceFeature([], tf.string),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    _, example = tf.parse_single_sequence_example(
        record, sequence_features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      shape = tf.shape(example[name])
      # sequence_examples come with dynamic/unknown dimension which we reshape
      # to explicit dimension for the fewshot "batch" size.
      example[name] = tf.reshape(t, tf.concat([[fewshot_batch], shape[1:]], 0))

    return example

  def input_fn(params):
    """The actual input function."""
    d = tf.data.TFRecordDataset(input_file)
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=params["batch_size"],
            drop_remainder=drop_remainder))

    return d

  return input_fn


def extract_relation_representations(input_layer, input_ids, tokenizer):
  """Extracts relation representation from sentence sequence layer."""
  entity_representations = []
  entity_marker_ids = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])
  for entity_marker_id in entity_marker_ids:
    mask = tf.to_float(tf.equal(input_ids, entity_marker_id))
    mask = tf.broadcast_to(tf.expand_dims(mask, -1), tf.shape(input_layer))
    entity_representation = tf.reduce_max(
        mask * input_layer, axis=1, keepdims=True)
    entity_representations.append(entity_representation)

  output_layer = tf.concat(entity_representations, axis=2)
  output_layer = tf.squeeze(output_layer, [1])
  tf.logging.info("entity marker pooling AFTER output shape %s",
                  output_layer.shape)

  return output_layer


def create_model(bert_config,
                 is_training,
                 fewshot_num_examples_per_class,
                 input_ids,
                 input_mask,
                 segment_ids,
                 use_one_hot_embeddings,
                 tokenizer=None,
                 class_examples_combiner="max"):
  """Creates a classification model."""
  if not is_training:
    bert_config.hidden_dropout_prob = 0.0
    bert_config.attention_probs_dropout_prob = 0.0

  # unroll fewshot batches to extract BERT representations.
  fewshot_size = input_ids.shape[1].value
  sequence_length = input_ids.shape[2].value

  bert_input_ids = tf.reshape(input_ids, [-1, sequence_length])
  bert_input_mask = tf.reshape(input_mask, [-1, sequence_length])
  bert_segment_ids = tf.reshape(segment_ids, [-1, sequence_length])
  tf.logging.info(
      "shapes %s %s %s" %
      (bert_input_ids.shape, bert_input_mask.shape, bert_segment_ids.shape))

  model = modeling.BertModel(
      config=bert_config,
      is_training=FLAGS.train_bert_model if is_training else False,
      input_ids=bert_input_ids,
      input_mask=bert_input_mask,
      token_type_ids=bert_segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # [batch_size, fewshot_size * seq_len, hidden_size]
  output_layer = model.get_sequence_output()
  tf.logging.info("BERT model output shape %s", output_layer.shape)

  # The "pooler" converts the encoded sequence tensor of shape
  # [batch_size, seq_length, hidden_size] to a tensor of shape
  # [batch_size, 2*hidden_size].
  with tf.variable_scope("cls/entity_relation"):
    # [batch_size, fewshot_size, 2 * hidden_size]
    output_layer = extract_relation_representations(output_layer,
                                                    bert_input_ids, tokenizer)
    output_layer = modeling.layer_norm(output_layer)

  def _combine_multi_example_logits(logits):
    """Combine per-example logits into a per-class logit."""
    logits = tf.reshape(
        logits, [-1, fewshot_num_classes, fewshot_num_examples_per_class, 1])
    if class_examples_combiner == "max":
      logits = tf.reduce_max(logits, axis=2)
    if class_examples_combiner == "mean":
      logits = tf.reduce_mean(logits, axis=2)
    if class_examples_combiner == "logsumexp":
      logits = tf.reduce_logsumexp(logits, axis=2)
    if class_examples_combiner == "min":
      logits = tf.reduce_min(logits, axis=2)
    if class_examples_combiner == "sigmoid_mean":
      logits = tf.sigmoid(logits)
      logits = tf.reduce_mean(logits, axis=2)
    return logits

  fewshot_num_classes = int((fewshot_size - 1) / fewshot_num_examples_per_class)
  hidden_size = output_layer.shape[-1].value
  with tf.variable_scope("loss"):
    # [batch_size, fewshot_size, hidden_size]
    output_weights = tf.reshape(output_layer, [-1, fewshot_size, hidden_size])

    # Extract query representation from output.
    # [batch_size, fewshot_size - 1, hidden_size]
    output_layer = tf.reshape(output_weights[:, 0, :], [-1, 1, hidden_size])

    # Remove query from targets.
    # [batch_size, 1, hidden_size]
    output_weights = output_weights[:, 1:, :]

    # Dot product based distance metric.
    # [batch_size, fewshot_size - 1, 1]
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)

    if fewshot_num_examples_per_class > 1:
      # [batch_size, fewshot_num_classes, 1]
      logits = _combine_multi_example_logits(logits)

    # [batch_size, fewshot_num_classes]
    logits = tf.reshape(logits, [-1, fewshot_num_classes])

  return logits


def model_fn_builder(bert_config,
                     use_one_hot_embeddings,
                     fewshot_num_examples_per_class,
                     tokenizer=None,
                     class_examples_combiner="max"):
  """Returns `model_fn` closure for tf.Estimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for tf.Estimator."""
    del labels, params

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT mode is supported: %s" % (mode))

    tf.logging.info("*** Features *** %s %s" % (type(features), features))
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    guid = features["guid"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    logits = create_model(
        bert_config=bert_config,
        is_training=False,
        fewshot_num_examples_per_class=fewshot_num_examples_per_class,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        tokenizer=tokenizer,
        class_examples_combiner=class_examples_combiner)

    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    output_spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions={
            "predictions": predictions,
            "guid": guid,
        })
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # Retrieve entries from FewRel input file. We do this before model building
  # so we can determine the number of classes and examples per class to use
  # in model building.
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  processor = fewrel.FewRelProcessor(
      tokenizer, FLAGS.max_seq_length, add_entity_markers=True)
  (predict_examples, fewshot_num_classes_eval,
   fewshot_num_examples_per_class) = processor.process_file(FLAGS.input)

  # Build model.
  model_fn = model_fn_builder(
      bert_config=bert_config,
      use_one_hot_embeddings=False,
      fewshot_num_examples_per_class=fewshot_num_examples_per_class,
      tokenizer=tokenizer,
      class_examples_combiner=FLAGS.fewshot_examples_combiner)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, params={"batch_size": FLAGS.predict_batch_size})

  # Convert examples into tensorflow examples, and store to a file.
  temp_dir = tempfile.mkdtemp()
  predict_file = os.path.join(temp_dir, "predict.tf_record")
  file_based_convert_examples_to_features(
      predict_examples, FLAGS.max_seq_length, tokenizer, predict_file)

  input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      fewshot_num_classes=fewshot_num_classes_eval,
      fewshot_num_examples_per_class=fewshot_num_examples_per_class,
      drop_remainder=False)

  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Num examples = %d", len(predict_examples))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  # Perform predictions.
  predictions = []
  for item in estimator.predict(
      input_fn=input_fn, checkpoint_path=FLAGS.checkpoint):
    tf.logging.info("%s\t%s", item["guid"], item["predictions"])
    predictions.append(int(item["predictions"]))

  # Dump predictions to output file.
  output_predictions_file = os.path.join(FLAGS.output)
  with tf.gfile.GFile(output_predictions_file, "w") as writer:
    json.dump(predictions, writer)
    writer.write("\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  flags.mark_flag_as_required("output")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("checkpoint")
  tf.app.run()

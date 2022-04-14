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
"""Define the paragraph reconstruction model."""

import collections
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf


def create_model(model, labels, num_choices):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    labels: ground truth paragraph order
    num_choices: number of negatives samples + 1

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  dense_layer_weights = tf.get_variable(
      "dense_layer", [(1 + num_choices) * hidden_size, num_choices],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  with tf.variable_scope("cpc_loss"):
    example_repr = tf.reshape(output_layer,
                              (-1, (num_choices + 1) * hidden_size))
    logits = tf.matmul(example_repr, dense_layer_weights)

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    probabilities = tf.nn.softmax(logits, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

  return (loss, per_example_loss, logits, probabilities)


def create_model_bilin(model, labels, num_choices):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    labels: ground truth paragraph order
    num_choices: number of negatives samples + 1

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cpc_loss"):
    output = tf.reshape(output_layer, (-1, num_choices + 1, hidden_size))
    contexts = output[:, 0, :]
    targets = output[:, 1:, :]

    softmax_weights = tf.get_variable(
        "cpc_weights", [8, hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    softmax_weights = softmax_weights[4:5]

    context_encoded = tf.matmul(softmax_weights, contexts, transpose_b=True)
    context_encoded = tf.transpose(context_encoded, perm=[2, 0, 1])

    logits = tf.matmul(targets, context_encoded, transpose_b=True)
    logits = tf.transpose(logits, perm=[0, 2, 1])
    logits = tf.squeeze(logits, axis=[1])
    # labels = tf.squeeze(labels, axis=[1])

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    probabilities = tf.nn.softmax(logits, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

  return (loss, per_example_loss, logits, probabilities)


def load_hellaswag(data_dir):
  data = []
  with tf.gfile.Open(data_dir, "r") as handle:
    for line in handle:
      data.append(json.loads(line))
  return data


def convert_single_example_for_bilinear(ex_index,
                                        example,
                                        max_seq_length,
                                        tokenizer,
                                        do_copa=False):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  # Add padding examples here

  example_type = collections.namedtuple(
      "Example", ["input_ids", "input_mask", "segment_ids", "labels"])

  if do_copa:
    sents = [example["premise"], example["choice1"], example["choice2"]]
    sents = [tokenizer.tokenize(sent) for sent in sents]

    for ending_i in range(1, 3):
      assert len(sents[0]) + len(sents[ending_i]) < (max_seq_length -
                                                     3), "Example is too long"
  else:
    sents = [example["ctx_a"]
            ] + [example["ctx_b"] + " " + e for e in example["endings"]]
    sents = [tokenizer.tokenize(sent) for sent in sents]

    for ending_i in range(1, 5):
      assert len(sents[0]) + len(sents[ending_i]) < (max_seq_length -
                                                     3), "Example is too long"

  input_id_list, sengment_id_list, input_mask_list = [], [], []

  # context encoder:
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in sents[0]:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

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

  input_id_list.append(input_ids)
  sengment_id_list.append(segment_ids)
  input_mask_list.append(input_mask)

  choices_size = 2 if do_copa else 4
  for i in range(choices_size):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in sents[0]:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in sents[i + 1]:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

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

    input_id_list.append(input_ids)
    sengment_id_list.append(segment_ids)
    input_mask_list.append(input_mask)

  labels = [0] * 8
  labels[4] = example["label"]

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("tokens: %s" % " ".join(tokens))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  feature = example_type(input_id_list, input_mask_list, sengment_id_list,
                         labels)
  return feature


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  # Add padding examples here

  example_type = collections.namedtuple(
      "Example", ["input_ids", "input_mask", "segment_ids", "labels"])

  sents = [example["ctx"]] + example["endings"]
  sents = [tokenizer.tokenize(sent) for sent in sents]

  for ending_i in range(1, 5):
    assert len(sents[0]) + len(sents[ending_i]) < (max_seq_length -
                                                   3), "Example is too long"

  input_id_list, sengment_id_list, input_mask_list = [], [], []
  for i in range(4):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in sents[0]:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in sents[i + 1]:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(0)

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

    input_id_list.append(input_ids)
    sengment_id_list.append(segment_ids)
    input_mask_list.append(input_mask)

  labels = [0] * 8
  labels[4] = example["label"]

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("tokens: %s" % " ".join(tokens))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  feature = example_type(input_id_list, input_mask_list, sengment_id_list,
                         labels)
  return feature


def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  dirname = os.path.dirname(output_file)
  if not tf.gfile.Exists(dirname):
    tf.gfile.MakeDirs(dirname)
  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    input_feature = convert_single_example(ex_index, example, max_seq_length,
                                           tokenizer)

    features = collections.OrderedDict()
    for i in range(4):
      features["input_ids" + str(i)] = create_int_feature(
          input_feature.input_ids[i])
      features["input_mask" + str(i)] = create_int_feature(
          input_feature.input_mask[i])
      features["segment_ids" + str(i)] = create_int_feature(
          input_feature.segment_ids[i])
    features["labels"] = create_int_feature(input_feature.labels)
    features["label_types"] = create_int_feature([4])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_convert_examples_for_bilinear(examples,
                                             max_seq_length,
                                             tokenizer,
                                             output_file,
                                             do_copa=False):
  """Convert a set of `InputExample`s to a TFRecord file."""

  dirname = os.path.dirname(output_file)
  if not tf.gfile.Exists(dirname):
    tf.gfile.MakeDirs(dirname)
  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      tf.logging.info("DANITER:Writing example %d of %d" %
                      (ex_index, len(examples)))

    input_feature = convert_single_example_for_bilinear(ex_index, example,
                                                        max_seq_length,
                                                        tokenizer, do_copa)

    features = collections.OrderedDict()
    input_size = 3 if do_copa else 5
    for i in range(input_size):
      features["input_ids" + str(i)] = create_int_feature(
          input_feature.input_ids[i])
      features["input_mask" + str(i)] = create_int_feature(
          input_feature.input_mask[i])
      features["segment_ids" + str(i)] = create_int_feature(
          input_feature.segment_ids[i])
    features["labels"] = create_int_feature(input_feature.labels)
    features["label_types"] = create_int_feature([4])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def load_record(filename):
  record_data = []
  with tf.gfile.Open(filename, "r") as handle:
    for line in handle:
      record_data.append(json.loads(line))
  return record_data


def file_based_convert_examples_record(examples, max_seq_length, tokenizer,
                                       output_file):
  """Convert record json data to tfrecord."""
  dirname = os.path.dirname(output_file)
  if not tf.gfile.Exists(dirname):
    tf.gfile.MakeDirs(dirname)
  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      tf.logging.info("DANITER:Writing example %d of %d" %
                      (ex_index, len(examples)))

    input_features = convert_single_example_record(example, max_seq_length,
                                                   tokenizer)

    for input_feature in input_features:
      features = collections.OrderedDict()
      cand_size = 25
      features["context_input_ids"] = create_int_feature(input_feature.text)
      for i in range(len(input_feature.completions)):
        features["comp_input_ids" + str(i)] = create_int_feature(
            input_feature.completions[i])
      for i in range(len(input_feature.completions), cand_size):
        rand_entry = np.random.randint(len(input_feature.completions))
        features["comp_input_ids" + str(i)] = create_int_feature(
            input_feature.completions[rand_entry])

      assert len(features.keys()) == (cand_size + 1)
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
  writer.close()


Example = collections.namedtuple("Example", ["text", "completions"])


def create_completion(question, ent):
  return question.replace("@placeholder", ent)


def convert_example_to_cands(rec):
  """Generate candidate completions for each example."""
  cand_size = 25
  example_list = []
  text = rec["passage"]["text"]
  text_clean = text.replace("@highlight", " ")
  ent_set = set()
  for ent in rec["passage"]["entities"]:
    ent_str = text[ent["start"]:ent["end"] + 1]
    ent_set.add(ent_str.lower())
  for qa in rec["qas"]:
    question = qa["query"]
    answers = set([q["text"].lower() for q in qa["answers"]])

    # incorrect ents
    neg_ents = [ent for ent in ent_set if ent not in answers]
    completions = []
    completions.append(create_completion(question, list(answers)[0]))
    completions.extend([
        create_completion(question, neg_ent)
        for neg_ent in neg_ents[:cand_size - 1]
    ])
    example_list.append(Example(text_clean, completions))
  return example_list


def convert_single_example_record(example, max_seq_length, tokenizer):
  """Convert single record example to features."""

  example_list = convert_example_to_cands(example)
  feature_list = []
  for struct_example in example_list:
    text = struct_example.text
    completions = struct_example.completions
    text = tokenizer.tokenize(text)
    completions = [tokenizer.tokenize(c) for c in completions]
    context_input_ids = tokenizer.convert_tokens_to_ids(text)
    comp_input_ids = [tokenizer.convert_tokens_to_ids(c) for c in completions]

    # cut size and pad
    context_input_ids = context_input_ids[:max_seq_length]
    context_input_ids += [0] * (max_seq_length - len(context_input_ids))
    max_target_len = 100
    comp_input_ids = [
        c[:max_target_len] + ([0] * (max_target_len - len(c)))
        for c in comp_input_ids
    ]

    feature = Example(context_input_ids, comp_input_ids)
    feature_list.append(feature)
  return feature_list

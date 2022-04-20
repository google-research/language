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
"""Classes for processing different datasets into a common format."""

import collections
import json
import random

from bert import tokenization

import tensorflow.compat.v1 as tf
from tqdm import tqdm
from tensorflow.contrib import data as contrib_data


class Example:
  """A single training/test example for QA."""

  def __init__(self,
               qas_id,
               question_text,
               subject_entity,
               relations=None,
               answer_mention=None,
               answer_entity=None,
               bridge_mention=None,
               bridge_entity=None,
               inference_chain=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.subject_entity = subject_entity
    self.relations = relations
    self.answer_mention = answer_mention
    self.answer_entity = answer_entity
    self.bridge_mention = bridge_mention
    self.bridge_entity = bridge_entity
    self.inference_chain = inference_chain

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    if self.answer_mention:
      s += ", answer_mention: %d" % self.answer_mention[0]
    return s


class InputFeatures:
  """A single set of features of data."""

  def __init__(self,
               qas_id,
               qry_tokens,
               qry_input_ids,
               qry_input_mask,
               relation_input_ids,
               relation_input_mask,
               qry_entity_id,
               answer_mention=None,
               answer_entity=None,
               bridge_entity=None,
               bridge_mention=None):
    self.qas_id = qas_id
    self.qry_tokens = qry_tokens
    self.qry_input_ids = qry_input_ids
    self.qry_input_mask = qry_input_mask
    self.relation_input_ids = relation_input_ids
    self.relation_input_mask = relation_input_mask
    self.qry_entity_id = qry_entity_id
    self.answer_mention = answer_mention
    self.answer_entity = answer_entity
    self.bridge_mention = bridge_mention
    self.bridge_entity = bridge_entity


class FeatureWriter:
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training, has_bridge):
    self.filename = filename
    self.is_training = is_training
    self.has_bridge = has_bridge
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    features = collections.OrderedDict()
    features["qas_ids"] = create_bytes_feature(feature.qas_id)
    features["qry_input_ids"] = create_int_feature(feature.qry_input_ids)
    features["qry_input_mask"] = create_int_feature(feature.qry_input_mask)
    features["qry_entity_id"] = create_int_feature(feature.qry_entity_id)

    if feature.relation_input_ids:
      for ii in range(len(feature.relation_input_ids)):
        features["rel_input_ids_%d" % ii] = create_int_feature(
            feature.relation_input_ids[ii])
        features["rel_input_mask_%d" % ii] = create_int_feature(
            feature.relation_input_mask[ii])

    if self.is_training:
      if feature.answer_mention is not None:
        features["answer_mentions"] = create_int_feature(feature.answer_mention)
      features["answer_entities"] = create_int_feature(feature.answer_entity)

    if self.has_bridge:
      if feature.bridge_mention is not None:
        features["bridge_mentions"] = create_int_feature(feature.bridge_mention)
      for ii, bridge_entity in enumerate(feature.bridge_entity):
        features["bridge_entities_%d" % ii] = create_int_feature(bridge_entity)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def get_tokens_and_mask(text, tokenizer, max_length):
  """Tokenize text and pad to max_length."""
  query_tokens = tokenizer.tokenize(text)

  if len(query_tokens) > max_length - 2:
    # -2 for [CLS], [SEP]
    query_tokens = query_tokens[0:max_length - 2]

  qry_tokens = []
  qry_tokens.append("[CLS]")
  for token in query_tokens:
    qry_tokens.append(token)
  qry_tokens.append("[SEP]")

  qry_input_ids = tokenizer.convert_tokens_to_ids(qry_tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  qry_input_mask = [1] * len(qry_input_ids)

  # Zero-pad up to the sequence length.
  while len(qry_input_ids) < max_length:
    qry_input_ids.append(0)
    qry_input_mask.append(0)

  assert len(qry_input_ids) == max_length
  assert len(qry_input_mask) == max_length

  return qry_input_ids, qry_input_mask, qry_tokens


def convert_examples_to_features(examples, tokenizer, max_query_length,
                                 entity2id, output_fn):
  """Loads a data file into a list of `InputBatch`s."""
  for (example_index, example) in tqdm(enumerate(examples)):
    qry_input_ids, qry_input_mask, qry_tokens = get_tokens_and_mask(
        example.question_text, tokenizer, max_query_length)
    relation_input_ids, relation_input_mask = [], []
    if example.relations is not None:
      for relation in example.relations:
        rel_input_ids, rel_input_mask, _ = get_tokens_and_mask(
            relation, tokenizer, max_query_length)
        relation_input_ids.append(rel_input_ids)
        relation_input_mask.append(rel_input_mask)
    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s", example.qas_id)
      tf.logging.info(
          "qry_tokens: %s",
          " ".join([tokenization.printable_text(x) for x in qry_tokens]))
      tf.logging.info("qry_input_ids: %s",
                      " ".join([str(x) for x in qry_input_ids]))
      tf.logging.info("qry_input_mask: %s",
                      " ".join([str(x) for x in qry_input_mask]))
      for ii in range(len(relation_input_ids)):
        tf.logging.info("relation_input_ids_%d: %s", ii,
                        " ".join([str(x) for x in relation_input_ids[ii]]))
        tf.logging.info("relation_input_mask_%d: %s", ii,
                        " ".join([str(x) for x in relation_input_mask[ii]]))
      tf.logging.info("qry_entity_id: %s (%d)", example.subject_entity[0],
                      entity2id.get(example.subject_entity[0], None))
      tf.logging.info("answer entity: %s", str(example.answer_entity))

    feature = InputFeatures(
        qas_id=example.qas_id.encode("utf-8"),
        qry_tokens=qry_tokens,
        qry_input_ids=qry_input_ids,
        qry_input_mask=qry_input_mask,
        relation_input_ids=relation_input_ids,
        relation_input_mask=relation_input_mask,
        qry_entity_id=[entity2id.get(ee, 0) for ee in example.subject_entity],
        answer_mention=example.answer_mention,
        answer_entity=example.answer_entity,
        bridge_mention=example.bridge_mention,
        bridge_entity=example.bridge_entity)

    # Run callback
    output_fn(feature)


def input_fn_builder(input_file, is_training, drop_remainder,
                     names_to_features):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

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
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, names_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class OneHopDataset:
  """Reads a dataset of one-hop queries and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file, p=subject_mention_probability)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=False)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.FixedLenFeature([], tf.int64),
    }
    if is_training:
      names_to_features["answer_mentions"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["answer_entities"] = tf.FixedLenFeature([], tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=True,
        names_to_features=names_to_features)

  def read_examples(self, queries_file, p=1.0):
    """Read a json file into a list of Example."""
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for line in tqdm(reader):
        item = json.loads(line.strip())

        qas_id = item["id"]
        relation = random.choice(item["relation"]["text"])
        if item["subject"]["name"] is None or random.uniform(0., 1.) < p:
          question_text = (
              random.choice(item["subject"]["mentions"])["text"] + " . " +
              relation)
        else:
          question_text = item["subject"]["name"] + " . " + relation
        answer_mention = item["object"]["global_mention"]
        answer_entity = item["object"]["ent_id"]

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=[item["subject"]["wikidata_id"]],
            relations=[relation],
            answer_mention=[answer_mention],
            answer_entity=[answer_entity])
        examples.append(example)

    return examples


class TwoHopDataset:
  """Reads a dataset of one-hop queries and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file, p=subject_mention_probability)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=True)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.FixedLenFeature([], tf.int64),
        "rel_input_ids_0": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_mask_0": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_ids_1": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_mask_1": tf.FixedLenFeature([self.max_qry_length], tf.int64),
    }
    if is_training:
      names_to_features["answer_mentions"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["answer_entities"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["bridge_mentions"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["bridge_entities_0"] = tf.FixedLenFeature([], tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=True,
        names_to_features=names_to_features)

  def read_examples(self, queries_file, p=1.0):
    """Read a json file into a list of Example."""
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for line in tqdm(reader):
        item = json.loads(line.strip())

        qas_id = item["id"]
        relation_1 = random.choice(item["relation"][0]["text"])
        relation_2 = random.choice(item["relation"][1]["text"])
        if item["subject"]["name"] is None or random.uniform(0., 1.) < p:
          question_text = (
              random.choice(item["subject"]["mentions"])["text"] + " . " +
              relation_1 + " . " + relation_2)
        else:
          question_text = (
              item["subject"]["name"] + " . " + relation_1 + " . " + relation_2)
        answer_mention = item["object"]["global_mention"]
        answer_entity = item["object"]["ent_id"]
        bridge_mention = item["bridge"]["global_mention_1"]
        bridge_entity = [item["bridge"]["ent_id"]]

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=[item["subject"]["wikidata_id"]],
            relations=[relation_1, relation_2],
            answer_mention=[answer_mention],
            answer_entity=[answer_entity],
            bridge_mention=[bridge_mention],
            bridge_entity=[bridge_entity])
        examples.append(example)

    return examples


class ThreeHopDataset:
  """Reads a dataset of three-hop queries and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file, p=subject_mention_probability)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=True)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.FixedLenFeature([], tf.int64),
        "rel_input_ids_0": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_mask_0": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_ids_1": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_mask_1": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_ids_2": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "rel_input_mask_2": tf.FixedLenFeature([self.max_qry_length], tf.int64),
    }
    if is_training:
      names_to_features["answer_mentions"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["answer_entities"] = tf.FixedLenFeature([], tf.int64)
      names_to_features["bridge_mentions"] = tf.FixedLenFeature([], tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=True,
        names_to_features=names_to_features)

  def read_examples(self, queries_file, p=1.0):
    """Read a json file into a list of Example."""
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for line in tqdm(reader):
        item = json.loads(line.strip())

        qas_id = item["id"]
        relation_1 = random.choice(item["relation"][0]["text"])
        relation_2 = random.choice(item["relation"][1]["text"])
        relation_3 = random.choice(item["relation"][2]["text"])
        if item["subject"]["name"] is None or random.uniform(0., 1.) < p:
          question_text = (
              random.choice(item["subject"]["mentions"])["text"] + " . " +
              relation_1 + " . " + relation_2 + " . " + relation_3)
        else:
          question_text = (
              item["subject"]["name"] + " . " + relation_1 + " . " +
              relation_2 + " . " + relation_3)
        answer_mention = item["object"]["global_mention"]
        answer_entity = item["object"]["ent_id"]
        bridge_mention = item["bridge_0"]["global_mention_1"]
        bridge_entity = [item["bridge_%d" % ii]["ent_id"] for ii in range(2)]

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=[item["subject"]["wikidata_id"]],
            relations=[relation_1, relation_2, relation_3],
            answer_mention=[answer_mention],
            answer_entity=[answer_entity],
            bridge_mention=[bridge_mention],
            bridge_entity=[bridge_entity])
        examples.append(example)

    return examples


class WikiMovieDataset:
  """Reads the wikimovie dataset and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    del subject_mention_probability

    num_entities = len(entity2id)
    del entity2id
    entity2id = {i: i for i in range(num_entities)}

    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training
    self.has_bridge = False
    self.num_bridge = 0

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=self.has_bridge)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.FixedLenFeature([], tf.int64),
    }
    if is_training:
      names_to_features["answer_entities"] = tf.VarLenFeature(tf.int64)
    if is_training and self.has_bridge:
      for ii in range(self.num_bridge):
        names_to_features["bridge_entities_%d" % ii] = tf.VarLenFeature(
            tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=True,
        names_to_features=names_to_features)

  def read_examples(self, queries_file):
    """Read a json file into a list of Example."""
    self.max_qry_answers = 0
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for ii, line in tqdm(enumerate(reader)):
        item = json.loads(line.strip())

        qas_id = str(ii)
        question_text = item["question"]
        answer_entities = [answer["kb_id"] for answer in item["answers"]]
        if item["entities"]:
          subject_entities = item["entities"][0]["kb_id"]
        else:
          subject_entities = 0

        if len(answer_entities) > self.max_qry_answers:
          self.max_qry_answers = len(answer_entities)

        inference_chain = "::".join(item["inference_chains"][0])

        bridge_entities = None
        if len(item["intermediate_entities"]) > 2:
          bridge_entities = [[
              bridge["kb_id"] for bridge in intermediate
          ] for intermediate in item["intermediate_entities"][1:-1]]
          self.has_bridge = True
          self.num_bridge = len(bridge_entities)

        if self.has_bridge:
          assert bridge_entities is not None, (qas_id)

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=[subject_entities],
            answer_entity=answer_entities,
            bridge_entity=bridge_entities,
            inference_chain=inference_chain)
        examples.append(example)
    tf.logging.info("Maximum answers per question = %d", self.max_qry_answers)

    return examples


class HotpotQADataset:
  """Reads the hotpotqa dataset and converts to TFRecords."""

  def __init__(self, in_file, tokenizer, subject_mention_probability,
               max_qry_length, is_training, entity2id, tfrecord_filename):
    """Initialize dataset."""
    del subject_mention_probability

    self.gt_file = in_file
    self.max_qry_length = max_qry_length
    self.is_training = is_training

    # Read examples from JSON file.
    self.examples = self.read_examples(in_file, entity2id)
    self.num_examples = len(self.examples)

    if is_training:
      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(self.examples)

    # Write to TFRecords file.
    writer = FeatureWriter(
        filename=tfrecord_filename,
        is_training=self.is_training,
        has_bridge=False)
    convert_examples_to_features(
        examples=self.examples,
        tokenizer=tokenizer,
        max_query_length=self.max_qry_length,
        entity2id=entity2id,
        output_fn=writer.process_feature)
    writer.close()

    # Create input_fn.
    names_to_features = {
        "qas_ids": tf.FixedLenFeature([], tf.string),
        "qry_input_ids": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_input_mask": tf.FixedLenFeature([self.max_qry_length], tf.int64),
        "qry_entity_id": tf.VarLenFeature(tf.int64),
    }
    if is_training:
      names_to_features["answer_entities"] = tf.VarLenFeature(tf.int64)
    self.input_fn = input_fn_builder(
        input_file=tfrecord_filename,
        is_training=self.is_training,
        drop_remainder=True,
        names_to_features=names_to_features)

  def read_examples(self, queries_file, entity2id):
    """Read a json file into a list of Example."""
    self.max_qry_answers = 0
    num_qrys_without_answer, num_qrys_without_all_answers = 0, 0
    num_qrys_without_entity, num_qrys_without_all_entities = 0, 0
    tf.logging.info("Reading examples from %s", queries_file)
    with tf.gfile.Open(queries_file, "r") as reader:
      examples = []
      for line in tqdm(reader):
        item = json.loads(line.strip())

        qas_id = item["_id"]
        question_text = item["question"]
        answer_entities = []
        for answer in item["supporting_facts"]:
          if answer["kb_id"].lower() in entity2id:
            answer_entities.append(entity2id[answer["kb_id"].lower()])
        if not answer_entities:
          num_qrys_without_answer += 1
          if self.is_training:
            continue
        if len(answer_entities) != len(item["supporting_facts"]):
          num_qrys_without_all_answers += 1
        subject_entities = []
        for entity in item["entities"]:
          if entity["kb_id"].lower() in entity2id:
            subject_entities.append(entity["kb_id"].lower())
        if not subject_entities:
          num_qrys_without_entity += 1
          if self.is_training:
            continue
        if len(subject_entities) != len(item["entities"]):
          num_qrys_without_all_entities += 1

        if len(answer_entities) > self.max_qry_answers:
          self.max_qry_answers = len(answer_entities)

        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            subject_entity=subject_entities,
            answer_entity=answer_entities,
            inference_chain=item["type"])
        examples.append(example)
    tf.logging.info("Number of valid questions = %d", len(examples))
    tf.logging.info("Questions without any answer = %d",
                    num_qrys_without_answer)
    tf.logging.info("Questions without all answers = %d",
                    num_qrys_without_all_answers)
    tf.logging.info("Questions without any entity = %d",
                    num_qrys_without_entity)
    tf.logging.info("Questions without all entities = %d",
                    num_qrys_without_all_entities)
    tf.logging.info("Maximum answers per question = %d", self.max_qry_answers)

    return examples

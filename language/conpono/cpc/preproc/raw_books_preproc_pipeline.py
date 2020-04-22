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
"""Beam pipeline to convert CC News to shareded TFRecords."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import os
import random
from absl import app
from absl import flags
import apache_beam as beam
from bert import tokenization
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    "input_file", None, "Path to raw input files."
    "Assumes the filenames wiki.{train|valid|test}.raw")
flags.DEFINE_string("output_file", None, "Output TF example file.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_sent_length", 70, "Maximum sequence length.")
flags.DEFINE_integer("max_para_length", 30, "Maximum sequence length.")
flags.DEFINE_integer("random_seed", 12345, "A random seed")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_float("test_size", 0.1,
                   "Size of test set by factor of total dataset.")

FLAGS = flags.FLAGS



def split_line_by_sentences(line):
  return sent_tokenize(line)


def read_file(filename):
  """Read the contents of filename (str) and split into documents by chapter."""

  all_documents = []
  document = []
  with tf.gfile.GFile(filename, "r") as reader:
    for line in reader:
      line = line.strip()
      line = tokenization.convert_to_unicode(line)
      line = line.replace(u"\u2018", "'").replace(u"\u2019", "'")
      sents = split_line_by_sentences(line)
      for sent_line in sents:
        if not sent_line or len(sent_line) < 4:  # Arbitrary min length for line
          continue
        if sent_line.lower()[:7] == "chapter":
          if document:
            all_documents.append(document)
            document = []
        else:
          document.append(sent_line)
        if len(document) == FLAGS.max_para_length:
          all_documents.append(document)
          document = []
    if document:
      all_documents.append(document)

  # Remove small documents
  all_documents = [x for x in all_documents if len(x) >= 8]

  return all_documents


def create_bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def convert_instance_to_tf_example(tokenizer, sent_tokens, max_sent_length,
                                   max_para_length):
  """Convert a list of strings into a tf.Example."""

  input_ids_list = [
      tokenizer.convert_tokens_to_ids(tokens) for tokens in sent_tokens
  ]
  features = collections.OrderedDict()

  # pack or trim sentences to max_sent_length
  # pack paragraph to max_para_length
  sent_tensor = []
  for i in range(max_para_length):
    if i >= len(input_ids_list):
      sent_tensor.append([0] * max_sent_length)
    else:
      padded_ids = np.pad(
          input_ids_list[i], (0, max_sent_length),
          mode="constant")[:max_sent_length]
      sent_tensor.append(padded_ids)
  sent_tensor = np.ravel(np.stack(sent_tensor))
  features["sents"] = create_int_feature(sent_tensor)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example


def preproc_doc(document):
  """Convert document to list of TF Examples for binary order classification.

  Args:
      document: a CCNews article (ie. a list of sentences)

  Returns:
      A list of tfexamples of binary orderings of pairs of sentences in the
      document. The tfexamples are serialized to string to be written directly
      to TFRecord.
  """
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # document = [
  #     tokenization.convert_to_unicode(
  #         unidecode.unidecode(line.decode("utf-8"))) for line in document
  # ]

  sent_tokens = [tokenizer.tokenize(sent) for sent in document if sent]
  sent_tokens = [sent for sent in sent_tokens if len(sent) > 1]
  if len(sent_tokens) < 8:
    return []

  # Convert token lists into ids and add any needed tokens and padding for BERT
  tf_example = convert_instance_to_tf_example(tokenizer, sent_tokens,
                                              FLAGS.max_sent_length,
                                              FLAGS.max_para_length)

  # Serialize TFExample for writing to file.
  tf_examples = [tf_example.SerializeToString()]

  return tf_examples


def ccnews_pipeline():
  """Read Books Corpus filenames and create Beam pipeline."""

  # set a random seed for reproducability
  rng = random.Random(FLAGS.random_seed)

  # BooksCorpus is organized into directories of genre and files of books
  # adventure-all.txt seems to contain all the adventure books in 1 file
  # romance-all.txt is the same. None of the other directories have this,
  # so we will skip it to not double count those books
  file_name_set = set()
  input_files_by_genre = collections.defaultdict(list)
  for path, _, fnames in tf.gfile.Walk(FLAGS.input_file):
    genre = path.split("/")[-1]
    for fname in fnames:
      if fname == "adventure-all.txt" or fname == "romance-all.txt":
        continue
      if fname in file_name_set:
        continue
      file_name_set.add(fname)
      input_files_by_genre[genre].append(path + "/" + fname)

  # Sort genres and iterate in order for reproducability
  train_files, test_files = [], []
  for genre, file_list in sorted(input_files_by_genre.items()):
    rng.shuffle(file_list)
    genre_size = len(file_list)
    test_size = int(FLAGS.test_size * genre_size)
    test_files.extend(file_list[:test_size])
    train_files.extend(file_list[test_size:])
    assert len(file_list[:test_size]) + \
        len(file_list[test_size:]) == len(file_list)

  # make sure there is no test train overlap
  for filename in train_files:
    assert filename not in test_files

  rng.shuffle(train_files)
  rng.shuffle(test_files)

  def pipeline(root):
    """Beam pipeline for converting CCNews files to TF Examples."""
    _ = (
        root | "Create test files" >> beam.Create(test_files)
        | "Read test files" >> beam.FlatMap(read_file)
        | "test Shuffle" >> beam.Reshuffle()
        | "Preproc test docs" >> beam.FlatMap(preproc_doc)
        | "record test Shuffle" >> beam.Reshuffle()
        | "Write to test tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + ".cc_cpc.test.tfrecord", num_shards=50))
    _ = (
        root | "Create train files" >> beam.Create(train_files)
        | "Read train files" >> beam.FlatMap(read_file)
        | "train Shuffle" >> beam.Reshuffle()
        | "Preproc train docs" >> beam.FlatMap(preproc_doc)
        | "record train Shuffle" >> beam.Reshuffle()
        | "Write to train tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + ".cc_cpc.train.tfrecord", num_shards=450))
    return

  return pipeline


def main(_):
  # If using Apache BEAM, execute runner here.

if __name__ == "__main__":
  app.run(main)

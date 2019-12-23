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
"""Beam pipeline to convert BooksCorpus to shareded TFRecords."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import hashlib
import os
import random
from absl import app
from absl import flags
import apache_beam as beam
from bert import tokenization
from language.conpono.create_pretrain_data.preprocessing_utils import convert_instance_to_tf_example
from language.conpono.create_pretrain_data.preprocessing_utils import create_instances_from_document
from language.conpono.create_pretrain_data.preprocessing_utils import create_paragraph_order_from_document
import nltk
from nltk.tokenize import sent_tokenize
import tensorflow as tf


FORMAT_BINARY = "binary"
FORMAT_PARAGRAPH = "paragraph"

flags.DEFINE_string("input_file", None, "Path to raw input files.")
flags.DEFINE_string("output_file", None, "Output TF example file.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_float("test_size", 0.1,
                   "Size of test set by factor of total dataset.")
flags.DEFINE_float("dev_size", 0.1,
                   "Size of dev set by factor of total dataset.")
flags.DEFINE_integer("random_seed", 12345, "A random seed")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_enum(
    "format", FORMAT_PARAGRAPH, [FORMAT_BINARY, FORMAT_PARAGRAPH],
    "Build a dataset of either binary order or paragraph reconstrucition")

FLAGS = flags.FLAGS



def read_file(filename):
  """Read the contents of filename (str) and split into documents by chapter."""

  all_documents = []
  document = []
  with tf.gfile.GFile(filename, "r") as reader:
    for line in reader:
      line = line.strip()
      if not line:
        continue
      if line.lower()[:7] == "chapter":
        if document:
          all_documents.append(document)
          document = []
      else:
        document.append(line)
    if document:
      all_documents.append(document)

  return all_documents


def split_line_by_sentences(line):
  return sent_tokenize(line)


def preproc_doc(document):
  """Convert document to list of TF Examples for binary order classification.

  Args:
      document: a chapter from one book as a list of lines

  Returns:
      A list of tfexamples of binary orderings of pairs of sentences in the
      document. The tfexamples are serialized to string to be written directly
      to TFRecord.
  """

  # Each document is a list of lines
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # set a random seed for reproducability
  # since this function is run in parallel, if we hardcode a seed, all
  # documents will have the same permutations. Instead we use the hash of the
  # first sentence as the seed so it is different for each document and it
  # is still reproducible.
  hash_object = hashlib.md5(document[0])
  rng = random.Random(int(hash_object.hexdigest(), 16) % (10**8))

  # Each document is composed of a list of sentences. We create paragraphs
  # by keeping together sentences on the same line and adding adjacent sentences
  # if there are fewer than 5 to form the paragraph.
  # The utility functions below expect the document to be split by paragraphs.
  list_of_paragraphs = []
  paragraph = []
  for line in document:
    line = tokenization.convert_to_unicode(line)
    line = line.replace(u"\u2018", "'").replace(u"\u2019", "'")
    sents = split_line_by_sentences(line)
    for sent in sents:
      tokens = tokenizer.tokenize(sent)
      if tokens:
        paragraph.append(tokens)
    if len(paragraph) > 5:
      list_of_paragraphs.append(paragraph)
      paragraph = []

  # In case of any empty paragraphs, remove them.
  list_of_paragraphs = [x for x in list_of_paragraphs if x]

  # Convert the list of paragraphs into TrainingInstance object
  # See preprocessing_utils.py for definition
  if FLAGS.format == FORMAT_BINARY:
    instances = create_instances_from_document(list_of_paragraphs,
                                               FLAGS.max_seq_length, rng)
  elif FLAGS.format == FORMAT_PARAGRAPH:
    instances = create_paragraph_order_from_document(list_of_paragraphs,
                                                     FLAGS.max_seq_length, rng)

  # Convert token lists into ids and add any needed tokens and padding for BERT
  tf_examples = [
      convert_instance_to_tf_example(tokenizer, instance,
                                     FLAGS.max_seq_length)[0]
      for instance in instances
  ]

  # Serialize TFExample for writing to file.
  tf_examples = [example.SerializeToString() for example in tf_examples]

  return tf_examples


def books_pipeline():
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
  train_files, dev_files, test_files = [], [], []
  for genre, file_list in sorted(input_files_by_genre.items()):
    rng.shuffle(file_list)
    genre_size = len(file_list)
    test_size = int(FLAGS.test_size * genre_size)
    dev_size = int(FLAGS.dev_size * genre_size)
    test_files.extend(file_list[:test_size])
    dev_files.extend(file_list[test_size:test_size + dev_size])
    train_files.extend(file_list[test_size + dev_size:])
    assert len(file_list[:test_size]) + \
        len(file_list[test_size:test_size+dev_size]) + \
        len(file_list[test_size+dev_size:]) == len(file_list)

  # make sure there is no test train overlap
  for filename in train_files:
    assert filename not in test_files
    assert filename not in dev_files
  for filename in dev_files:
    assert filename not in test_files

  rng.shuffle(train_files)
  rng.shuffle(dev_files)
  rng.shuffle(test_files)

  def pipeline(root):
    """Beam pipeline for converting Books Corpus files to TF Examples."""
    _ = (
        root | "Create test files" >> beam.Create(test_files)
        | "Read test files" >> beam.FlatMap(read_file)
        | "test Shuffle" >> beam.Reshuffle()
        | "Preproc test docs" >> beam.FlatMap(preproc_doc)
        | "record test Shuffle" >> beam.Reshuffle()
        | "Write to test tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + "." + FLAGS.format + ".test.tfrecord",
            num_shards=100))
    _ = (
        root | "Create dev files" >> beam.Create(dev_files)
        | "Read dev files" >> beam.FlatMap(read_file)
        | "dev Shuffle" >> beam.Reshuffle()
        | "Preproc dev docs" >> beam.FlatMap(preproc_doc)
        | "record dev Shuffle" >> beam.Reshuffle()
        | "Write to dev tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + "." + FLAGS.format + ".dev.tfrecord",
            num_shards=100))
    _ = (
        root | "Create train files" >> beam.Create(train_files)
        | "Read train files" >> beam.FlatMap(read_file)
        | "train Shuffle" >> beam.Reshuffle()
        | "Preproc train docs" >> beam.FlatMap(preproc_doc)
        | "record train Shuffle" >> beam.Reshuffle()
        | "Write to train tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + "." + FLAGS.format + ".train.tfrecord",
            num_shards=500))
    return

  return pipeline


def main(_):
  # If using Apache BEAM, execute runner here.

if __name__ == "__main__":
  app.run(main)

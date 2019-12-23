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
"""Beam pipeline to convert WikiText103 to shareded TFRecords."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import hashlib
import random
from absl import app
from absl import flags
import apache_beam as beam
from bert import tokenization
from language.conpono.cpc.preproc.preprocessing_utils import convert_instance_to_tf_example
from language.conpono.cpc.preproc.preprocessing_utils import create_instances_from_document
from language.conpono.cpc.preproc.preprocessing_utils import LONG_CTX
from language.conpono.cpc.preproc.preprocessing_utils import ONE_SENT_CTX
import tensorflow as tf


flags.DEFINE_string(
    "input_file", None, "Path to raw input files."
    "Assumes the filenames wiki.{train|valid|test}.raw")
flags.DEFINE_string("output_file", None, "Output TF example file.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("random_seed", 12345, "A random seed")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_enum("format", LONG_CTX, [LONG_CTX, ONE_SENT_CTX],
                  "format of preprocessed data")

FLAGS = flags.FLAGS


def read_file(filename):
  """Read the contents of filename (str) and split into documents by chapter."""

  all_documents = []
  # Input file format:
  # See internal docs for an example
  # Documents are contiguously stored. Headers to sections are denoted by
  # "=" on each side. The number of "=" indicates the depth. So the title for
  # United States is "= United States =". "= = = Population = = =" indicates
  # a section 3 levels deep (United States -> Demographics -> Population).
  # There are blank lines between sections.
  # Each line is a paragraph. Periods that are sentence delimiters have a space
  # on each side. Periods in abbreviations do not have spaces.

  # For parallel processing, we first split the single file into documents.
  # Each document is a list of lines.
  with tf.gfile.GFile(filename, "r") as reader:
    for line in reader:
      line = line.strip()
      if not line:
        continue

      # Headers like (" = John Doe = ") are document delimiters
      if line.startswith("= ") and line[2] != "=":
        all_documents.append([])
      # Skip lines that are headers
      if line[0] == "=":
        continue
      all_documents[-1].append(line)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]

  # for each document grab 3 random docs
  all_docs_with_rands = []
  for doc in all_documents:
    all_docs_with_rands.append((doc, random.sample(all_documents, 3)))

  return all_docs_with_rands


def split_line_by_sentences(line):
  # Put trailing period back but not on the last element
  # because that usually leads to double periods.
  sentences = [l + "." for l in line.split(" . ")]
  sentences[-1] = sentences[-1][:-1]
  return sentences


def preproc_doc(document_tuple):
  """Convert document to list of TF Examples for binary order classification.

  Args:
      document_tuple: a wikipedia article as a list of lines and a tuple of 3
        random docs

  Returns:
      A list of tfexamples of binary orderings of pairs of sentences in the
      document. The tfexamples are serialized to string to be written directly
      to TFRecord.
  """
  document, other_docs = document_tuple
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # set a random seed for reproducability
  hash_object = hashlib.md5(document[0])
  rng = random.Random(int(hash_object.hexdigest(), 16) % (10**8))

  # Each document is composed of a list of text lines. Each text line is a
  # paragraph. We split the line into sentences but keep the paragraph grouping.
  # The utility functions below expect the document to be split by paragraphs.
  list_of_paragraphs = []
  special_chars = {"@,@": ",", "@.@": ".", "@-@": "-"}
  for line in document:
    line = tokenization.convert_to_unicode(line)
    line = line.replace(u"\u2018", "'").replace(u"\u2019", "'")
    sents = split_line_by_sentences(line)

    # These special chars in the wiki dataset should be replaced with the
    # raw characters
    for sp_char in special_chars.items():
      sents = [sent.replace(sp_char[0], sp_char[1]) for sent in sents]

    if len(sents) < (5 if FLAGS.format == ONE_SENT_CTX else 8):
      continue
    sent_tokens = [tokenizer.tokenize(sent) for sent in sents if sent]
    sent_tokens = [sent for sent in sent_tokens if len(sent) > 1]
    if len(sent_tokens) < (5 if FLAGS.format == ONE_SENT_CTX else 8):
      continue
    list_of_paragraphs.append(sent_tokens)

  # In case of any empty paragraphs, remove them.
  list_of_paragraphs = [x for x in list_of_paragraphs if x]

  # Redo above with otherdocs
  list_of_para_other_docs = []
  for doc in other_docs:
    for line in doc:
      line = tokenization.convert_to_unicode(line)
      line = line.replace(u"\u2018", "'").replace(u"\u2019", "'")
      sents = split_line_by_sentences(line)
      sent_tokens = [tokenizer.tokenize(sent) for sent in sents if sent]
      list_of_para_other_docs.append(sent_tokens)

  # In case of any empty paragraphs, remove them.
  list_of_para_other_docs = [x for x in list_of_para_other_docs if x]

  # we need to be able to sample from multiple paragraphs
  if len(list_of_para_other_docs) == 1 or \
      sum([len(x) for x in list_of_para_other_docs]) < 35:
    return []

  # Convert the list of paragraphs into TrainingInstance object
  # See preprocessing_utils.py for definition
  instances = create_instances_from_document(list_of_paragraphs,
                                             FLAGS.max_seq_length, rng,
                                             list_of_para_other_docs,
                                             FLAGS.format)

  # Convert token lists into ids and add any needed tokens and padding for BERT
  tf_examples = [
      convert_instance_to_tf_example(tokenizer, instance,
                                     FLAGS.max_seq_length)[0]
      for instance in instances
  ]

  # Serialize TFExample for writing to file.
  tf_examples = [example.SerializeToString() for example in tf_examples]

  return tf_examples


def wiki_pipeline():
  """Read WikiText103 filenames and create Beam pipeline."""

  train_files = FLAGS.input_file + "/wiki.train.raw"
  dev_files = FLAGS.input_file + "/wiki.valid.raw"
  test_files = FLAGS.input_file + "/wiki.test.raw"

  def pipeline(root):
    """Beam pipeline for converting WikiText103 files to TF Examples."""
    _ = (
        root | "Create test files" >> beam.Create([test_files])
        | "Read test files" >> beam.FlatMap(read_file)
        | "test Shuffle" >> beam.Reshuffle()
        | "Preproc test docs" >> beam.FlatMap(preproc_doc)
        | "record test Shuffle" >> beam.Reshuffle()
        | "Write to test tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + ".cpc." + FLAGS.format + ".test.tfrecord",
            num_shards=10))
    _ = (
        root | "Create dev files" >> beam.Create([dev_files])
        | "Read dev files" >> beam.FlatMap(read_file)
        | "dev Shuffle" >> beam.Reshuffle()
        | "Preproc dev docs" >> beam.FlatMap(preproc_doc)
        | "record dev Shuffle" >> beam.Reshuffle()
        | "Write to dev tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + ".cpc." + FLAGS.format + ".dev.tfrecord",
            num_shards=10))
    _ = (
        root | "Create train files" >> beam.Create([train_files])
        | "Read train files" >> beam.FlatMap(read_file)
        | "train Shuffle" >> beam.Reshuffle()
        | "Preproc train docs" >> beam.FlatMap(preproc_doc)
        | "record train Shuffle" >> beam.Reshuffle()
        | "Write to train tfrecord" >> beam.io.WriteToTFRecord(
            FLAGS.output_file + ".cpc." + FLAGS.format + ".train.tfrecord",
            num_shards=100))
    return

  return pipeline


def main(_):
  # If using Apache BEAM, execute runner here.


if __name__ == "__main__":
  app.run(main)

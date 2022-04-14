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
"""Preprocess a QA file into TFRecords for training dualencoder models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import random

from absl import app
from absl import flags
from bert import tokenization

import six
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("input_file", None,
                    "The input JSONL file (gzipped) to read questions from.")

flags.DEFINE_string("output_file", None,
                    "The output TFRecord file to store examples to.")

flags.DEFINE_string("feature_file", None,
                    "The output TFRecord file to store examples to.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 192,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 64,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 48,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool(
    "version_2_with_negative", True,
    "If true, the input contain some that do not have an answer.")


class Example(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


def read_examples(input_file):
  """Read a SQuAD-like json file into a list of Examples."""

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  reader = tf.gfile.Open(input_file, "rb")
  if input_file.endswith(".gz"):
    reader = gzip.GzipFile(fileobj=reader)
  next(reader)
  examples = []
  for line in tqdm(reader):
    item = json.loads(line.strip())
    paragraph_text = item["context"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
      if is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          doc_tokens.append(c)
        else:
          doc_tokens[-1] += c
        prev_is_whitespace = False
      char_to_word_offset.append(len(doc_tokens) - 1)

    for qa in item["qas"]:
      qas_id = qa["qid"]
      question_text = qa["question"]

      start_position = None
      end_position = None
      orig_answer_text = None
      is_impossible = False

      start_position = -1
      end_position = -1
      orig_answer_text = ""
      if FLAGS.version_2_with_negative:
        is_impossible = qa["is_impossible"]
      if not is_impossible:
        answer_offset = qa["detected_answers"][0]["char_spans"][0][0]
        answer_end = qa["detected_answers"][0]["char_spans"][0][1]
        answer_length = answer_end - answer_offset + 1
        orig_answer_text = item["context"][answer_offset:answer_end + 1]
        start_position = char_to_word_offset[answer_offset]
        end_position = char_to_word_offset[answer_offset + answer_length - 1]
        # Only add answers where the text can be exactly recovered from the
        # document. If this CAN'T happen it's likely due to weird Unicode
        # stuff so we will just skip the example.
        #
        # Note that this means for training mode, every example is NOT
        # guaranteed to be preserved.
        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
        cleaned_answer_text = " ".join(
            tokenization.whitespace_tokenize(orig_answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
          tf.logging.warning("Example %d", len(examples))
          tf.logging.warning(json.dumps(item, indent=2))
          tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                             actual_text, cleaned_answer_text)
          continue

      example = Example(
          qas_id=qas_id,
          question_text=question_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
      examples.append(example)
  reader.close()

  return examples


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename):
    self.filename = filename
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
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["qas_id"] = create_bytes_feature(feature.qas_id)
    features["doc_input_ids"] = create_int_feature(feature.doc_input_ids)
    features["doc_input_mask"] = create_int_feature(feature.doc_input_mask)
    features["doc_segment_ids"] = create_int_feature(feature.doc_segment_ids)
    features["qry_input_ids"] = create_int_feature(feature.qry_input_ids)
    features["qry_input_mask"] = create_int_feature(feature.qry_input_mask)
    features["qry_segment_ids"] = create_int_feature(feature.qry_segment_ids)

    features["start_positions"] = create_int_feature([feature.start_position])
    features["end_positions"] = create_int_feature([feature.end_position])
    impossible = 0
    if feature.is_impossible:
      impossible = 1
    features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               qas_id,
               example_index,
               doc_span_index,
               doc_tokens,
               doc_token_to_orig_map,
               doc_token_is_max_context,
               doc_input_ids,
               doc_input_mask,
               doc_segment_ids,
               qry_tokens,
               qry_input_ids,
               qry_input_mask,
               qry_segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.qas_id = qas_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.doc_tokens = doc_tokens
    self.doc_token_to_orig_map = doc_token_to_orig_map
    self.doc_token_is_max_context = doc_token_is_max_context
    self.doc_input_ids = doc_input_ids
    self.doc_input_mask = doc_input_mask
    self.doc_segment_ids = doc_segment_ids
    self.qry_tokens = qry_tokens
    self.qry_input_ids = qry_input_ids
    self.qry_input_mask = qry_input_mask
    self.qry_segment_ids = qry_segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def convert_examples_to_features(examples, tokenizer, max_doc_length,
                                 doc_stride, max_query_length, output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in tqdm(enumerate(examples)):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length - 2:
      query_tokens = query_tokens[0:max_query_length - 2]  # -2 for [CLS], [SEP]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_doc_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      doc_tokens, qry_tokens = [], []
      doc_token_to_orig_map = {}
      doc_token_is_max_context = {}
      doc_segment_ids, qry_segment_ids = [], []

      # Question
      qry_tokens.append("[CLS]")
      qry_segment_ids.append(0)
      for token in query_tokens:
        qry_tokens.append(token)
        qry_segment_ids.append(0)
      qry_tokens.append("[SEP]")
      qry_segment_ids.append(0)

      # Document
      doc_tokens.append("[CLS]")
      doc_segment_ids.append(1)
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        doc_token_to_orig_map[len(
            doc_tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        doc_token_is_max_context[len(doc_tokens)] = is_max_context
        doc_tokens.append(all_doc_tokens[split_token_index])
        doc_segment_ids.append(1)
      doc_tokens.append("[SEP]")
      doc_segment_ids.append(1)

      doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
      qry_input_ids = tokenizer.convert_tokens_to_ids(qry_tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      doc_input_mask = [1] * len(doc_input_ids)
      qry_input_mask = [1] * len(qry_input_ids)

      # Zero-pad up to the sequence length.
      while len(doc_input_ids) < max_doc_length:
        doc_input_ids.append(0)
        doc_input_mask.append(0)
        doc_segment_ids.append(0)
      while len(qry_input_ids) < max_query_length:
        qry_input_ids.append(0)
        qry_input_mask.append(0)
        qry_segment_ids.append(0)

      assert len(doc_input_ids) == max_doc_length
      assert len(doc_input_mask) == max_doc_length
      assert len(doc_segment_ids) == max_doc_length
      assert len(qry_input_ids) == max_query_length
      assert len(qry_input_mask) == max_query_length
      assert len(qry_segment_ids) == max_query_length

      start_position = None
      end_position = None
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      doc_offset = 1
      if not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset
      if example.is_impossible:
        start_position = 0
        end_position = 0

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s", unique_id)
        tf.logging.info("example_index: %s", example_index)
        tf.logging.info("doc_span_index: %s", doc_span_index)
        tf.logging.info(
            "doc_tokens: %s",
            " ".join([tokenization.printable_text(x) for x in doc_tokens]))
        tf.logging.info(
            "qry_tokens: %s",
            " ".join([tokenization.printable_text(x) for x in qry_tokens]))
        tf.logging.info(
            "doc_token_to_orig_map: %s", " ".join([
                "%d:%d" % (x, y)
                for (x, y) in six.iteritems(doc_token_to_orig_map)
            ]))
        tf.logging.info(
            "doc_token_is_max_context: %s", " ".join([
                "%d:%s" % (x, y)
                for (x, y) in six.iteritems(doc_token_is_max_context)
            ]))
        tf.logging.info("doc_input_ids: %s",
                        " ".join([str(x) for x in doc_input_ids]))
        tf.logging.info("doc_input_mask: %s",
                        " ".join([str(x) for x in doc_input_mask]))
        tf.logging.info("doc_segment_ids: %s",
                        " ".join([str(x) for x in doc_segment_ids]))
        tf.logging.info("qry_input_ids: %s",
                        " ".join([str(x) for x in qry_input_ids]))
        tf.logging.info("qry_input_mask: %s",
                        " ".join([str(x) for x in qry_input_mask]))
        tf.logging.info("qry_segment_ids: %s",
                        " ".join([str(x) for x in qry_segment_ids]))
        if example.is_impossible:
          tf.logging.info("impossible example")
        if not example.is_impossible:
          answer_text = " ".join(doc_tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d", start_position)
          tf.logging.info("end_position: %d", end_position)
          tf.logging.info("answer: %s",
                          tokenization.printable_text(answer_text))

      feature = InputFeatures(
          unique_id=unique_id,
          qas_id=example.qas_id.encode("utf-8"),
          example_index=example_index,
          doc_span_index=doc_span_index,
          doc_tokens=doc_tokens,
          doc_token_to_orig_map=doc_token_to_orig_map,
          doc_token_is_max_context=doc_token_is_max_context,
          doc_input_ids=doc_input_ids,
          doc_input_mask=doc_input_mask,
          doc_segment_ids=doc_segment_ids,
          qry_tokens=qry_tokens,
          qry_input_ids=qry_input_ids,
          qry_input_mask=qry_input_mask,
          qry_segment_ids=qry_segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=example.is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1


def main(_):
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  examples = read_examples(input_file=FLAGS.input_file)

  # Pre-shuffle the input to avoid having to make a very large shuffle
  # buffer in in the `input_fn`.
  rng = random.Random(12345)
  rng.shuffle(examples)

  # We write to a temporary file to avoid storing very large
  # constant tensors in memory.
  writer = FeatureWriter(filename=FLAGS.output_file)
  features = []

  def append_feature(feature):
    features.append(feature)
    writer.process_feature(feature)

  convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_doc_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      output_fn=append_feature)
  writer.close()
  tf.logging.info("%d original examples read.", len(examples))
  tf.logging.info("%d split records written.", writer.num_features)

  if FLAGS.feature_file is not None:
    json.dump([[vars(ee) for ee in examples], [vars(ff) for ff in features]],
              tf.gfile.Open(FLAGS.feature_file, "w"))


if __name__ == "__main__":
  app.run(main)

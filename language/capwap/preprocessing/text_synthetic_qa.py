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
"""Convert text data to the format expected by the question generation model.

This data is used as part of line 5 of Alg. 1 (LEARN, input to QGEN).

It augments the out-of-domain COCO data (with Wikipedia text).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os

from absl import app
from absl import flags
import apache_beam as beam
from language.capwap.utils import io_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf


DATA_DIR = os.getenv("CAPWAP_DATA", "data")

flags.DEFINE_integer("query_length", 20, "Maximum query length.")

flags.DEFINE_integer("max_length", 128, "Maximum input length.")

flags.DEFINE_float("unk_rate", 0.1, "Maximum [UNK] token rate.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT directory.")

flags.DEFINE_string("input_path", None,
                    "Input path (or glob pattern) to (sharded) text files.")

flags.DEFINE_string("output_path",
                    os.path.join(DATA_DIR, "WIKI/processed/qgen/contexts"),
                    "Output path.")
FLAGS = flags.FLAGS

# Extractive
ANSWER_TYPE = 3

RCInputs = collections.namedtuple("RCInputs",
                                  ["input_ids", "input_mask", "segment_ids"])


class GetTokens(beam.PTransform):
  """Convert input text lines to tokens."""

  def __init__(self, params):
    # Lazily init vocab.
    self._params = params
    self._vocab = None

    # Create a counter for the number of items that we've mapped.
    self._example_counter = beam.metrics.Metrics.counter("non-empty", "counter")

  def _maybe_actually_init(self):
    """Lazily create example converter."""
    if self._vocab is None:
      self._vocab = text_utils.Vocab.load(self._params["vocab_path"])

  def transform(self, value):
    self._maybe_actually_init()
    tokens = self._vocab.tokenize(value)
    if not tokens:
      return ""
    num_unk = len([t for t in tokens if t == self._vocab.UNK])
    if num_unk / len(tokens) > self._params["unk_rate"]:
      return ""
    self._example_counter.inc()
    return " ".join(tokens)

  def expand(self, collection):
    return collection | "Transform" >> beam.Map(self.transform)


class GetExamples(beam.PTransform):
  """Convert input tokens to RC examples."""

  def __init__(self, params):
    # Lazily init vocab.
    self._params = params
    self._session = None

    # Create a counter for the number of items that we've mapped.
    self._example_counter = beam.metrics.Metrics.counter("example", "counter")

  def _maybe_actually_init(self):
    """Lazily create example converter."""
    if self._session is None:
      self._vocab = text_utils.Vocab.load(self._params["vocab_path"])
      self._graph = tf.Graph()
      with self._graph.as_default():
        # Placeholder for input lines of tokenized text.
        self._text = tf.placeholder(tf.string, [])

        # Truncate text.
        tokens = tf.string_split([self._text]).values
        length = self._params["max_length"] - self._params["query_length"] - 3
        tokens = tokens[:length]

        # Create full input together with empty question.
        question = ["[PAD]"] * self._params["query_length"]
        inputs = tf.concat([[self._vocab.CLS], question, [self._vocab.SEP],
                            tokens, [self._vocab.SEP]],
                           axis=0)

        # Convert to ids.
        lookup_table = self._vocab.get_string_lookup_table()
        input_ids = tf.cast(lookup_table.lookup(inputs), tf.int32)
        input_mask = tf.ones_like(input_ids)
        segment_ids = tf.concat([[0] * (self._params["query_length"] + 2),
                                 tf.fill(tf.shape(tokens), 1), [1]],
                                axis=0)

        # Pad to final length.
        pad = [[0, self._params["max_length"] - tf.size(input_ids)]]
        input_ids = tf.pad(input_ids, pad)
        input_mask = tf.pad(input_mask, pad)
        segment_ids = tf.pad(segment_ids, pad)
        self._rc_inputs = RCInputs(input_ids, input_mask, segment_ids)

        # Initialize session.
        self._session = tf.Session()
        self._session.run(tf.initialize_all_tables())

  def transform(self, value):
    self._maybe_actually_init()
    inputs = self._session.run(self._rc_inputs, {self._text: value})
    features = dict(
        unique_ids=io_utils.int64_feature(0),
        input_ids=io_utils.int64_feature_list(inputs.input_ids),
        input_mask=io_utils.int64_feature_list(inputs.input_mask),
        segment_ids=io_utils.int64_feature_list(inputs.segment_ids),
        start_positions=io_utils.int64_feature(0),
        end_positions=io_utils.int64_feature(0),
        answer_types=io_utils.int64_feature(ANSWER_TYPE))
    example = tf.train.Example(features=tf.train.Features(feature=features))

    self._example_counter.inc()
    return example

  def expand(self, collection):
    return collection | "Transform" >> beam.Map(self.transform)


def pipeline(root, input_path, output_path, params):
  _ = (
      root
      | "Read" >> beam.io.ReadFromText(input_path)
      | GetTokens(params)
      | "KeepNonEmpty" >> beam.Filter(len)
      | GetExamples(params)
      | "Write" >> beam.io.tfrecordio.WriteToTFRecord(
          output_path, coder=beam.coders.ProtoCoder(tf.train.Example)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  params = dict(
      max_length=FLAGS.max_length,
      query_length=FLAGS.query_length,
      unk_rate=FLAGS.unk_rate,
      vocab_path=FLAGS.vocab_path,
  )
  # Implement Apache Beam logic appropriate for your setup here in order
  # to run the pipeline on your cluster.


if __name__ == "__main__":
  flags.mark_flag_as_required("input_path")
  tf.disable_v2_behavior()
  app.run(main)

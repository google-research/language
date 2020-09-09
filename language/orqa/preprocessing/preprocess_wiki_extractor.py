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
r"""Create a database of text blocks.

Each input file assumes lines with the following JSON format:
```
{
  "title": "Document Tile",
  "text": "This is a full document."
}
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from html import parser
import multiprocessing
import os
import random
import tempfile

from absl import app
from absl import flags
from language.orqa.preprocessing import wiki_preprocessor
from language.orqa.utils import bert_utils
import nltk
import tensorflow.compat.v1 as tf

flags.DEFINE_string("bert_hub_module_path",
                    "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                    "Path to the BERT TF-Hub module.")
flags.DEFINE_integer("max_block_length", 288, "Maximum block length.")
flags.DEFINE_string("input_pattern", None, "Path to input data")
flags.DEFINE_string("output_dir", None, "Path to output records.")
flags.DEFINE_integer("num_threads", 12, "Number of threads.")

FLAGS = flags.FLAGS


def get_sentence_splitter():
  temp_dir = tempfile.mkdtemp()
  nltk.download("punkt", download_dir=temp_dir)
  return nltk.data.load(
      os.path.join(temp_dir, "tokenizers/punkt/english.pickle"))


def create_block_info(input_path, preprocessor):
  """Create block info."""
  results = []
  html_parser = parser.HTMLParser()
  with tf.io.gfile.GFile(input_path) as input_file:
    for line in input_file:
      results.extend(
          wiki_preprocessor.example_from_json_line(line, html_parser,
                                                   preprocessor))
  return results


def main(_):
  pool = multiprocessing.Pool(FLAGS.num_threads)
  tf.logging.info("Using hub module %s", FLAGS.bert_hub_module_path)
  tokenizer = bert_utils.get_tokenizer(FLAGS.bert_hub_module_path)
  preprocessor = wiki_preprocessor.Preprocessor(get_sentence_splitter(),
                                                FLAGS.max_block_length,
                                                tokenizer)
  mapper = functools.partial(create_block_info, preprocessor=preprocessor)
  block_count = 0
  input_paths = tf.io.gfile.glob(FLAGS.input_pattern)
  random.shuffle(input_paths)
  tf.logging.info("Processing %d input files.", len(input_paths))

  tf.io.gfile.makedirs(FLAGS.output_dir)
  blocks_path = os.path.join(FLAGS.output_dir, "blocks.tfr")
  examples_path = os.path.join(FLAGS.output_dir, "examples.tfr")
  titles_path = os.path.join(FLAGS.output_dir, "titles.tfr")

  with tf.python_io.TFRecordWriter(blocks_path) as blocks_writer:
    with tf.python_io.TFRecordWriter(examples_path) as examples_writer:
      with tf.python_io.TFRecordWriter(titles_path) as titles_writer:
        for block_info in pool.imap_unordered(mapper, input_paths):
          for title, block, examples in block_info:
            blocks_writer.write(block.encode("utf-8"))
            examples_writer.write(examples)
            titles_writer.write(title.encode("utf-8"))
            block_count += 1
            if block_count % 10000 == 0:
              tf.logging.info("Wrote %d blocks.", block_count)
  tf.logging.info("Wrote %d blocks in total.", block_count)

if __name__ == "__main__":
  app.run(main)

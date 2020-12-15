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
r"""Generate tf records from passages/query json file for inference.

python -m language.multivec.utils.data_processor.data_processor \
  --bert_hub_module_path=${HUB_DIR} \
  --max_seq_length=260 \
  --input_pattern=${INPUT_DIR}/passages*.json \
  --output_path=${OUTPUT_DIR}/passage.tfr \
  --num_threads=12 \
"""

import functools
import json
import multiprocessing
import os

from absl import app
from absl import flags
from bert import tokenization
import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

flags.DEFINE_string("bert_hub_module_path", None,
                    "Path to the BERT TF-Hub module.")
flags.DEFINE_integer("max_seq_length", 288, "Maximum block length.")
flags.DEFINE_string("input_pattern", None, "Path to input data")
flags.DEFINE_string("output_path", None, "Path to output records.")
flags.DEFINE_integer("num_threads", 12, "Number of threads.")

FLAGS = flags.FLAGS


def add_int64_feature(key, values, example):
  example.features.feature[key].int64_list.value.extend(values)


class Preprocessor(object):
  """Preprocessor."""

  def __init__(self, max_seq_length, tokenizer):
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length
    tf.logging.info("Max sequence length {}".format(self._max_seq_length))

  def create_example(self, key, text):
    """Create example."""
    tokens = ["[CLS]"]
    tokens.extend(self._tokenizer.tokenize(text))
    if len(tokens) > self._max_seq_length - 1:
      tokens = tokens[:self._max_seq_length - 1]
    tokens.append("[SEP]")
    inputs_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    example = tf.train.Example()
    add_int64_feature("input_ids", inputs_ids, example)
    add_int64_feature("key", [int(key)], example)
    return example.SerializeToString()

  def example_from_json_line(self, key, text):
    if not isinstance(key, six.text_type):
      key = int(key.decode("utf-8"))
    if not isinstance(text, six.text_type):
      text = text.decode("utf-8")
    return self.create_example(key, text)


def create_block_info(input_path, preprocessor):
  """Create block info."""
  results = []
  with tf.io.gfile.GFile(input_path) as fid:
    input_file = fid.read()
    data = json.loads(input_file)
    for key in data:
      text = data[key]
      results.append(preprocessor.example_from_json_line(key, text))
  return results


def get_tokenization_info(module_handle):
  with tf.Graph().as_default():
    bert_module = hub.Module(module_handle)
    with tf.Session() as sess:
      return sess.run(bert_module(signature="tokenization_info", as_dict=True))


def get_tokenizer(module_handle):
  tokenization_info = get_tokenization_info(module_handle)
  return tokenization.FullTokenizer(
      vocab_file=tokenization_info["vocab_file"],
      do_lower_case=tokenization_info["do_lower_case"])


def main(_):
  pool = multiprocessing.Pool(FLAGS.num_threads)
  tf.logging.info("Using hub module %s", FLAGS.bert_hub_module_path)
  tokenizer = get_tokenizer(FLAGS.bert_hub_module_path)
  preprocessor = Preprocessor(FLAGS.max_seq_length, tokenizer)
  mapper = functools.partial(create_block_info, preprocessor=preprocessor)
  block_count = 0
  input_paths = tf.io.gfile.glob(FLAGS.input_pattern)
  tf.logging.info("Processing %d input files.", len(input_paths))
  output_dir = os.path.dirname(FLAGS.output_path)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)

  with tf.python_io.TFRecordWriter(FLAGS.output_path) as examples_writer:
    for examples in pool.imap_unordered(mapper, input_paths):
      for example in examples:
        examples_writer.write(example)
        block_count += 1
        if block_count % 10000 == 0:
          tf.logging.info("Wrote %d blocks.", block_count)
  tf.logging.info("Wrote %d blocks in total.", block_count)


if __name__ == "__main__":
  app.run(main)

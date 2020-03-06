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
# Lint as: python3
"""Convert NQ data."""
import gzip
import json
import multiprocessing
import os
import random


from absl import app
from absl import flags

import bs4
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("input_pattern", None, "Input path.")
flags.DEFINE_string("output_path", None, "Output path")
flags.DEFINE_integer("max_threads", 50, "Maximum workers in the pool.")
flags.DEFINE_integer("max_tokens", 5, "Maximum tokens in each short answer.")
flags.DEFINE_boolean("fork_workers", True, "Fork workers for more parallelism.")


def _convert_qa_pairs(
    input_path):
  """Generate TF examples."""
  tf.logging.info("Converting examples in %s.", input_path)
  pairs = []
  with gzip.open(input_path) as input_file:
    for line in input_file:
      json_example = json.loads(line)
      question_text = json_example["question_text"]

      # Convert to bytes so that we can index by byte offsets from the data.
      document_html = json_example["document_html"].encode("utf-8")

      answer_texts = set()
      for annotation in json_example["annotations"]:
        for sa in annotation["short_answers"]:
          if sa["end_token"] - sa["start_token"] <= FLAGS.max_tokens:
            raw_html = document_html[sa["start_byte"]:sa["end_byte"]]
            answer_texts.add(bs4.BeautifulSoup(raw_html, "lxml").text)
      if answer_texts:
        pairs.append(dict(
            question=question_text,
            answer=list(answer_texts)))
  tf.logging.info("Done converting examples from %s", input_path)
  return pairs


def main(_):
  input_paths = tf.gfile.Glob(FLAGS.input_pattern)
  tf.logging.info("Converting input %d files: %s", len(input_paths),
                  str(input_paths))
  num_threads = min(FLAGS.max_threads, len(input_paths))
  if FLAGS.fork_workers:
    pool = multiprocessing.Pool(num_threads)
  else:
    pool = multiprocessing.dummy.Pool(num_threads)
  sharded_pairs = pool.map(_convert_qa_pairs, input_paths)

  # pylint: disable=g-complex-comprehension
  sharded_pairs = [p for l in sharded_pairs for p in l]
  # pylint: enable=g-complex-comprehension

  random.shuffle(sharded_pairs)
  tf.logging.info("Found %d pairs.", len(sharded_pairs))
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_path))
  with tf.io.gfile.GFile(FLAGS.output_path, "w") as output_file:
    for p in sharded_pairs:
      output_file.write(json.dumps(p))
      output_file.write("\n")

if __name__ == "__main__":
  app.run(main)

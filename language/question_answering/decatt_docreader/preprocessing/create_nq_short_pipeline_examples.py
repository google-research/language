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
"""Create TF examples with gold contexts for examples with short answers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import multiprocessing
import os

from absl import app
from absl import flags

import six
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("input_pattern", None, "Path to input jsonl data.")
flags.DEFINE_string("output_dir", None, "Path to output tf.Examples.")
flags.DEFINE_integer("max_threads", 50, "Maximum workers in the pool.")
flags.DEFINE_boolean("fork_workers", True, "Fork workers for more parallelism.")


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _string_feature(value):
  return _bytes_feature(value.lower().encode("utf8"))


def _generate_tf_examples(input_file):
  """Generate TF examples."""
  for line in input_file:
    if not isinstance(line, six.text_type):
      line = line.decode("utf-8")
    json_example = json.loads(line)
    question_tokens_feature = (
        _string_feature(" ".join(json_example["question_tokens"])))
    for annotation in json_example["annotations"]:
      short_answers = annotation["short_answers"]
      if short_answers:
        # Only use the first short answer during training.
        short_answer = short_answers[0]
        long_answer = annotation["long_answer"]
        assert short_answer["start_token"] >= long_answer["start_token"]
        assert short_answer["end_token"] <= long_answer["end_token"]
        long_answer_tokens = json_example["document_tokens"][
            long_answer["start_token"]:long_answer["end_token"]]
        features = {}
        features["question"] = question_tokens_feature
        features["context"] = _string_feature(" ".join(
            t["token"] for t in long_answer_tokens))

        # All span offsets are inclusive-exclusive, but it's more convenient
        # to use inclusive-inclusive offsets for modeling.
        features["answer_start"] = _int64_feature(short_answer["start_token"] -
                                                  long_answer["start_token"])
        features["answer_end"] = _int64_feature(short_answer["end_token"] -
                                                long_answer["start_token"] - 1)
        yield tf.train.Example(features=tf.train.Features(feature=features))


def _create_short_answer_examples(input_path):
  """Create short examples."""
  input_basename = os.path.basename(input_path)
  output_basename = input_basename.replace(".jsonl.gz", ".short_pipeline.tfr")
  output_path = os.path.join(FLAGS.output_dir, output_basename)
  tf.logging.info("Converting examples in %s to tf.Examples.", input_path)
  with gzip.GzipFile(fileobj=tf.gfile.GFile(input_path, "rb")) as input_file:
    with tf.python_io.TFRecordWriter(output_path) as writer:
      for i, tf_example in enumerate(_generate_tf_examples(input_file)):
        writer.write(tf_example.SerializeToString())
        if i % 100 == 0:
          tf.logging.info("Wrote %d examples to %s", i + 1, output_path)
  tf.logging.info("Done converting examples from %s", input_path)


def main(_):
  input_paths = tf.gfile.Glob(FLAGS.input_pattern)
  tf.logging.info("Converting input %d files: %s", len(input_paths),
                  str(input_paths))
  tf.gfile.MakeDirs(FLAGS.output_dir)
  num_threads = min(FLAGS.max_threads, len(input_paths))
  if FLAGS.fork_workers:
    pool = multiprocessing.Pool(num_threads)
  else:
    pool = multiprocessing.dummy.Pool(num_threads)
  pool.map(_create_short_answer_examples, input_paths)


if __name__ == "__main__":
  app.run(main)

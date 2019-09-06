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
"""Create long answer TF examples from the original data in jsonl format.

See nq_long_dataset.py for details on how to use this data.
"""

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


def _int64_list_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _string_list_feature(values):
  values = [v.lower().encode("utf8") for v in values]
  return _bytes_list_feature(values)


def _generate_tf_examples(input_file):
  """Generate TF examples."""
  for line in input_file:
    if not isinstance(line, six.text_type):
      line = line.decode("utf-8")
    json_example = json.loads(line)
    long_answer_candidates = json_example["long_answer_candidates"]

    long_answer_indices = []
    candidate_strs = []
    curr_index = 0
    candidate_index_map = {}
    for candidate in long_answer_candidates:
      if candidate["top_level"]:
        # Only use candidate if it is a top level candidate.
        candidate_tokens = json_example["document_tokens"][
            candidate["start_token"]:candidate["end_token"]]
        candidate_tokens = [t["token"] for t in candidate_tokens]
        candidate_str = u" ".join(candidate_tokens)
        candidate_index_map[curr_index] = len(candidate_strs)
        candidate_strs.append(candidate_str)
      # Increment current index regardless.
      curr_index += 1

    question_str = u" ".join(json_example["question_tokens"])
    for annotation in json_example["annotations"]:
      candidate_index = annotation["long_answer"]["candidate_index"]

      # Remap candidate index to account for the fact that we are only
      # considering top level candidates for simplicity.
      if candidate_index in candidate_index_map:
        adjusted_candidate_index = candidate_index_map[candidate_index]
      else:
        adjusted_candidate_index = -1

      long_answer_indices.append(adjusted_candidate_index)

    features = {}
    features["question"] = _string_list_feature([question_str])
    features["context"] = _string_list_feature(candidate_strs)
    features["long_answer_indices"] = _int64_list_feature(long_answer_indices)
    yield tf.train.Example(features=tf.train.Features(feature=features))


def _create_long_answer_examples(input_path):
  """Create long answer examples."""
  input_basename = os.path.basename(input_path)
  output_basename = input_basename.replace(".jsonl.gz", ".long.tfr")
  output_path = os.path.join(FLAGS.output_dir, output_basename)
  tf.logging.info("Converting examples in %s to tf.Examples.", input_path)
  with gzip.GzipFile(fileobj=tf.gfile.GFile(input_path, "rb")) as input_file:
    with tf.python_io.TFRecordWriter(output_path) as writer:
      for i, tf_example in enumerate(_generate_tf_examples(input_file)):
        writer.write(tf_example.SerializeToString())
        if i % 100 == 0:
          tf.logging.info("Wrote %d examples to %s", i + 1, output_path)


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
  pool.map(_create_long_answer_examples, input_paths)


if __name__ == "__main__":
  app.run(main)

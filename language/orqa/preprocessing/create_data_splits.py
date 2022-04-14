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
"""Split multiple open-domain QA datasets into train, dev, and test."""
import json
import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("nq_train_path", None,
                    "Path to the Natural Questions (open) train data.")
flags.DEFINE_string("nq_dev_path", None,
                    "Path to the Natural Questions (open) dev data.")
flags.DEFINE_string("wb_train_path", None, "Path to WebQuestions train data.")
flags.DEFINE_string("wb_test_path", None, "Path to WebQuestions test data.")
flags.DEFINE_string("ct_train_path", None, "Path to CuratedTrec train data.")
flags.DEFINE_string("ct_test_path", None, "Path to CuratedTrec test data.")
flags.DEFINE_string("output_dir", None, "Output directory.")


def get_resplit_path(name, split):
  return os.path.join(FLAGS.output_dir,
                      "{}.resplit.{}.jsonl".format(name, split))


def resplit_data(input_path, output_path, keep_fn):
  """Resplit a dataset of jsonlines with QA pairs."""
  count = 0
  with tf.io.gfile.GFile(output_path, "w") as output_file:
    for line in tf.io.gfile.GFile(input_path):
      example = json.loads(line)
      if keep_fn(example["question"]):
        output_file.write(line)
        count += 1
  tf.logging.info("Wrote {} examples to {}".format(count, output_path))


def resplit_nq():
  """Resplit the Natural Questions dataset."""
  hash_fn = make_hash_fn()
  resplit_data(
      input_path=FLAGS.nq_train_path,
      output_path=get_resplit_path("NaturalQuestions", "train"),
      keep_fn=lambda x: hash_fn(x) != 0)
  resplit_data(
      input_path=FLAGS.nq_train_path,
      output_path=get_resplit_path("NaturalQuestions", "dev"),
      keep_fn=lambda x: hash_fn(x) == 0)
  resplit_data(
      input_path=FLAGS.nq_dev_path,
      output_path=get_resplit_path("NaturalQuestions", "test"),
      keep_fn=lambda x: True)


def resplit_wb():
  """Resplit the WebQuestions dataset."""
  hash_fn = make_hash_fn()
  resplit_data(
      input_path=FLAGS.wb_train_path,
      output_path=get_resplit_path("WebQuestions", "train"),
      keep_fn=lambda x: hash_fn(x) != 0)
  resplit_data(
      input_path=FLAGS.wb_train_path,
      output_path=get_resplit_path("WebQuestions", "dev"),
      keep_fn=lambda x: hash_fn(x) == 0)
  resplit_data(
      input_path=FLAGS.wb_test_path,
      output_path=get_resplit_path("WebQuestions", "test"),
      keep_fn=lambda x: True)


def resplit_ct():
  """Resplit the CuratedTrec dataset."""
  hash_fn = make_hash_fn()
  resplit_data(
      input_path=FLAGS.ct_train_path,
      output_path=get_resplit_path("CuratedTrec", "train"),
      keep_fn=lambda x: hash_fn(x) != 0)
  resplit_data(
      input_path=FLAGS.ct_train_path,
      output_path=get_resplit_path("CuratedTrec", "dev"),
      keep_fn=lambda x: hash_fn(x) == 0)
  resplit_data(
      input_path=FLAGS.ct_test_path,
      output_path=get_resplit_path("CuratedTrec", "test"),
      keep_fn=lambda x: True)


def make_hash_fn():
  session = tf.Session()
  placeholder = tf.placeholder(tf.string, [])
  hash_bucket = tf.strings.to_hash_bucket_fast(placeholder, 100000)
  result = tf.mod(hash_bucket, 10)

  def _hash_fn(x):
    return session.run(result, feed_dict={placeholder: x})

  return _hash_fn


def main(_):
  tf.io.gfile.makedirs(FLAGS.output_dir)
  resplit_nq()
  resplit_wb()
  resplit_ct()


if __name__ == "__main__":
  flags.mark_flag_as_required("nq_train_path")
  flags.mark_flag_as_required("nq_dev_path")
  flags.mark_flag_as_required("wb_train_path")
  flags.mark_flag_as_required("wb_test_path")
  flags.mark_flag_as_required("ct_train_path")
  flags.mark_flag_as_required("ct_test_path")
  flags.mark_flag_as_required("output_dir")
  app.run(main)

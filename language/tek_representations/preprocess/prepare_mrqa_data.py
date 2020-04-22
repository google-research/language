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
r"""Converts an mrqa dataset file to tf examples.

Notes:
  - this program needs to be run for every mrqa training shard.
  - this program only outputs the first n top level contexts for every example,
    where n is set through --max_contexts.
  - the restriction from --max_contexts is such that the annotated context might
    not be present in the output examples. --max_contexts=8 leads to about
    85% of examples containing the correct context. --max_contexts=48 leads to
    about 97% of examples containing the correct context.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
from language.tek_representations import run_mrqa
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

flags.DEFINE_string("split", "train",
                    "Train and dev split to read from and write to.")

flags.DEFINE_string("input_data_dir", "", "input_data_dir")

flags.DEFINE_string("output_data_dir", "", "output_data_dir")

flags.DEFINE_integer("n_shards", 50, "number of shards for this split")


def get_examples(input_jsonl_pattern):
  for input_path in tf.gfile.Glob(input_jsonl_pattern):
    with tf.gfile.Open(input_path) as input_file:
      for line in input_file:
        yield json.loads(line)


def get_shard():
  return "%05d-of-%05d" % (FLAGS.task_id, FLAGS.n_shards)


def main(_):
  examples_processed = 0
  creator_fn = run_mrqa.CreateTFExampleFn(is_training=FLAGS.is_training)

  instances = []
  input_file = os.path.join(
      FLAGS.input_data_dir,
      "%s.jsonl-%s" % (FLAGS.split, get_shard()))
  for example in get_examples(input_file):
    for instance in creator_fn.process(example):
      instances.append(instance)
    if examples_processed % 100 == 0:
      tf.logging.info("Examples processed: %d", examples_processed)
    examples_processed += 1
    if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
      break

  random.shuffle(instances)
  tf.logging.info("Total no: of instances in current shard: %d", len(instances))
  rec_output_file = os.path.join(FLAGS.output_data_dir,
                                 "%s.tfrecord-%s" % (FLAGS.split, get_shard()))
  with tf.python_io.TFRecordWriter(rec_output_file) as writer:
    for instance, _ in instances:
      writer.write(instance)
  if not FLAGS.is_training:
    fea_output_file = os.path.join(
        FLAGS.output_data_dir,
        "%s.features.jsonl-%s" % (FLAGS.split, get_shard()))
    with tf.gfile.Open(fea_output_file, "w") as writer:
      for _, instance in instances:
        writer.write(instance)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()

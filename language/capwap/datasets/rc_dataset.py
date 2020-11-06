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
"""Creates a tf.data.Dataset for an image-based RC dataset.

The RC datasets are structured (mostly) SQuAD-style. The examples consist of
questions and contexts with (intended) extractive answers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import json

from language.capwap.utils import io_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

KEYS = frozenset(
    {"question_id", "image_id", "beam_id", "context_inputs", "question_inputs"})


def parse_example(serialized_example):
  """Parse a serialized example proto."""
  features = tf.io.parse_single_example(
      serialized_example,
      dict(
          beam_id=tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
          image_id=tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
          question_id=tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
          context=tf.io.FixedLenFeature(shape=[], dtype=tf.string),
          question=tf.io.FixedLenFeature(shape=[], dtype=tf.string)))
  return features


def iter_rc_examples(context_file, qa_file, vocab):
  """Join examples by id and yield."""
  # image_id --> qid --> list(beams)
  contexts = collections.defaultdict(lambda: collections.defaultdict(list))
  with tf.io.gfile.GFile(context_file, "r") as f:
    for line in f:
      line = json.loads(line)
      for sample in line["token_ids"]:
        sample = [vocab.i2t(idx) for idx in sample]
        try:
          eos_idx = sample.index(vocab.SEP)
          sample = sample[:eos_idx]
        except ValueError:
          pass
        sample = " ".join(sample)
        question_id = line.get("question_id", -1)
        if question_id < 0:
          question_id = "default"
        contexts[line["image_id"]][question_id].append(sample)

  with tf.io.gfile.GFile(qa_file, "r") as f:
    rc_dataset = json.load(f)

  # JSON keys are automatically stored as strings, so we map between
  # ints and strings as necessary. We ignore answers.
  for image_id, qas in rc_dataset.items():
    image_id = int(image_id)
    for question_id, qa in qas.items():
      question_id = int(question_id)
      captions = contexts[image_id]
      if question_id in captions:
        conditional_contexts = captions[question_id]
      else:
        conditional_contexts = captions["default"]
      for i, context in enumerate(conditional_contexts):
        features = dict(
            beam_id=io_utils.int64_feature(i),
            image_id=io_utils.int64_feature(image_id),
            question_id=io_utils.int64_feature(question_id),
            context=io_utils.string_feature(context),
            question=io_utils.string_feature(qa["question"]))
        yield tf.train.Example(features=tf.train.Features(feature=features))


def preprocess_mapper(features, params, lookup_table):
  """Model-specific preprocessing of features from the dataset."""
  features["question_inputs"] = text_utils.build_text_inputs(
      text=features["question"],
      length=params["question_length"],
      lookup_table=lookup_table)

  features["context_inputs"] = text_utils.build_text_inputs(
      text=features["context"],
      length=params["context_length"],
      lookup_table=lookup_table)

  # Remove extra inputs.
  features = {f: features[f] for f in features if f in KEYS}

  return features


def get_dataset(params, mode, caption_file, qa_file, scratch_file, vocab):
  """Gets a tf.data.Dataset with RC data for eval. Processes data online."""
  assert mode == tf.estimator.ModeKeys.PREDICT

  # Create writer for temp tfrecord data.
  tf.logging.info("Converting examples to TFRecords...")
  writer = tf.io.TFRecordWriter(scratch_file)
  count = 0
  for example in iter_rc_examples(caption_file, qa_file, vocab):
    count += 1
    writer.write(example.SerializeToString())
    if count % 10000 == 0:
      tf.logging.info("Serialized %d examples...", count)
  writer.close()
  tf.logging.info("Serialized %d examples.", count)

  dataset = tf.data.TFRecordDataset(scratch_file, buffer_size=16 * 1024 * 1024)
  dataset = dataset.map(
      parse_example, num_parallel_calls=params["num_input_threads"])
  dataset = dataset.map(
      functools.partial(
          preprocess_mapper,
          params=params,
          lookup_table=vocab.get_string_lookup_table()),
      num_parallel_calls=params["num_input_threads"])

  # Batch.
  dataset = dataset.batch(params["batch_size"], drop_remainder=False)
  dataset = dataset.prefetch(params["prefetch_batches"])

  return dataset

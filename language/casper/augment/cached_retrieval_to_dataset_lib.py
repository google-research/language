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
r"""Helper methods for generating a retrieval-augmented dataset."""
import json
import os
import random
from typing import Iterable, Iterator, List, Optional, Sequence

from absl import logging
from language.casper.augment import casper_converters
from language.casper.utils import data_types
import tensorflow as tf


RawExample = data_types.RawExample
AugmentedExample = data_types.AugmentedExample


def expand_path_patterns(path_patterns: Iterable[str]) -> Sequence[str]:
  """Expands the glob patterns in the given list."""
  paths = []
  for pattern in path_patterns:
    paths.extend(tf.io.gfile.glob(pattern))
  return paths


def _bytes_feature(value: bytes) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _to_tf_example(example: AugmentedExample) -> tf.train.Example:
  feature = {
      "inputs": _bytes_feature(example.inputs.encode()),
      "targets": _bytes_feature(example.targets.encode()),
  }
  tf_ex = tf.train.Example(features=tf.train.Features(feature=feature))
  return tf_ex.SerializeToString()


def read_orig_examples(data_paths: Iterable[str]) -> Iterator[RawExample]:
  """Reads and deserializes JSONLs from files."""
  source_files = expand_path_patterns(data_paths)
  for source_file in source_files:
    logging.info("Start reading from %s", source_file)
    with tf.io.gfile.GFile(source_file) as reader:
      for line in reader:
        yield json.loads(line)


def write_examples(examples: Sequence[AugmentedExample],
                   base_path: str,
                   file_format: str = "tfr",
                   num_shards: Optional[int] = None) -> None:
  """Writes examples to sharded TSV or TFRecord files."""
  if not num_shards:
    # Automatically compute the number of shards
    num_shards = max(1, len(examples) // 5000)
  shard_size = len(examples) // num_shards + 1
  for i in range(num_shards):
    filename = "{}.{}-{:05d}-of-{:05d}".format(base_path, file_format, i,
                                               num_shards)
    shard_examples = examples[i * shard_size:(i + 1) * shard_size]
    logging.info("Writing to %s", filename)
    if file_format == "tsv":
      with tf.io.gfile.GFile(filename, "w") as writer:
        for example in shard_examples:
          writer.write("{}\t{}\n".format(example.inputs, example.targets))
    elif file_format == "tfr":
      with tf.io.TFRecordWriter(filename) as writer:
        for example in shard_examples:
          writer.write(_to_tf_example(example))
    else:
      raise ValueError("Unknown file format: {}".format(file_format))


def create_augmented_examples(orig_examples: Iterable[RawExample],
                              converter: casper_converters.BaseExampleConverter,
                              split: str,
                              log_every: int = 1000) -> List[AugmentedExample]:
  """Creates AugmentedExamples from the raw JSONLs.

  Args:
    orig_examples: An Iterable of deserialzied JSONs.
    converter: A subclass of BaseExampleConverter.
    split: Split name (used only for logging).
    log_every: Logging frequency.

  Returns:
    a list of AugmentedExamples.
  """
  examples = []
  for i, orig_example in enumerate(orig_examples):
    if i % log_every == 0:
      logging.info("[%s:%d] Produced %d examples", split, i, len(examples))
    converter.verify_exemplars(orig_example)
    examples.extend(converter.convert(orig_example))
  logging.info("[%s] Produced %d examples total.", split, len(examples))
  return examples


def generate_dataset(orig_train: Iterable[RawExample],
                     orig_dev: Iterable[RawExample],
                     orig_test: Iterable[RawExample],
                     converter: casper_converters.BaseExampleConverter,
                     output_dir: str,
                     seed: int = 42,
                     log_every: int = 1000,
                     train_filename: Optional[str] = "train",
                     dev_filename: Optional[str] = "dev",
                     test_filename: Optional[str] = "test",
                     file_format: str = "tfr") -> None:
  """Generates and writes retrieval-augmented dataset files.

  Args:
    orig_train: Iterable of deserialized JSONs for training data.
    orig_dev: Iterable of deserialized JSONs for dev data.
    orig_test: Iterable of deserialized JSONs for test data.
    converter: A subclass of BaseExampleConverter.
    output_dir: Output directory.
    seed: Random seed.
    log_every: Logging frequency.
    train_filename: Training data filename prefix.
    dev_filename: Dev data filename prefix.
    test_filename: Test data filename prefix.
    file_format: Output file format.
  """
  random.seed(seed)
  tf.io.gfile.makedirs(output_dir)

  converter.stats.clear()
  examples = create_augmented_examples(
      orig_train, converter, split="train", log_every=log_every)
  if examples:
    # Shuffle the training data.
    random.shuffle(examples)
    base_path = os.path.join(output_dir, train_filename)
    write_examples(examples, base_path, file_format=file_format)
    logging.info("Train data stats: %s", dict(converter.stats))
  else:
    logging.warn("No train examples generated.")

  converter.stats.clear()
  examples = create_augmented_examples(
      orig_dev, converter, split="dev", log_every=log_every)
  if examples:
    # The dev data is not shuffled for easier error analysis.
    base_path = os.path.join(output_dir, dev_filename)
    write_examples(examples, base_path, file_format=file_format)
    logging.info("Dev data stats: %s", dict(converter.stats))
  else:
    logging.warn("No dev examples generated.")

  converter.stats.clear()
  examples = create_augmented_examples(
      orig_test, converter, split="test", log_every=log_every)
  if examples:
    # The test data is not shuffled for easier error analysis.
    base_path = os.path.join(output_dir, test_filename)
    write_examples(examples, base_path, file_format=file_format)
    logging.info("Test data stats: %s", dict(converter.stats))
  else:
    logging.warn("No test examples generated.")

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
"""ORQA dataset."""
import json
import os

from language.common.utils import tensor_utils
import tensorflow.compat.v1 as tf


def parse_line(tf_line):
  """Parse line."""
  def _do_parse(line):
    example = json.loads(line)
    return example["question"], example["answer"]

  tf_question, tf_answers = tensor_utils.shaped_py_func(
      func=_do_parse,
      inputs=[tf_line],
      types=[tf.string, tf.string],
      shapes=[[], [None]],
      stateful=False)
  return dict(question=tf_question, answers=tf_answers)


def get_dataset(data_root, name, split):
  """Gets a tf.data.Dataset."""
  assert split in ("train", "dev", "test")
  dataset_path = os.path.join(data_root,
                              "{}.resplit.{}.jsonl".format(name, split))
  dataset = tf.data.TextLineDataset(dataset_path)
  dataset = dataset.map(parse_line, num_parallel_calls=12)

  if split == "train":
    with tf.io.gfile.GFile(dataset_path) as f:
      dataset_size = sum(1 for _ in f)
    dataset = dataset.shuffle(dataset_size)
  return dataset

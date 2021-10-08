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
"""Text classification dataset."""
import json
import os

from language.common.utils import tensor_utils
import tensorflow.compat.v1 as tf

_FEVER_SPLIT_TO_FILENAME = {
    "train": "train.jsonl",
    "dev": "shared_task_dev.jsonl",
}

_FEVER_LABEL_DICT = {
    "REFUTES": 0,
    "SUPPORTS": 1,
    "NOT ENOUGH INFO": 2,
}

_HOVER_SPLIT_TO_FILENAME = {
    "train": "hover_train_release_v1.1.json",
    "dev": "hover_dev_release_v1.1.json",
}

_HOVER_LABEL_DICT = {
    "NOT_SUPPORTED": 0,
    "SUPPORTED": 1,
}


def parse_fever_line(tf_line):
  """Convert FEVER sample from a string."""

  def _do_parse(line):
    example = json.loads(line)
    return example["claim"], _FEVER_LABEL_DICT[example["label"]]

  tf_question, tf_answers = tensor_utils.shaped_py_func(
      func=_do_parse,
      inputs=[tf_line],
      types=[tf.string, tf.int64],
      shapes=[[], []],
      stateful=False)
  return dict(question=tf_question, answers=tf_answers)


def get_fever_dataset(data_root, split):
  """Gets a tf.data.Dataset for FEVER."""
  filename = _FEVER_SPLIT_TO_FILENAME[split]
  dataset_path = os.path.join(data_root, "fever", filename)
  dataset = tf.data.TextLineDataset(dataset_path)
  dataset = dataset.map(parse_fever_line, num_parallel_calls=12)

  if split == "train":
    with tf.io.gfile.GFile(dataset_path) as f:
      dataset_size = sum(1 for _ in f)
    dataset = dataset.shuffle(dataset_size)
  return dataset


def get_hover_dataset(data_root, split):
  """Gets a tf.data.Dataset for FEVER."""
  filename = _HOVER_SPLIT_TO_FILENAME[split]
  dataset_path = os.path.join(data_root, "hover", filename)

  with tf.io.gfile.GFile(dataset_path) as f:
    data = json.load(f)
  tf.logging.info("Loaded %d samples from %s" % (len(data), dataset_path))

  dataset = tf.data.Dataset.from_tensor_slices(
      dict(
          question=[row["claim"] for row in data],
          answers=[
              tf.cast(_HOVER_LABEL_DICT[row["label"]], tf.int64) for row in data
          ]))

  if split == "train":
    dataset = dataset.shuffle(len(data))
  return dataset


def get_dataset(data_root, name, split):
  """Gets a tf.data.Dataset."""
  if name == "fever":
    return get_fever_dataset(data_root, split)
  elif name == "hover":
    return get_hover_dataset(data_root, split)
  else:
    raise ValueError("Unknown dataset " + name)

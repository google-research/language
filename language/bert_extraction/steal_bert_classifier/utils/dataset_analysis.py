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
"""Script to analyze the class distribution and their entropies in a dataset."""
import numpy as np

from scipy import stats
import tensorflow.compat.v1 as tf
from tqdm import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("input_path", None,
                    "Path to the dataset which needs to be analyzed.")
flags.DEFINE_enum("task_name", "sst2", ["sst2", "mnli"],
                  "The dataset's name, useful for task-specific information.")

FLAGS = flags.FLAGS

num_labels = {"sst2": 2, "mnli": 3}


def main(_):

  with gfile.Open(FLAGS.input_path, "r") as f:
    sents_data = f.read().strip().split("\n")[1:]

  classes = [0 for _ in range(num_labels[FLAGS.task_name])]
  entropies = []

  # Assume that the last three columns are probability information
  for x in tqdm(sents_data):
    probs = (x.split("\t"))[-num_labels[FLAGS.task_name]:]
    probs = [float(x1) for x1 in probs]
    entropies.append(stats.entropy(probs))
    classes[np.argmax(probs)] += 1

  class_distro = []
  for i, cls1 in enumerate(classes):
    class_distro.append(float(cls1) / len(sents_data))
    logging.info("Class %d = %.6f (%d / %d)", i,
                 float(cls1) / len(sents_data), cls1, len(sents_data))

  class_entropy = stats.entropy(class_distro)
  logging.info("Class distribution self-entropy = %.8f", class_entropy)
  logging.info("Average per-instance self-entropy = %.8f", np.mean(entropies))
  logging.info("Max per-instance self-entropy = %.8f", np.max(entropies))
  logging.info("Min per-instance self-entropy = %.8f", np.min(entropies))
  logging.info("Std per-instance self-entropy = %.8f", np.std(entropies))
  return


if __name__ == "__main__":
  app.run(main)

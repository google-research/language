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
"""Script to analyze the class distributions and compare a pair of datasets.

The two datasets have the same set of inputs but different outputs, such as a
comparison between the student and teacher on a attack dataset.
"""
import numpy as np

from scipy import stats

import tensorflow.compat.v1 as tf
from tqdm import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("input_path1", None,
                    "Dataset path containing teacher outputs")
flags.DEFINE_string("input_path2", None,
                    "Dataset path containing student outputs")
flags.DEFINE_string(
    "task_name", "sst2",
    "Task in consideration, to know task-specific information.")

FLAGS = flags.FLAGS

num_labels = {"sst2": 2, "sst-2": 2, "mnli": 3}


def main(_):

  with gfile.Open(FLAGS.input_path1, "r") as f:
    sents_data1 = f.read().strip().split("\n")[1:]

  with gfile.Open(FLAGS.input_path2, "r") as f:
    sents_data2 = f.read().strip().split("\n")[1:]

  assert len(sents_data1) == len(sents_data2)

  self_entropies1 = []
  self_entropies2 = []
  cross_entropies = []
  l2_distances = []
  margins1 = []
  margins2 = []

  for x1, x2 in tqdm(zip(sents_data1, sents_data2)):

    probs1 = (x1.split("\t"))[-num_labels[FLAGS.task_name]:]
    probs1 = np.array([float(x) for x in probs1])

    probs2 = (x2.split("\t"))[-num_labels[FLAGS.task_name]:]
    probs2 = np.array([float(x) for x in probs2])

    cross_entropies.append(stats.entropy(probs1, probs2))

    l2_distances.append(np.linalg.norm(probs1 - probs2))

    self_entropies1.append(stats.entropy(probs1))
    self_entropies2.append(stats.entropy(probs2))

    sorted_probs1 = np.sort(probs1)[::-1]
    sorted_probs2 = np.sort(probs2)[::-1]
    margins1.append(sorted_probs1[0] - sorted_probs1[1])
    margins2.append(sorted_probs2[0] - sorted_probs2[1])

  logging.info("Average cross_entropies = %.8f", np.mean(cross_entropies))
  logging.info("Average l2_distances = %.8f", np.mean(l2_distances))

  logging.info("Average self_entropies1 = %.8f", np.mean(self_entropies1))
  logging.info("Average self_entropies2 = %.8f", np.mean(self_entropies2))

  logging.info("Average margin1 = %.8f", np.mean(margins1))
  logging.info("Average margin2 = %.8f", np.mean(margins2))

  logging.info("spearman cross_entropies, self_entropies1 = %.8f",
               stats.spearmanr(cross_entropies, self_entropies1)[0])
  logging.info("spearman cross_entropies, self_entropies2 = %.8f",
               stats.spearmanr(cross_entropies, self_entropies2)[0])
  logging.info("spearman l2_distances, self_entropies1 = %.8f",
               stats.spearmanr(l2_distances, self_entropies1)[0])
  logging.info("spearman l2_distances, self_entropies2 = %.8f",
               stats.spearmanr(l2_distances, self_entropies2)[0])
  logging.info("spearman self_entropies1, self_entropies2 = %.8f",
               stats.spearmanr(self_entropies1, self_entropies2)[0])

  logging.info("spearman cross_entropies, margins1 = %.8f",
               stats.spearmanr(cross_entropies, margins1)[0])
  logging.info("spearman cross_entropies, margins2 = %.8f",
               stats.spearmanr(cross_entropies, margins2)[0])
  logging.info("spearman l2_distances, margins1 = %.8f",
               stats.spearmanr(l2_distances, margins1)[0])
  logging.info("spearman l2_distances, margins2 = %.8f",
               stats.spearmanr(l2_distances, margins2)[0])
  logging.info("spearman margins1, margins2 = %.8f",
               stats.spearmanr(margins1, margins2)[0])

  for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
    top_frac = int(frac * len(l2_distances))
    top_frac_l2_distances = np.argsort(l2_distances)[-1 * top_frac:]
    top_frac_self_entropies1 = np.argsort(self_entropies1)[-1 * top_frac:]
    top_frac_self_entropies2 = np.argsort(self_entropies2)[-1 * top_frac:]
    top_frac_margins1 = np.argsort(margins1)[:top_frac]
    top_frac_margins2 = np.argsort(margins2)[:top_frac]

    logging.info(
        "intersection top %.1f l2_distances, self_entropies1 = %d / %d", frac,
        len(np.intersect1d(top_frac_l2_distances, top_frac_self_entropies1)),
        top_frac)

    logging.info(
        "intersection top %.1f l2_distances, self_entropies2 = %d / %d", frac,
        len(np.intersect1d(top_frac_l2_distances, top_frac_self_entropies2)),
        top_frac)

    logging.info(
        "intersection top %.1f self_entropies1, self_entropies2 = %d / %d",
        frac,
        len(np.intersect1d(top_frac_self_entropies1,
                           top_frac_self_entropies2)), top_frac)

    logging.info(
        "intersection top %.1f all three = %d / %d", frac,
        len(
            np.intersect1d(
                top_frac_l2_distances,
                np.intersect1d(top_frac_self_entropies1,
                               top_frac_self_entropies2))), top_frac)

    logging.info("intersection top %.1f l2_distances, margins1 = %d / %d", frac,
                 len(np.intersect1d(top_frac_l2_distances, top_frac_margins1)),
                 top_frac)

    logging.info("intersection top %.1f l2_distances, margins2 = %d / %d", frac,
                 len(np.intersect1d(top_frac_l2_distances, top_frac_margins2)),
                 top_frac)

    logging.info("intersection top %.1f margins1, margins2 = %d / %d", frac,
                 len(np.intersect1d(top_frac_margins1, top_frac_margins2)),
                 top_frac)

    logging.info(
        "intersection top %.1f all three = %d / %d", frac,
        len(
            np.intersect1d(top_frac_l2_distances,
                           np.intersect1d(top_frac_margins1,
                                          top_frac_margins2))), top_frac)

  return


if __name__ == "__main__":
  app.run(main)

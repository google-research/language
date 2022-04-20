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
"""Compare performance of a watermarked / nonwatermarked model on watermark."""

import numpy as np
import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("task_name", "sst2", "Name of task to preprocess")
flags.DEFINE_string("watermark_path", None,
                    "Path with watermark labels and victim's original labels")
flags.DEFINE_string("watermark_probs_path", None,
                    "Path with watermark predictions from watermarked model")
flags.DEFINE_string(
    "nonwatermark_model_watermark_probs_path", None,
    "Path with watermark predictions from nonwatermarked model")
FLAGS = flags.FLAGS

num_labels = {"sst-2": 2, "mnli": 3}


def main(_):
  task_name = FLAGS.task_name.lower()

  with gfile.Open(FLAGS.watermark_path, "r") as f:
    wm_data = f.read().strip().split("\n")

  header = wm_data[0].split("\t")
  watermark_prob_indices = [
      header.index("watermark%d_prob" % i) for i in range(num_labels[task_name])
  ]
  original_prob_indices = [
      header.index("original%d_prob" % i) for i in range(num_labels[task_name])
  ]
  wm_data = wm_data[1:]

  with gfile.Open(FLAGS.watermark_probs_path, "r") as f:
    probs_data = f.read().strip().split("\n")

  assert len(wm_data) == len(probs_data)

  wm_matches = 0
  orig_matches = 0

  # Also store the original distribution of the watermark labels
  watermark_distro = [0 for _ in range(num_labels[task_name])]

  for wm_instance, prob_str in zip(wm_data, probs_data):
    wm_shards = wm_instance.split("\t")
    wm_probs = np.array([float(wm_shards[i]) for i in watermark_prob_indices])
    orig_probs = np.array([float(wm_shards[i]) for i in original_prob_indices])
    pred_probs = np.array([float(yy) for yy in prob_str.split("\t")])

    if np.argmax(pred_probs) == np.argmax(wm_probs):
      wm_matches += 1
    if np.argmax(pred_probs) == np.argmax(orig_probs):
      orig_matches += 1

    watermark_distro[np.argmax(wm_probs)] += 1

  watermark_distro = [str(x) for x in watermark_distro]

  # For a watermarked model, we expect the watermark matches to be high and
  # the original matches to be low.
  logging.info("Watermark matches = %d / %d", wm_matches, len(wm_data))
  logging.info("Original matches = %d / %d", orig_matches, len(wm_data))
  logging.info("Watermark distro = %s", ", ".join(watermark_distro))

  if FLAGS.nonwatermark_model_watermark_probs_path:
    with gfile.Open(FLAGS.nonwatermark_model_watermark_probs_path, "r") as f:
      non_wm_probs_data = f.read().strip().split("\n")

    assert len(wm_data) == len(non_wm_probs_data)

    wm_matches = 0
    orig_matches = 0

    for wm_instance, prob_str in zip(wm_data, non_wm_probs_data):
      wm_shards = wm_instance.split("\t")
      wm_probs = np.array([float(wm_shards[i]) for i in watermark_prob_indices])
      orig_probs = np.array(
          [float(wm_shards[i]) for i in original_prob_indices])
      non_wm_pred_probs = np.array([float(yy) for yy in prob_str.split("\t")])

      if np.argmax(non_wm_pred_probs) == np.argmax(wm_probs):
        wm_matches += 1
      if np.argmax(non_wm_pred_probs) == np.argmax(orig_probs):
        orig_matches += 1

    # For a non-watermarked model, we expect Original matches to be high and
    # watermarked matches to be low.
    logging.info("Non-watermarked model, Watermark matches = %d / %d",
                 wm_matches, len(wm_data))
    logging.info("Non-watermarked model, Original matches = %d / %d",
                 orig_matches, len(wm_data))


if __name__ == "__main__":
  app.run(main)

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
"""Script to combine multiple shards from discrete_invert_embeddings to a single dataset."""

import tensorflow.compat.v1 as tf
from tqdm import tqdm

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("shards_pattern", None,
                    "Glob pattern to identify the set of files to be combined")
flags.DEFINE_string("task_name", "mnli",
                    "Task name to understand the input data format")
flags.DEFINE_string("output_path", None,
                    "Output path where the combined dataset is exported")
FLAGS = flags.FLAGS

num_labels = {"sst2": 2, "mnli": 3}
relevant_headers = {
    "sst2": ["original_index", "sentence"],
    "mnli": ["original_index", "sentence1", "sentence2"]
}


def main(_):
  task_name = FLAGS.task_name.lower()
  # skip the original index header for the final file for compatiblity with
  # run_classifier_distillation.py
  output_data = ["index\t" + "\t".join(relevant_headers[task_name][1:])]
  shards = gfile.Glob(FLAGS.shards_pattern)
  # sort the shard according to their starting point
  shards.sort(key=lambda x: int(x[x.rfind(".") + 1:x.rfind("-")]))

  for shard in tqdm(shards):
    logging.info("Loading file %s", shard)
    with gfile.Open(shard, "r") as f:
      # read the dataset ignoring the header
      dataset = f.read().strip().split("\n")
      header = dataset[0].split("\t")
      dataset = dataset[1:]

    relevant_indices = [header.index(x) for x in relevant_headers[task_name]]

    logging.info("Dataset size = %d, Relevant indices = %s", len(dataset),
                 relevant_indices)

    for point in dataset:
      point_parts = point.split("\t")
      output_data.append("\t".join([point_parts[x] for x in relevant_indices]))

  logging.info("Final dataset of size %d from %d files",
               len(output_data) - 1, len(shards))

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")


if __name__ == "__main__":
  app.run(main)

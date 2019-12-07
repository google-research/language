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
"""Concatenate a list of datasets."""
import tensorflow as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("dataset_paths", None, "CSV list of datasets to combine")
flags.DEFINE_string("output_path", None,
                    "New output directory where output corpus will be dumped")
flags.DEFINE_string("task_name", "mnli", "Task in consideration")

FLAGS = flags.FLAGS

num_labels = {"sst2": 2, "mnli": 3}
relevant_headers = {"sst2": ["sentence"], "mnli": ["sentence1", "sentence2"]}


def main(_):

  output_data = []
  dataset_paths = FLAGS.dataset_paths.split(",")

  for dp in dataset_paths:
    with gfile.Open(dp, "r") as f:
      base_dataset = f.read().strip().split("\n")
      base_dataset_header = base_dataset[0]
      base_dataset = base_dataset[1:]

    indices_base_dataset = [
        base_dataset_header.split("\t").index(x)
        for x in relevant_headers[FLAGS.task_name]
    ]

    for point in base_dataset:
      input_shards = [
          point.split("\t")[index] for index in indices_base_dataset
      ]
      output_data.append(("%d\t" % len(output_data)) + "\t".join(input_shards))

    logging.info("Final dataset size = %d", len(output_data))

  final_header = "index\t" + "\t".join(relevant_headers[FLAGS.task_name])
  output_data = [final_header] + output_data

  with gfile.Open(FLAGS.output_path, "w") as f:
    f.write("\n".join(output_data) + "\n")


if __name__ == "__main__":
  app.run(main)

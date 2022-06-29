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
r"""The Official Evalucation Script for UpdateRouge.

    Before using this script, please first run convert_task_to_jsonl.py to
    genenerate two files: input_labels.jsonl and input_only.jsonl

    Apply your model on the input_only.jsonl to generate the predictions as
    pred.jsonl (check the sample file for the data format).

    Now use this script with the following arguments.

     --input_labels_jsonl=input_labels.jsonl
     --prediction_jsonl=pred.jsonl
     --task_name=wikidiff_diff_all_text_reference_gold_test

    to get the evaluation output.

"""
import json

from absl import app
from absl import flags
from language.fruit import tasks  # pylint: disable=unused-import
import seqio
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_labels_jsonl",
    None,
    "JSONL file containing input and (targeted) outputs.",
)

flags.DEFINE_string(
    "prediction_jsonl",
    None,
    "JSONL file containing prediction.",
)

flags.DEFINE_string(
    "task_name",
    None,
    "SeqIO task name",
)

flags.DEFINE_multi_string(
    "filter",
    [],
    "Metrics to filter",
)


def _maybe_unpack(x):
  if isinstance(x, seqio.metrics.Scalar):
    return x.value
  elif isinstance(x, seqio.metrics.Text):
    return x.textdata
  else:
    return x


def main(_):
  task = seqio.TaskRegistry.get(FLAGS.task_name)
  predict_metric_fns = task.predict_metric_fns

  targets = []
  predictions = []
  with tf.io.gfile.GFile(FLAGS.input_labels_jsonl, "r") as input_f:
    for line in input_f:
      instance = json.loads(line)
      targets.append(instance)

  with tf.io.gfile.GFile(FLAGS.prediction_jsonl, "r") as pred_f:
    for line in pred_f:
      instance = json.loads(line)
      predictions.append(instance)

  all_metrics = {}
  filtered_metrics = FLAGS.filter
  for metric_fn in predict_metric_fns:
    result = metric_fn(targets, predictions)
    for k, v in result.items():
      if k in all_metrics:
        raise ValueError(f"Duplicate metric key {k}")
      elif k in filtered_metrics:
        continue
      all_metrics[k] = _maybe_unpack(v)

  print(json.dumps(all_metrics))


if __name__ == "__main__":
  app.run(main)

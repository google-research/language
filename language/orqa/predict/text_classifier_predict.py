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
"""ORQA predictions."""
import json
from absl import flags
from absl import logging
from language.orqa.models import text_classifier_model
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Model directory.")
flags.DEFINE_string("dataset_path", None, "Data path.")
flags.DEFINE_string("predictions_path", None,
                    "Path to file where predictions will be written")
flags.DEFINE_string("question_key", "claim",
                    "Feature name for the question in the input JSON data.")
flags.DEFINE_boolean("print_prediction_samples", True,
                     "Whether to print a sample of the predictions.")


def main(_):
  predictor = text_classifier_model.get_predictor(FLAGS.model_dir)
  with tf.io.gfile.GFile(FLAGS.predictions_path, "w") as predictions_file:
    with tf.io.gfile.GFile(FLAGS.dataset_path) as dataset_file:
      for i, line in enumerate(dataset_file):
        example = json.loads(line)
        question = example[FLAGS.question_key]
        predictions = predictor(question)
        predicted_answer = predictions["answer"].tolist()
        example["prediction"] = predicted_answer
        predictions_file.write(json.dumps(example))
        predictions_file.write("\n")
        if FLAGS.print_prediction_samples and i & (i - 1) == 0:
          logging.info("[%d] '%s' -> '%s'", i, question, predicted_answer)

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()

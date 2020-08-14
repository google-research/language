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
"""ORQA evaluation."""
import json
from absl import flags
from absl import logging
from language.orqa.models import orqa_model
from language.orqa.utils import eval_utils
import six
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Model directory.")
flags.DEFINE_string("dataset_path", None, "Data path.")
flags.DEFINE_boolean("print_prediction_samples", False,
                     "Whether to print a sample of the predictions.")


def main(_):
  predictor = orqa_model.get_predictor(FLAGS.model_dir)
  example_count = 0
  correct_count = 0
  with tf.io.gfile.GFile(FLAGS.dataset_path) as dataset_file:
    for i, line in enumerate(dataset_file):
      example = json.loads(line)
      question = example["question"]
      answers = example["answer"]
      predictions = predictor(question)
      predicted_answer = six.ensure_text(predictions["answer"], errors="ignore")
      is_correct = eval_utils.is_correct(
          answers=[six.ensure_text(a) for a in answers],
          prediction=predicted_answer,
          is_regex=False)
      correct_count += int(is_correct)
      example_count += 1
      if FLAGS.print_prediction_samples and i & (i - 1) == 0:
        logging.info("[%d] '%s' -> '%s'", i, question, predicted_answer)

  logging.info("Accuracy: %.4f (%d/%d)", correct_count / float(example_count),
               correct_count, example_count)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()

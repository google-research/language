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
"""Main file for running a NQ long-answer experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.utils import experiment_utils
from language.question_answering.models import nq_long_model

import tensorflow as tf


def main(_):
  model_function, train_input_fn, eval_input_fn, serving_input_receiver_fn = (
      nq_long_model.experiment_functions())

  best_exporter = tf.estimator.BestExporter(
      name="best",
      serving_input_receiver_fn=serving_input_receiver_fn,
      event_file_pattern="eval_default/*.tfevents.*",
      compare_fn=nq_long_model.compare_metrics)

  experiment_utils.run_experiment(
      model_fn=model_function,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      exporters=[best_exporter])


if __name__ == "__main__":
  tf.app.run()

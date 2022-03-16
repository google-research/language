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
"""Utilities for TF writer."""

import tensorflow as tf


def get_summary_writer(write_dir):
  return tf.summary.create_file_writer(write_dir)


def write_metrics(writer, metrics_dict, step):
  for metric_name, metric_value in metrics_dict.items():
    with writer.as_default():
      tf.summary.scalar(metric_name, metric_value, step=step)

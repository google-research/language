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
"""Functions supporting evaluation scripts."""

import collections


def print_avg_by_key(examples, metrics, key_fn):
  """Prints avg results partitioned according to some `key_fn`.

  Args:
    examples: List of Example instances.
    metrics: List of float values corresponding to `examples`.
    key_fn: Function that takes an Example and returns a key to aggregate over.
  """
  key_to_metrics = collections.defaultdict(list)
  for example, metric in zip(examples, metrics):
    key = key_fn(example)
    key_to_metrics[key].append(metric)
  # Compute averages.
  key_to_avg = {
      key: sum(vals) / len(vals) for key, vals in key_to_metrics.items()
  }
  for key, val in key_to_avg.items():
    print("%s (%s): %s" % (key, len(key_to_metrics[key]), val))


def print_avg_by_template(examples, metrics):
  """Prints results partitioned by template."""
  key_fn = lambda example: example.metadata.template
  return print_avg_by_key(examples, metrics, key_fn)


def print_avg(examples, metrics):
  key_fn = lambda unused_x: "all"
  return print_avg_by_key(examples, metrics, key_fn)

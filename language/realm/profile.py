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
"""Profiling utilities."""
import collections
import functools
import time

# Maps from a name to a duration.
TOTAL_DURATION = collections.Counter()

# Maps from a name to a count.
TOTAL_COUNT = collections.Counter()


def reset():
  global TOTAL_DURATION
  global TOTAL_COUNT
  TOTAL_DURATION = collections.Counter()
  TOTAL_COUNT = collections.Counter()


def profiled_function(fn):

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    with Timer(fn.__name__):
      return fn(*args, **kwargs)

  return wrapped


class Timer(object):
  """Times a block of code, and adds the result to a global tally."""

  def __init__(self, name):
    self.name = name

  def __enter__(self):
    self.start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    duration = time.time() - self.start_time
    TOTAL_DURATION[self.name] += duration
    TOTAL_COUNT[self.name] += 1


def print_report():
  for name, seconds in TOTAL_DURATION.most_common():
    count = TOTAL_COUNT[name]
    print('{:.2f} ({}) {}'.format(seconds, count, name))

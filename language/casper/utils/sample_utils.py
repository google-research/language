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
"""Utilities related to random sampling."""
import random



def uniform_sample(pool, max_num_items):
  """Samples k items using uniform sampling without replacement."""
  pool = list(pool)
  random.shuffle(pool)
  return pool[:max_num_items]


def geometric_sample(pool, max_num_items,
                     sample_prob):
  """Samples k items using geometric sampling without replacement.

  At each step, item #i will be sampled with probability sample_prob^i (except
  the last item which gets the remaining probability mass). Item #i will then
  be removed from the pool.

  Args:
    pool: An ordered list of items to sample.
    max_num_items: Maximum number of items to sample.
    sample_prob: Probability for the geometric sampling.

  Returns:
    A list of sampled items.
  """
  pool = list(pool)
  sampled = []
  for _ in range(max_num_items):
    if not pool:
      break
    i = 0
    while random.random() > sample_prob and i < len(pool) - 1:
      i += 1
    sampled.append(pool.pop(i))
  return sampled

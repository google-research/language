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
"""Split dataset tsv file into two separate files based on templates.

This generates template splits, also referred to as query splits, from:
https://arxiv.org/abs/1806.09029
"""

import collections
import random


def split_by_template(examples, template_fn, max_num_examples_1, seed=0):
  """Split examples into two sets with disjoint templates.

  Args:
    examples: Collection of (source, target) examples.
    template_fn: Function mapping target strings to template strings.
    max_num_examples_1: Maximum number of examples to include in examples_1. The
      script will ensure that less than or equal to this number of examples are
      added to examples_1, but cannot gaurantee meeting this exact number,
      especially when there are many examples per template.
    seed: Random seed.

  Returns:
    (examples_1, examples_2), each containing a subset of input examples.
  """
  template_to_examples = collections.defaultdict(list)
  for example in examples:
    target = example[1]
    template = template_fn(target)
    template_to_examples[template].append(example)

  templates = list(template_to_examples.keys())
  random.seed(seed)
  random.shuffle(templates)

  examples_1 = []
  examples_2 = []

  # Start adding examples to examples_1 until this maximum is reached.
  add_to_1 = True

  # Iterate through each template.
  for template in templates:
    # Get examples associated with the given template.
    template_examples = template_to_examples[template]

    if len(examples_1) + len(template_examples) > max_num_examples_1:
      # Continuing to add to examples_1 will exceed our maximum example target.
      # Start adding to examples_2.
      add_to_1 = False

    if add_to_1:
      examples_1.extend(template_examples)
    else:
      examples_2.extend(template_examples)
  return examples_1, examples_2

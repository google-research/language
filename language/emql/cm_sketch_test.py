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
"""Tests for emql.cm_sketch."""

from language.emql import cm_sketch
import numpy as np
import tensorflow.compat.v1 as tf


class CmSketchTest(tf.test.TestCase):

  def setUp(self):
    super(CmSketchTest, self).setUp()
    self.xs = np.arange(100)
    self.cm_context = cm_sketch.CountMinContext(width=1000, depth=17)
    self.sketch = self.cm_context.get_sketch(self.xs)
    self.cm_context.add_set(self.sketch, self.xs)
    self.new_xs = np.arange(50, 200)
    self.new_sketch = self.cm_context.get_sketch(self.new_xs)

  def test_contain(self):
    self.assertTrue(self.cm_context.contain(self.sketch, 50))
    self.assertFalse(self.cm_context.contain(self.sketch, 200))

  def test_intersection(self):
    intersection_sketch = self.cm_context.intersection(
        self.sketch, self.new_sketch)
    self.assertTrue(self.cm_context.contain(intersection_sketch, 80))
    self.assertFalse(self.cm_context.contain(intersection_sketch, 10))
    self.assertFalse(self.cm_context.contain(intersection_sketch, 150))

  def test_union(self):
    union_sketch = self.cm_context.union(self.sketch, self.new_sketch)
    self.assertTrue(self.cm_context.contain(union_sketch, 80))
    self.assertTrue(self.cm_context.contain(union_sketch, 10))
    self.assertTrue(self.cm_context.contain(union_sketch, 150))
    self.assertFalse(self.cm_context.contain(union_sketch, 300))


if __name__ == '__main__':
  tf.test.main()

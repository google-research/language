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
"""Tests for language.search_agents.muzero.utils.py."""

from language.search_agents.muzero import utils

import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_escape_for_lucene(self):
    self.assertEqual(utils.escape_for_lucene("foo:bar-baz"), "foo\\:bar\\-baz")


if __name__ == "__main__":
  tf.test.main()

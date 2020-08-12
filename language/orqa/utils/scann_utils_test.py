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
"""Tests for scann_utils.py."""
import os

from language.orqa.utils import scann_utils
import numpy as np
import tensorflow.compat.v1 as tf


class ScannUtilsTest(tf.test.TestCase):

  def test_scann_searcher(self):
    temp_dir = self.create_tempdir().full_path
    checkpoint_path = os.path.join(temp_dir, "dummy_db.ckpt")

    dummy_db = np.random.uniform(size=[1024, 32]).astype(np.float32)
    scann_utils.write_array_to_checkpoint("dummy_db", dummy_db, checkpoint_path)

    dummy_queries = np.random.uniform(size=[4, 32]).astype(np.float32)
    _, searcher = scann_utils.load_scann_searcher(
        var_name="dummy_db", checkpoint_path=checkpoint_path, num_neighbors=10)
    distance, index = searcher.search_batched(dummy_queries)
    self.assertAllEqual(distance.numpy().shape, [4, 10])
    self.assertAllEqual(index.numpy().shape, [4, 10])


if __name__ == "__main__":
  tf.test.main()

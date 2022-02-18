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
"""Tests for predict."""

import tempfile

from language.gscan.xattn_model import predict
from language.gscan.xattn_model import test_utils
import tensorflow as tf


class PredictTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_train_and_evaluate(self):
    config = test_utils.get_test_config()
    # Create a temporary directory where tensorboard metrics are written.
    workdir = tempfile.mkdtemp()
    predict.predict_and_evaluate(workdir=workdir, config=config)


if __name__ == '__main__':
  tf.test.main()

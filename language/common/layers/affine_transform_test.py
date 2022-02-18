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
"""Tests for language.common.layers.affine_transform."""
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils as keras_test_utils
from keras.utils.generic_utils import CustomObjectScope

from language.common.layers import affine_transform

import tensorflow as tf


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@test_combinations.run_all_keras_modes
class AffineTransformTest(test_combinations.TestCase):

  def test_layer_api_compatibility(self):
    input_array = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                               [2.0, 3.0, 5.0]])

    cls = affine_transform.AffineTransform
    with CustomObjectScope({cls.__name__: cls}):
      output = keras_test_utils.layer_test(
          cls,
          kwargs={
              'output_size': 1,
              'initializer': tf.keras.initializers.TruncatedNormal(stddev=0.02)
          },
          input_shape=(None),
          input_data=input_array)

    expected_values = tf.constant([[0.01368301], [0.01368301], [0.0314441]])
    self.assertAllClose(expected_values, output)


if __name__ == '__main__':
  tf.test.main()

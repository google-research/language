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
"""A Keras layer that implements an affine transform."""
import tensorflow as tf


class AffineTransform(tf.keras.layers.Layer):
  """"Applies an affine transform (i.e. y = Wx+b) to the input_tensor."""

  def __init__(self, output_size,
               initializer, **kwargs):
    super(AffineTransform, self).__init__(**kwargs)

    self._output_size = output_size
    self._initializer = tf.keras.initializers.get(initializer)

  def get_config(self):
    config = {
        'output_size': self._output_size,
        'initializer': tf.keras.initializers.serialize(self._initializer)
    }
    base_config = super(AffineTransform, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    super(AffineTransform, self).build(input_shape)
    input_size = input_shape[-1]
    self._scale = self.add_weight(
        name='output_scale',
        shape=[self._output_size, input_size],
        initializer=self._initializer)
    self._bias = self.add_weight(
        name='output_bias',
        shape=[self._output_size],
        initializer=tf.zeros_initializer())

  def compute_output_shape(self, input_shape):
    output_shape = input_shape.as_list()
    output_shape[-1] = self._output_size
    return tf.TensorShape(output_shape)

  def call(self, inputs):
    output_tensor = tf.matmul(inputs, self._scale, transpose_b=True)
    output_tensor = tf.nn.bias_add(output_tensor, self._bias)
    return output_tensor

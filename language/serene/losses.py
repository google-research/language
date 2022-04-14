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
"""More customizable loss functions than TF/keras provide."""
import tensorflow.compat.v2 as tf

_EPSILON = 1e-7


# Useful for setting a loss to zero when it is positionally required.
# For example, in keras if you provide an output/metric, there must be a
# corresponding loss. This helps satisfy this constraint in these situations
class ZeroLoss(tf.keras.losses.Loss):

  def __init__(self):
    super().__init__(name='zero_loss')

  def call(self, y_true, y_pred):
    return tf.constant(0.0, dtype=y_pred.dtype.base_dtype)


class WeightedBinaryCrossentropyFromProbs(tf.keras.losses.Loss):
  """Binary cross entropy (from probs) with support for class weights."""

  def __init__(self,
               positive_class_weight=None,
               negative_class_weight=None,
               name='binary_crossentropy'):
    super().__init__(name=name)
    self._positive_class_weight = positive_class_weight
    self._negative_class_weight = negative_class_weight

  def call(self, y_true, y_pred):
    return weighted_binary_cross_entropy_from_probs(
        y_true,
        y_pred,
        positive_class_weight=self._positive_class_weight,
        negative_class_weight=self._negative_class_weight,
    )

  def get_config(self):
    conf = super().get_config()
    conf.update({
        'positive_class_weight': self._positive_class_weight,
        'negative_class_weight': self._negative_class_weight,
    })
    return conf


def weighted_binary_cross_entropy_from_probs(target,
                                             output,
                                             positive_class_weight=None,
                                             negative_class_weight=None):
  """Compute the binary cross entropy weighted by the given weights.

  Args:
    target: Prediction target
    output: Model prediction
    positive_class_weight: Weight to give positive class
    negative_class_weight: Weight to give negative class

  Returns:
    Batch loss
  """
  epsilon = tf.constant(_EPSILON, dtype=output.dtype.base_dtype)
  output = tf.clip_by_value(output, epsilon, 1. - epsilon)
  if positive_class_weight is None:
    positive_class_weight = tf.constant(1.0, dtype=output.dtype.base_dtype)
  else:
    positive_class_weight = tf.constant(
        positive_class_weight, dtype=output.dtype.base_dtype)

  if negative_class_weight is None:
    negative_class_weight = tf.constant(1.0, dtype=output.dtype.base_dtype)
  else:
    negative_class_weight = tf.constant(
        negative_class_weight, dtype=output.dtype.base_dtype)

  # Compute cross entropy from probabilities.
  target = tf.cast(target, output.dtype.base_dtype)
  bce = positive_class_weight * target * tf.math.log(output + epsilon)
  bce += negative_class_weight * (1 - target) * tf.math.log(1 - output +
                                                            epsilon)
  return -bce

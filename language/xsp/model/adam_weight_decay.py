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
"""Adam optimizer with weight decay used for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from six.moves import zip
import tensorflow.compat.v1 as tf


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               pretrained_param_names=None,
               freeze_pretrained_steps=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.pretrained_param_names = pretrained_param_names
    self.freeze_pretrained_steps = freeze_pretrained_steps

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []

    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) +
          tf.multiply(1.0 - self.beta_2, tf.square(grad)))

      # TODO(petershaw): Experiment with bias correction.
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want to decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        # Note update will be subtracted from parameter value.
        update += self.weight_decay_rate * param

      # Optionally freeze pre-trained parameters for an initial number of steps.
      if (self.pretrained_param_names and
          param_name in self.pretrained_param_names):
        global_step = tf.train.get_global_step()
        update *= tf.to_float(global_step > self.freeze_pretrained_steps)

      update_with_lr = self.learning_rate * update

      tf.summary.scalar("updates/{}".format(param_name),
                        tf.reduce_mean(update_with_lr))

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      # Any substring match will prevent the parameter from using weight decay.
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def create_optimizer(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     pretrained_param_names,
                     freeze_pretrained_steps,
                     restart_warmup_after_unfreeze=True,
                     lr_after_restarting=0.):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32)

  num_train_steps_int = tf.constant(num_train_steps, dtype=tf.int32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  current_step_in_decay = global_steps_int - warmup_steps_int
  num_decay_steps = num_train_steps_int - warmup_steps_int

  global_steps_float = tf.cast(global_steps_int, tf.float32)

  if freeze_pretrained_steps and restart_warmup_after_unfreeze:
    freeze_pretrained_steps_int = tf.cast(freeze_pretrained_steps, tf.int32)
    global_steps_int -= (
        tf.cast(global_steps_int >= freeze_pretrained_steps_int, tf.int32) *
        freeze_pretrained_steps_int)
    if lr_after_restarting <= 0.:
      raise ValueError("Learning rate after restarting should not be zero: " +
                       str(lr_after_restarting))
    learning_rate = tf.cond(global_step < freeze_pretrained_steps,
                            lambda: init_lr, lambda: lr_after_restarting)

    current_step_in_decay = tf.cond(global_step < freeze_pretrained_steps,
                                    lambda: current_step_in_decay,
                                    lambda: global_steps_int - warmup_steps_int)

    after_unfreeze_decay_steps = num_train_steps_int - (
        freeze_pretrained_steps + warmup_steps_int)

    num_decay_steps = tf.cond(global_step < freeze_pretrained_steps,
                              lambda: num_decay_steps,
                              lambda: after_unfreeze_decay_steps)

    after_unfreeze_steps = global_steps_float - tf.cast(
        freeze_pretrained_steps_int, tf.float32)
    global_steps_float = tf.cond(global_step < freeze_pretrained_steps,
                                 lambda: global_steps_float,
                                 lambda: after_unfreeze_steps)

    tf.summary.scalar("is pretraining",
                      tf.cast(global_step < freeze_pretrained_steps, tf.int32))
  else:
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  tf.summary.scalar("global step count", global_steps_float)
  tf.summary.scalar("current base learning rate", learning_rate)
  tf.summary.scalar("global decay step", current_step_in_decay)
  tf.summary.scalar("total decay steps", num_decay_steps)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      tf.cast(current_step_in_decay, tf.float32),
      tf.cast(num_decay_steps, tf.float32),
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  tf.summary.scalar("decayed learning rate", learning_rate)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float

    tf.summary.scalar("warmup percent done", warmup_percent_done)

    warmup_learning_rate = learning_rate * warmup_percent_done

    is_warmup = global_steps_int < warmup_steps_int

    tf.summary.scalar("is warmup", tf.cast(is_warmup, tf.float32))
    learning_rate = tf.cond(is_warmup, lambda: warmup_learning_rate,
                            lambda: learning_rate)

  tf.summary.scalar("learning rate", learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
      pretrained_param_names=pretrained_param_names,
      freeze_pretrained_steps=freeze_pretrained_steps)

  if use_tpu:
    optimizer = tf.estimator.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      list(zip(grads, tvars)), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


def build_train_op(loss, model_config, use_tpu):
  """Creates the training operation."""
  return create_optimizer(
      loss,
      model_config.training_options.optimizer_learning_rate,
      model_config.training_options.training_steps,
      model_config.training_options.optimizer_warmup_steps,
      use_tpu,
      freeze_pretrained_steps=None,
      pretrained_param_names=None)


def build_train_op_with_pretraining(loss, model_config,
                                    pretrained_variable_names, use_tpu):
  return create_optimizer(
      loss,
      model_config.training_options.optimizer_learning_rate,
      model_config.training_options.training_steps,
      model_config.training_options.optimizer_warmup_steps,
      use_tpu,
      pretrained_param_names=pretrained_variable_names,
      freeze_pretrained_steps=model_config.training_options
      .freeze_pretrained_steps,
      lr_after_restarting=model_config.training_options
      .after_restart_learning_rate)

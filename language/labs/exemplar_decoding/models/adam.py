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
r"""Customized Adam.

   The key difference from the built-in AdamOptimizer is how weight_decay is
   handeled. In the built-in Adam, `\ell_2` penalty is included into momentum
   terms; while here `\ell_2` is separated from the loss, and directly applied
   to the parameters. More details: `https://arxiv.org/abs/1711.05101`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import tensorflow as tf


def warmup_cosine(x, warmup=0.002):
  s = tf.cast(x <= warmup, tf.float32)
  return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
  s = tf.cast(x <= warmup, tf.float32)
  return s*(x/warmup) + (1-s)*1


def warmup_linear(x, warmup=0.002):
  s = tf.cast(x <= warmup, tf.float32)
  return (s*(x/warmup) + (1-s))*(1-x)

schedules = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
}


def adam(params, grads, lr, schedule, t_total,
         b1=0.9,
         b2=0.999,
         e=1e-8,
         weight_decay=1e-2,
         bias_l2=True,
         max_grad_norm=1.):
  """Custom Adam optimzizer for weight decay and learning rate schedule.

  Implementation adapted from https://github.com/openai/finetune-transformer-lm.

  Args:
    params: Parameters to be optimzed.
    grads: Gradients.
    lr: learning rate.
    schedule: Type of learning rate scheduling
    t_total: Total training steps.
    b1: beta_1.
    b2: beta_2.
    e: epsilon.
    weight_decay: Weight decay coefficient.
    bias_l2: Pose l2 penalty on bias parameters or not.
    max_grad_norm: Norm of gradient ot be clipped to.

  Returns:
    A list of update operations.
  """
  t = tf.train.get_global_step()
  tt = t + 1
  updates = [t.assign(tt)]
  if max_grad_norm > 0:
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
  for p, g in zip(params, grads):
    if p is None or g is None:
      print("can't train", p.name, g)
    else:
      if isinstance(g, tf.IndexedSlices):
        g = tf.convert_to_tensor(g)

      # past 1st moment vector; same shape as p.
      m = tf.Variable(p * 0., dtype=tf.float32, trainable=False)

      # past 2nd moment vector; same shape as p.
      v = tf.Variable(p * 0., dtype=tf.float32, trainable=False)
      lrt = lr * tf.sqrt(1 - b2**(tf.cast(tt, tf.float32))) / \
          (1 - b1**(tf.cast(tt, tf.float32)))
      lrt *= schedule(tf.cast(t, tf.float32)/t_total)

      # new 1st moment vector; same shape as p.
      mt = b1 * m + (1 - b1) * g

      # new 2nd moment vector; same shape as p.
      vt = b2 * v + (1 - b2) * g * g

      if (len(p.get_shape()) > 1 or bias_l2) and weight_decay > 0:
        pt = p - lrt * (mt / (tf.sqrt(vt) + e) + weight_decay * p)
      else:
        pt = p - lrt * (mt / (tf.sqrt(vt) + e))
      updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
  return tf.group(*updates)

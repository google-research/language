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
"""Learning rate scheduler and other optimizer utils."""

from typing import Callable

import flax
import jax.numpy as jnp


def create_learning_rate_scheduler(
    learning_rate: float,
    warmup: bool = False,
    warmup_steps: int = 1000,
    linear_decay: bool = False,
    max_steps: int = 500000,
    decay_minimum_factor: float = 0.0,
) -> Callable[[int], float]:
  """Creates learning rate scheduler with options for warmup and linear decay.

  Args:
    learning_rate: base learning rate to be modified.
    warmup: if true, applies learning rate warmup.
    warmup_steps: number of warmup steps.
    linear_decay: if true, applies linear learning rate decay.
    max_steps: number of steps after which learning rate is fully decayed.
    decay_minimum_factor: proportion of lr left after max_steps.

  Returns:
    Function that maps from step to lr.
  """

  def step_fn(step: int) -> float:
    factor = 1.0
    if warmup:
      factor *= jnp.minimum(1.0, step / warmup_steps)
    if linear_decay:
      step_decay = (1.0 - decay_minimum_factor) / (max_steps - warmup_steps)
      factor *= jnp.minimum(1.0, 1.0 - step_decay * (step - warmup_steps))
    return jnp.asarray(learning_rate * factor)

  return step_fn


def create_dict_mask(input_dict, mask_keys):
  flattened_dict = flax.traverse_util.flatten_dict(input_dict)
  flattened_mask_dict = {
      key: not any([mask_key in ''.join(key) for mask_key in mask_keys
                   ]) for key in flattened_dict
  }
  mask_dict = flax.traverse_util.unflatten_dict(flattened_mask_dict)
  return mask_dict

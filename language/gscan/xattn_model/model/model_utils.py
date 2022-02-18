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
"""Util funtion for modeling."""

import jax.numpy as jnp


def shift_right(x, axis=1):
  """Shift the input to the right by padding in the end on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  padded = jnp.pad(
      x[:, 1:], pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded


def shift_left(x, axis=1):
  """Shift the input to the left by padding in the front on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  padded = jnp.pad(
      x[:, :-1], pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded

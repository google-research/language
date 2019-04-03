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
"""Base functionality ofr modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow as tf


__all__ = ["AbstractNMTModule"]


@six.add_metaclass(abc.ABCMeta)
class AbstractNMTModule(object):
  """Abstract base class for neural machine translation modules."""

  def __init__(self, name):
    """Creates a new NMT module.

    Args:
      name: String used as the scope name of the module's subgraph.
    """
    self.name = name

  def __call__(self, reuse=None, **kwargs):
    with tf.variable_scope(self.name, reuse=reuse):
      outputs = self._build(**kwargs)
    return outputs

  @abc.abstractmethod
  def _build(self, **kwargs):
    """Must be implemented by a subclass."""
    raise NotImplementedError("Abstract Method")

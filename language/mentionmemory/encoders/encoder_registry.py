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
"""Contains registry and registration function for tasks and encoders."""



from language.mentionmemory.encoders import base_encoder

_ENCODER_REGISTRY = {}

_BaseEncoderVar = TypeVar('_BaseEncoderVar', bound='base_encoder.BaseEncoder')


def register_encoder(
    name):
  """Register encoder.

  Encoder should implement BaseEncoder abstraction. Used as decorator, for
  example:

  @register_encoder('my_encoder')
  class MyEncoder(BaseEncoder):

  Args:
    name: name of registered encoder.

  Returns:
    Mapping from BaseEncoder to BaseEncoder.
  """

  def _wrap(cls):
    """Decorator inner wrapper needed to support `name` argument."""
    if not issubclass(cls, base_encoder.BaseEncoder):
      raise TypeError(
          'Invalid encoder. Encoder %s does not subclass BaseEncoder.' %
          cls.__name__)

    if name in _ENCODER_REGISTRY:
      raise ValueError(
          'Encoder name %s has already been registered with class %s' %
          (name, _ENCODER_REGISTRY[name].__name__))

    _ENCODER_REGISTRY[name] = cls

    return cls

  return _wrap


def get_registered_encoder(name):
  """Takes in encoder name and returns corresponding encoder from registry."""
  return _ENCODER_REGISTRY[name]

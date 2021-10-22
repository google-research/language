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
"""Tests for registry."""

from absl.testing import absltest

from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry


@encoder_registry.register_encoder('decorated_encoder')
class DecoratedEncoder(base_encoder.BaseEncoder):
  pass


class UnDecoratedEncoder(base_encoder.BaseEncoder):
  pass


class InvalidEncoder(object):
  pass


class EncoderRegistryTest(absltest.TestCase):

  def test_decorated_encoder(self):
    """Simple test to verify that decorated encoders have been registered."""
    encoder_name = encoder_registry.get_registered_encoder('decorated_encoder')
    self.assertIsNotNone(encoder_name)
    self.assertEqual(encoder_name.__name__, 'DecoratedEncoder')

  def test_undecorated_encoder(self):
    """Simple test to verify that we can register encoders at runtime."""
    # Register the encoder.
    encoder_registry.register_encoder('undecorated_encoder')(UnDecoratedEncoder)

    # Retrieve it.
    encoder_name = encoder_registry.get_registered_encoder(
        'undecorated_encoder')
    self.assertIsNotNone(encoder_name)
    self.assertEqual(encoder_name.__name__, 'UnDecoratedEncoder')

    # Verify that we can still access previously registerd decorated layers.
    encoder_name = encoder_registry.get_registered_encoder('decorated_encoder')
    self.assertIsNotNone(encoder_name)
    self.assertEqual(encoder_name.__name__, 'DecoratedEncoder')

  def test_invalid_encoder(self):
    """Verify we get an exception when trying to register invalid encoder."""
    with self.assertRaises(TypeError):
      encoder_registry.register_encoder('invalid_encoder')(InvalidEncoder)  # pytype: disable=wrong-arg-types

  def test_multiple_encoder_registrations(self):
    """Verify that re-using an already registered name raises an exception."""
    with self.assertRaises(ValueError):
      encoder_registry.register_encoder('decorated_encoder')(UnDecoratedEncoder)


if __name__ == '__main__':
  absltest.main()

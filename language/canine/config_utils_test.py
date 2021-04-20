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
import json

from language.canine import config_utils
import tensorflow.compat.v1 as tf


class SimpleInnerConfigObject(config_utils.Config):

  def __init__(self, meaning_of_life=42, pi=3.14):
    self.meaning_of_life = meaning_of_life
    self.pi = pi


class SimpleConfigObject(config_utils.Config):

  def __init__(
      self,
      hello="world",
      one=1,
      truthy=False,
      nested_thing="not_actually_nested",
      nested=SimpleInnerConfigObject(),
      deprecated_feature="",
  ):
    self.hello = hello
    self.one = one
    self.truthy = truthy
    self.nested_thing = nested_thing
    self.nested = nested
    self._private = object()  # Unserializable by JSON.
    self.deprecated_feature = "x"  # Deserializable from 'feature'.


class UsesProperties(config_utils.Config):

  def __init__(self, hello="world", meaning="of life"):
    self.hello = hello
    self.meaning = meaning

  @property
  def hello(self):
    # Return a modified form.
    return self._hello + "!"

  @hello.setter
  def hello(self, value):
    # Store the stored value directly.
    self._hello = value

  @property
  def meaning(self):
    # Return the stored value directly.
    return self._meaning

  @meaning.setter
  def meaning(self, value):
    # Store a modified form.
    self._meaning = value.upper()


class ConfigTest(tf.test.TestCase):

  def test_simple_config_object_access(self):
    c = SimpleConfigObject()
    self.assertEqual(c.hello, "world")
    self.assertEqual(c.one, 1)
    self.assertEqual(c.nested.meaning_of_life, 42)
    self.assertAlmostEqual(c.nested.pi, 3.14)

  def test_simple_config_with_constructor_arg(self):
    c = SimpleConfigObject(hello="there")
    self.assertEqual(c.hello, "there")

  def test_simple_config_object_access_error(self):
    c = SimpleConfigObject()
    # pylint: disable=no-member
    self.assertRaises(Exception, lambda: c.not_defined)
    self.assertRaises(Exception, lambda: c.nested.not_defined)
    # pylint: enable=no-member

  def test_simple_config_object_from_simple_dict(self):
    d = {
        "hello": "universe",
        "one": 2,
        "nested_thing": "please don't try to insert into nested",
    }
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.hello, "universe")
    self.assertEqual(c.one, 2)
    self.assertEqual(c.nested_thing, "please don't try to insert into nested")

  def test_simple_config_object_from_namespace_dict(self):
    d = {
        "nested.meaning_of_life": 43,
        "nested.pi": 3.33,
    }
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.nested.meaning_of_life, 43)
    self.assertAlmostEqual(c.nested.pi, 3.33)

  def test_simple_config_object_from_simple_dict_undefined(self):
    with self.assertRaises(ValueError):
      SimpleConfigObject.from_dict({"not_defined": -1})
    with self.assertRaises(ValueError):
      SimpleConfigObject.from_dict({"nested.not_defined": -1})

  def test_nested_json_serialization(self):
    config = SimpleConfigObject()
    obj = json.loads(config.to_json_string())
    tf.logging.info(str(obj))
    self.assertEqual(obj["one"], 1)
    self.assertEqual(obj["hello"], "world")
    self.assertEqual(obj["nested.meaning_of_life"], 42)
    self.assertAlmostEqual(obj["nested.pi"], 3.14)

  def test_private_field_serialize(self):
    config = SimpleConfigObject()
    json_object = json.loads(config.to_json_string())
    self.assertNotIn("_private", json_object)

  def test_private_field_deserialize(self):
    d = {"_private": "should fail"}
    with self.assertRaises(ValueError):
      SimpleConfigObject.from_dict(d)

  def test_deprecated_field_deserialize(self):
    d = {"feature": "x"}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.deprecated_feature, "x")

    d = {"deprecated_feature": "x"}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.deprecated_feature, "x")

  def test_properties_defaults(self):
    config = UsesProperties()
    self.assertEqual(config.hello, "world!")
    self.assertEqual(config.meaning, "OF LIFE")

  def test_properties_constructor_args(self):
    config = UsesProperties(hello="first", meaning="second")
    self.assertEqual(config.hello, "first!")
    self.assertEqual(config.meaning, "SECOND")

  def test_properties_set_fields(self):
    config = UsesProperties()
    config.hello = "first"
    config.meaning = "second"
    self.assertEqual(config.hello, "first!")
    self.assertEqual(config.meaning, "SECOND")

  def test_properties_serialize(self):
    config = UsesProperties(hello="first", meaning="second")
    json_object = json.loads(config.to_json_string())
    self.assertLen(json_object, 2)
    self.assertEqual(json_object["hello"], "first!")
    self.assertEqual(json_object["meaning"], "SECOND")

  def test_properties_deserialize(self):
    d = {"hello": "third", "meaning": "fourth"}
    c = UsesProperties.from_dict(d)
    self.assertEqual(c.hello, "third!")
    self.assertEqual(c.meaning, "FOURTH")

  def test_bool_true(self):
    d = {"truthy": True}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.truthy, True)

  def test_bool_false(self):
    d = {"truthy": False}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.truthy, False)

  def test_bool_true_string(self):
    d = {"truthy": "true"}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.truthy, True)

  def test_bool_false_string(self):
    d = {"truthy": "false"}
    c = SimpleConfigObject.from_dict(d)
    self.assertEqual(c.truthy, False)


if __name__ == "__main__":
  tf.test.main()

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
"""Flexible python configuration class with object-based namespaces.

See `Config` class for details.
"""

import inspect
import json
import traceback



import dataclasses
import tensorflow.compat.v1 as tf

ConfigT = TypeVar("ConfigT", bound="Config")

JsonAtomicValue = Union[Text, int, float, bool, None]


class Config:
  """Represents config as nested object structure with JSON I/O.

  Analogous to bert.modeling.BertConfig, but:
  1) Requires any deserialized field name to exactly match the name of an
     already-defined field (to prevent typos).
  2) Allows field names that are objects to be deserialized from namespaces.

  This enables nesting config classes, which is useful when one model may rely
  on another model and so should expose the child model's configuration as part
  of its own -- preferably without having to clone-and-modify that config.
  However, we still want to retain the ability to read and write from a flat
  key-value store to remain compatible with flags-based hyperparamter storage
  and other hyperparameter tuning tooling (e.g. Vizier).

  Assumptions about inheriting classes:
  * Inheriting classes must implement a zero-arg constructor (typically, a
    constructor whose arguments all have defaults. This is used when
    deserializing.
  * Any attributes that should be interpreted as children in the config
    hierarchy must also inherit from `Config`.

  Private fields (starting with an underscore) will not be
  serialized/deserialized.

  Deprecated fields may be marked by prefixing with the string 'deprecated_' to
  visually indicate that they're present exclusively for backward compatibility.
  However, the field can still be deserialized with its previous name. For
  example, 'deprecated_feature' can be deserialized as 'feature' so that old
  configs do not become invalid, as long as code preserves old behavior.

  Example usage:

  ```
  class SimpleInnerConfigObject(config_utils.Config):

    def __init__(self, meaning_of_life=42, pi=3.14):
      self.meaning_of_life = meaning_of_life
      self.pi = pi


  class SimpleConfigObject(config_utils.Config):

    def __init__(self, hello="world", one=1, nested=SimpleInnerConfigObject()):
      self.hello = hello
      self.one = one
      self.nested = nested

  def read_my_config_from_dict_example():
    d = {
        "hello": "universe"
        "one": 2,
        "nested.meaning_of_life": 43,
        "nested.pi": 3.33
    }
    c = SimpleConfigObject.from_dict(d)

  def read_my_config_from_json_example():
    c = SimpleConfigObject.from_json_file("/path/to/config.json")

  def create_my_config_programmatically():
    c = SimpleConfigObject(hello="there")
    c.nested.meaning_of_life = 43
  ```

  See also `config_utils_test.py`.
  """

  def validate(self):
    """Raises an exception if this configuration is invalid."""
    # Run any custom validation logic provided by a subclass's implementation.
    self._validate_self()

    # Validate any sub-configs.
    for field_name in self._field_names:
      value = getattr(self, field_name)
      if isinstance(value, Config):
        value.validate()

  def _validate_self(self):
    """To be overridden by implementing classes for custom validation logic."""
    pass

  @classmethod
  def from_dict(cls,
                json_object,
                debug_filename = None):
    """Constructs a `Config` from a Python dictionary of parameters."""
    if not issubclass(cls, Config):
      raise ValueError(
          f"Expected `cls` to be a `Config` class but found '{cls.__name__}'")

    # Create instance of derived class using zero-arg constructor
    config = cls()
    for key, value in json_object.items():
      # Allow special comment key within JSON to make them more readable.
      if key.upper() == "__COMMENT__":
        continue
      # pylint: disable=protected-access
      config._set_value(
          key, value, debug_filename=debug_filename, debug_orig_key=key)
      # pylint: enable=protected-access
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    if not issubclass(cls, Config):
      raise ValueError(
          f"Expected `cls` to be a `Config` class but found '{cls.__name__}'")

    with tf.gfile.GFile(json_file, "r") as f:
      return cls.from_dict(json.load(f), debug_filename=json_file)

  def to_dict(self, namespace_prefix = ""):
    """Serializes this instance to a Python dictionary."""
    assert not namespace_prefix or namespace_prefix.endswith(".")

    output = dict()
    for key in self._field_names:
      value = getattr(self, key)
      if isinstance(value, Config):
        child_dict = value.to_dict(
            namespace_prefix=(namespace_prefix + key + "."))
        output.update(child_dict)
      else:
        output[namespace_prefix + key] = value
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    try:
      return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    except TypeError:
      e_str = traceback.format_exc()
      raise TypeError(
          "Could not JSON serialize a child attribute. "
          "Did you forget to make it an instance of `Config`?", e_str)

  def __str__(self):
    return self.to_json_string()

  # Private members:

  @property
  def _field_names(self):
    """Returns this config's field names as a set."""
    if dataclasses.is_dataclass(self):
      # Dataclasses have their own way of storing information about their
      # fields, so use that.
      field_names = {field.name for field in dataclasses.fields(self)}
    else:
      field_names = set(vars(self))
      field_names.update(name for name, _ in (
          inspect.getmembers(type(self), lambda m: isinstance(m, property))))

    return {
        field_name for field_name in field_names
        if not field_name.startswith("_")  # Skip private fields.
    }

  def _cast_value(self, target, value):
    # Prevent implicit casting "false" to `True`.
    if isinstance(target, bool) and isinstance(value, str):
      if value.lower() == "true":
        value = True
      elif value.lower() == "false":
        value = False
    return value

  def _set_value(self, key, value, debug_filename,
                 debug_orig_key):
    """Sets the value of a key-value pair, recursively traversing namespaces."""
    # Detect if this key has a namespace prefix.
    # If so, strip it and recursively delegate to corresponding child object.
    key_parts = key.split(".", 1)
    field_name = key_parts[0]

    # Allow deprecated fields to be suffixed with an underscore.
    if (field_name not in self._field_names and
        "deprecated_" + field_name in self._field_names):
      field_name = "deprecated_" + field_name

    if field_name not in self._field_names:
      error_msg = (f"Config field '{field_name}' is not defined in class "
                   f"'{type(self).__name__}' (specified key='{debug_orig_key}';"
                   f" defined keys=[{','.join(self._field_names)}])")
      if debug_filename:
        error_msg += f" (see file '{debug_filename}')"
      raise ValueError(error_msg)

    if len(key_parts) == 2:
      subkey = key_parts[1]
      child = getattr(self, field_name)
      if not isinstance(child, Config):
        raise ValueError(f"Expected `{debug_orig_key}` to be a `Config` class "
                         f"but found '{type(child).__name__}'")
      value = self._cast_value(target=child, value=value)
      # pylint: disable=protected-access
      child._set_value(
          subkey,
          value,
          debug_filename=debug_filename,
          debug_orig_key=debug_orig_key)
      # pylint: enable=protected-access
      return

    target = getattr(self, field_name)
    value = self._cast_value(target=target, value=value)

    # Otherwise, we're at the base case and can directly set the attribute.
    assert len(key_parts) == 1
    setattr(self, key, value)

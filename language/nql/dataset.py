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
"""Dataset object, which makes it easier to train NQL."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from language.nql import nql
import numpy as np
import tensorflow as tf



def tuple_generator_builder(context,
                            tuple_input,
                            type_specs,
                            normalize_outputs = True,
                            field_separator = '\t',
                            entity_separator = ' || '
                           ):
  """Create iterator over tuples produced by parsing lines from a file.

  The lines are delimited by field_separator, with each being a different type
  of feature.  By convention the last field is the desired output of the model
  given the first n-1 fields as input, and if normalize_outputs is
  True, then this field will be L1-normalized.

  The types of each field are given by the list of type_specs.  The possible
  type_specs are

    1) The string name of an entity type declared in context.  In this case the
  corresponding part of 'lines' should be a set of entity names, of the provided
  type, separated by the 'entity_separator' string.  This will be converted to a
  k-hot representation of that set of entities.

    2) The python type str.  In this case the corresponding part of 'lines' will
  be passed on as a tf.string.

  Args:
    context: a NeuralQueryContext
    tuple_input: Either a filename or an iterater over lines of data.
    type_specs: list of specifications for how to parse each tab-separated field
      of the line.  The possible specifications are listed above.
    normalize_outputs: treat the last line as a label and L1-normalize it
    field_separator: string to separate fields
    entity_separator: string to separate entity names

  Returns:
    a function taking no arguments that returns an Iterable[Tuple[Any]]

  Raises:
    ValueError, for incorrectly formatted lines
  """

  def tuple_generator():
    """Closure produced by tuple_generator_builder."""
    line_iter = tf.io.gfile.GFile(tuple_input) if isinstance(
        tuple_input, str) else tuple_input
    for line in line_iter:
      try:
        parts = line.strip().split(field_separator)
        if len(parts) != len(type_specs):
          raise ValueError('line does not have %d fields: %r' %
                           (len(type_specs), line.strip()))
        buf = []
        for i in range(len(parts)):
          spec = type_specs[i]
          if isinstance(spec, str) and context.is_type(spec):
            parsed_val = k_hot_array_from_string_list(
                context, spec, parts[i].split(entity_separator))
          elif spec == str:
            parsed_val = parts[i]
          else:
            raise ValueError('illegal type_spec %r' % spec)
          buf.append(parsed_val)
        if normalize_outputs:
          buf_sum = np.sum(buf[-1])
          if buf_sum:
            buf[-1] /= buf_sum
        yield tuple(buf)
      except (nql.EntityNameError, nql.TypeNameError) as ex:
        tf.logging.warn('Problem %r on line: %r', ex, line.strip())

  return tuple_generator


def tuple_dataset(context,
                  tuple_input,
                  type_specs,
                  normalize_outputs = True,
                  field_separator = '\t',
                  entity_separator = ' || '):
  """Produce a dataset by parsing lines from a file.

  Lines are formatted as described in documents for tuple_generator_builder.

  Args:
    context: passed to tuple_generator_builder
    tuple_input: passed to tuple_generator_builder
    type_specs: passed to tuple_generator_builder
    normalize_outputs: passed to tuple_generator_builder
    field_separator: passed to tuple_generator_builder
    entity_separator: passed to tuple_generator_builder

  Returns:
    tf.Data.Dataset over tuples, with one component for tab-separated field
  """

  return tf.data.Dataset.from_generator(
      tuple_generator_builder(context, tuple_input, type_specs,
                              normalize_outputs, field_separator,
                              entity_separator),
      tuple([spec_as_tf_type(spec) for spec in type_specs]),
      tuple([spec_as_shape(spec, context) for spec in type_specs]))


def spec_as_tf_type(spec):
  """Convert a type_spec to a tf type.

  Args:
    spec: a single specification for tuple_generator_builder

  Returns:
    type specification required by tf.data.Dataset.from_generator
  """
  if spec == str:
    return tf.string
  elif isinstance(spec, int):
    return tf.int32
  else:
    return tf.float32


def spec_as_shape(spec, context):
  """Convert a type_spec to a tf shape.

  Args:
    spec: a single specification for tuple_generator_builder
    context: a NQL context

  Returns:
    tensor shape specification, as required by tf.data.Dataset.from_generator
  """
  if spec == str:
    return tf.TensorShape([])
  elif isinstance(spec, int):
    return tf.TensorShape([spec])
  else:
    return tf.TensorShape([context.get_max_id(spec)])


# GOOGLE_INTERNAL: TODO(b/124102056) Consider moving into nql.
def k_hot_array_from_string_list(context,
                                 typename,
                                 entity_names):
  """Create a numpy array encoding a k-hot set.

  Args:
    context: a NeuralExpressionContext
    typename: type of entity_names
    entity_names: list of names of type typename

  Returns:
    A k-hot-array representation of the set of entity_names. For frozen
    dictionaries, unknown entity names are mapped to the unknown_id of their
    type or discarded if the unknown_value of the type is None. Unknown entity
    names will throw an nql.EntityNameException for non-frozen dictionaries.
    It is possible for this method to return an all-zeros array.
  """
  # Empty string is not a valid entity_name.
  ids = [context.get_id(e, typename) for e in entity_names if e]
  # None is not a valid id.
  valid_ids = [x for x in ids if x is not None]
  max_id = context.get_max_id(typename)
  result = np.zeros((max_id,), dtype='float32')
  if valid_ids:
    result[valid_ids] = 1.
  return result


def placeholder_for_type(context,
                         type_spec,
                         name = None):
  """Produce a Tensorflow placeholder for this type_spec.

  Args:
    context: a NeuralQueryContext
    type_spec: a single type_spec (see tuple_dataset)
    name: a name to use for the placeholder

  Returns:
    a Tensorflow placeholder

    Raises:
      ValueError, if the type_spec is invalid
  """
  if type_spec == str:
    return tf.placeholder(tf.string, shape=[None], name=name)
  elif isinstance(type_spec, str) and context.is_type(type_spec):
    name = name or ('%s_ph' % type_spec)
    return context.placeholder(name, type_spec).tf
  else:
    raise ValueError('bad type spec %r' % type_spec)


def build_feature_dict_mapper(feature_names):
  """Build a function for tf.data.Dataset.map.

  Args:
    feature_names: List of feature names.

  Returns:
    A function converting tuples into (dictionary of features, label).
  """

  def mapper(*tuple_args):
    d = {}
    for i in range(len(feature_names)):
      d[feature_names[i]] = tuple_args[i]
    return d, tuple_args[-1]

  return mapper

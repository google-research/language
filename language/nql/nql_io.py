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
"""IO Tools for supporting NQL."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from language.nql import nql_symbol
import numpy
import scipy.sparse
import tensorflow as tf



def lines_in(file_like):
  """Enumerate lines in a stream.

  Call file_like's iterator method and iterate over that.  Will
  skip any lines starting with # and blank lines.

  Args:
    file_like: An iterator over lines.

  Yields:
    a string for each line in the file or stream.
  """
  for line in file_like:
    if line[0] != '#' and line != '\n':
      yield line


def lines_in_all(files,
                 lines):
  """Enumerate lines in multiple files.

  Args:
    files: If specified, a file-like or array of file-like objects.
    lines: If specified, an array of strings.

  Yields:
    a string for each line in each file or string.
  """
  if files:
    file_set = files if isinstance(files, list) else [files]
    for one_file in file_set:
      for line in lines_in(one_file):
        yield line
  if lines:
    for line in lines_in(lines):
      yield line


def _read_numpy_item(input_file):
  """Read the item found in a numpy save file.

  Args:
    input_file: Filename string or FileLike object.

  Returns:
    The first item.
  """
  if isinstance(input_file, str):
    with tf.io.gfile.GFile(input_file, 'rb') as fh:
      item = numpy.load(fh).item()
  else:
    item = numpy.load(input_file).item()
  return item


def _write_numpy_item(output_file, item):
  """Write the item to a numpy save file.

  Args:
    output_file: Filename string or FileLike object.
    item: Some object to save as the first item.
  """
  if isinstance(output_file, str):
    with tf.io.gfile.GFile(output_file, 'wb') as fh:
      numpy.save(fh, item)
  else:
    numpy.save(output_file, item)


def read_sparse_matrix_dict(
    input_file):
  """Read a dictionary of relations from a file.

  Args:
    input_file: Filename string or FileLike object.

  Returns:
    A dictionary mapping relation names to scipy sparse matrices.
  """
  relation_dict = _read_numpy_item(input_file)
  return {
      rel_name: _numpy_dict_to_sparse_matrix(numpy_dict)
      for rel_name, numpy_dict in relation_dict.items()
  }


def write_sparse_tensor_dict(
    output_file,
    sparse_tensor_dict):
  """Write a dictionary of tf.SparseTensor values to a file.

  Args:
    output_file: Filename string or FileLike object.
    sparse_tensor_dict: Map from relation name to tf.SparseTensor values.
  """
  relation_dict = {
      rel_name: _sparse_tensor_to_numpy_dict(sparse_tensor)
      for rel_name, sparse_tensor in sparse_tensor_dict.items()
  }
  _write_numpy_item(output_file, relation_dict)


def read_symbol_table_dict(input_file,
                           restrict_to = ()):
  """Read a dictionary of SymbolTable values from a file.

  Args:
    input_file: Filename string or FileLike object.
    restrict_to: If defined, a list of types to restrict to.

  Returns:
    A dictionary mapping type names to SymbolTable values.
  """
  symbol_dict = _read_numpy_item(input_file)
  return {
      type_name: nql_symbol.create_from_dict(symbol_table_dict)
      for type_name, symbol_table_dict in symbol_dict.items()
      if not restrict_to or type_name in restrict_to
  }


def write_symbol_table_dict(
    output_file,
    symbol_table_dict,
    restrict_to = ()):
  """Write a dictionary of SymbolTable values to a file.

  Args:
    output_file: Filename string or FileLike object.
    symbol_table_dict: Map from type_name to SymbolTable values.
    restrict_to: If defined, a list of types to restrict to.
  """
  output_dict = {
      type_name: symbol_table.to_dict()
      for type_name, symbol_table in symbol_table_dict.items()
      if not restrict_to or type_name in restrict_to
  }
  _write_numpy_item(output_file, output_dict)


def _numpy_dict_to_sparse_matrix(
    numpy_dict):
  """Convert a dictionary of numpy arrays into a scipy sparse matrix.

  Args:
    numpy_dict: A dictionary representing the data.

  Returns:
    A scipy sparse matrix representing the data.
  """
  return scipy.sparse.coo_matrix((numpy_dict['values'],
                                  (numpy_dict['rows'], numpy_dict['cols'])),
                                 shape=numpy_dict['shape'],
                                 dtype='float32').transpose()


def _sparse_tensor_to_numpy_dict(
    sparse_tensor):
  """Convert a tf.SparseTensor into a dictionary of numpy arrays.

  Args:
    sparse_tensor: A SparseTensor of the trained relation.

  Returns:
    A dictionary representing the data.
  """
  return {
      'shape': sparse_tensor.dense_shape,
      'rows': numpy.array(sparse_tensor.indices[:, 0]),
      'cols': numpy.array(sparse_tensor.indices[:, 1]),
      'values': numpy.array(sparse_tensor.values)
  }

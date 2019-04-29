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
"""Tests for nql_io."""

import tempfile
from language.nql import nql_io
from language.nql import nql_symbol
import tensorflow as tf


def _equal_sparse(a, b):
  return (a.shape == b.shape) and (a.row == b.row).all() and (
      a.col == b.col).all() and (a.data == b.data).all()


def _make_sparse_tensor_dict():
  rel_name1 = 'real_stuff'
  # Note, these matrices are transposed.
  sparse_tensor1 = tf.SparseTensor(
      indices=[[0, 0], [99, 1]], values=[1., 2.], dense_shape=[100, 2])
  rel_name2 = 'other_stuff'
  sparse_tensor2 = tf.SparseTensor(
      indices=[[100, 0]], values=[3.], dense_shape=[1000, 2])
  return {
      rel_name1: tf.Session().run(sparse_tensor1),
      rel_name2: tf.Session().run(sparse_tensor2)
  }


def _make_symbol_table_dict():
  symbol_table_dict = {}
  type_name1 = 'type_safety'
  symbol_table1 = nql_symbol.SymbolTable()
  for symbol in 'abcdefghijklmnop':
    symbol_table1.insert(symbol)
  symbol_table1.freeze()
  type_name2 = 'fancy_type'
  symbol_table2 = nql_symbol.SymbolTable()
  for symbol in '1234567890':
    symbol_table2.insert(symbol)
  symbol_table2.pad_to_vocab_size(20)
  symbol_table_dict[type_name1] = symbol_table1
  symbol_table_dict[type_name2] = symbol_table2
  return symbol_table_dict


class NQLIOTest(tf.test.TestCase):

  def setUp(self):
    super(NQLIOTest, self).setUp()
    self.filename = tempfile.mktemp()

  def _relations_roundtrip(self, use_fh=False):
    """Test full io pipeline on serializing relations.

    Very basic test that creates two sparse tensors, serializes them,
    deserializes them, then compares them using the transformation
    methods in nql_io.

    Args:
      use_fh: Boolean indicating usage of FileLike instead of Filename.
    """
    sparse_tensor_dict = _make_sparse_tensor_dict()
    if use_fh:
      with open(self.filename, 'wb') as fh:
        nql_io.write_sparse_tensor_dict(fh, sparse_tensor_dict)
      with open(self.filename, 'rb') as fh:
        io_sparse_matrix_dict = nql_io.read_sparse_matrix_dict(fh)
    else:
      nql_io.write_sparse_tensor_dict(self.filename, sparse_tensor_dict)
      io_sparse_matrix_dict = nql_io.read_sparse_matrix_dict(self.filename)
    self.assertEqual(
        len(io_sparse_matrix_dict.keys()), len(sparse_tensor_dict.keys()))
    for rel_name, sparse_tensor in sparse_tensor_dict.items():
      ref_sparse_matrix = nql_io._numpy_dict_to_sparse_matrix(
          nql_io._sparse_tensor_to_numpy_dict(sparse_tensor))
      io_sparse_matrix = io_sparse_matrix_dict[rel_name]
      self.assertTrue(_equal_sparse(io_sparse_matrix, ref_sparse_matrix))

  def test_serialize_relations_using_filenames(self):
    self._relations_roundtrip(False)

  def test_serialize_relations_using_filehandles(self):
    self._relations_roundtrip(True)

  def _symbol_table_dict_roundtrip(self, use_fh=False):
    """Test full io pipeline on serializing relations.

    Very basic test that creates two symbol tables, serializes them,
    deserializes them, then compares them using methods in nql_io.

    Args:
      use_fh: Boolean indicating usage of FileLike instead of Filename.
    """
    symbol_table_dict = _make_symbol_table_dict()
    if use_fh:
      with open(self.filename, 'wb') as fh:
        nql_io.write_symbol_table_dict(fh, symbol_table_dict)
      with open(self.filename, 'rb') as fh:
        io_symbol_table_dict = nql_io.read_symbol_table_dict(fh)
    else:
      nql_io.write_symbol_table_dict(self.filename, symbol_table_dict)
      io_symbol_table_dict = nql_io.read_symbol_table_dict(self.filename)
    for type_name in io_symbol_table_dict.keys():
      self.assertIn(type_name, symbol_table_dict)
    for type_name, t in symbol_table_dict.items():
      io_t = io_symbol_table_dict[type_name]
      self.assertEqual(t.get_unk_id(), io_t.get_unk_id())
      self.assertEqual(t.get_max_id(), io_t.get_max_id())
      self.assertEqual(t._vocab_size, io_t._vocab_size)
      t_symbols = t.get_symbols()
      io_t_symbols = io_t.get_symbols()
      self.assertEqual(t_symbols, io_t_symbols)
      for symbol in t_symbols:
        self.assertEqual(t.get_id(symbol), io_t.get_id(symbol))

  def test_serialize_symbol_table_dict_using_filenames(self):
    self._symbol_table_dict_roundtrip(False)

  def test_serialize_symbol_table_dict_using_filehandles(self):
    self._symbol_table_dict_roundtrip(True)

  def _test_restrict_filtering(self, input_types, restrict_write, restrict_read,
                               expected_output_types):
    """Test restrict behavior on types written and read through tempfile.

    Args:
      input_types: A list of the types to start with.
      restrict_write: A list of the restrict_to to use while writing.
      restrict_read: A list of the restrict_to to use while reading.
      expected_output_types: A list of the types to expect returned.
    """
    symbol_table_dict = {t: nql_symbol.SymbolTable() for t in input_types}
    nql_io.write_symbol_table_dict(
        self.filename,
        symbol_table_dict,
        restrict_to=[r for r in restrict_write])
    io_symbol_table_dict = nql_io.read_symbol_table_dict(
        self.filename, restrict_to=[r for r in restrict_read])
    io_output_types = ''.join(sorted(io_symbol_table_dict.keys()))
    self.assertEqual(io_output_types, expected_output_types)

  def test_serialize_symbol_table_restrict_to(self):
    self._test_restrict_filtering('abcd', '', '', 'abcd')
    self._test_restrict_filtering('abcd', 'ab', '', 'ab')
    self._test_restrict_filtering('abcd', '', 'cd', 'cd')
    self._test_restrict_filtering('abcd', 'abc', 'bcd', 'bc')
    self._test_restrict_filtering('abcd', 'abcxyz', 'bcd123', 'bc')
    self._test_restrict_filtering('abcd', 'xyz', '', '')
    self._test_restrict_filtering('abcd', '', '123', '')


if __name__ == '__main__':
  tf.test.main()

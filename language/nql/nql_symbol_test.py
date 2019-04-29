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
"""Tests for nql_symbol."""

from language.nql import nql_symbol
import tensorflow as tf


class TestSymbolTable(tf.test.TestCase):

  def test_fixed_freeze_none(self):
    tab = nql_symbol.SymbolTable()
    for s in 'abcdefg':
      tab.insert(s)
    tab.freeze(unknown_marker=None)
    self.assertEqual(tab.get_id('Z'), None)

  def test_unk(self):
    tab = nql_symbol.SymbolTable()
    for s in 'abcdefg':
      tab.insert(s)
    self.assertEqual(tab.get_max_id(), 7)
    self.assertEqual(tab.get_unk_id(), None)
    # freezing adds an UNK symbol
    tab.freeze()
    self.assertEqual(tab.get_unk_id(), 7)
    self.assertEqual(tab.get_max_id(), 8)
    # new strings are now UNK'd out
    self.assertEqual(tab.get_id('h'), tab.get_id('z'))
    # even if you insert them
    tab.insert('h')
    self.assertEqual(tab.get_max_id(), 8)
    self.assertEqual(tab.get_id('h'), tab.get_id('z'))
    self.assertEqual(tab.get_id('h'), 7)

  def test_padding(self):
    tab = nql_symbol.SymbolTable()
    for s in 'abcdefg':
      tab.insert(s)
    self.assertTrue(tab.has_id('a'))
    self.assertEqual(tab.get_max_id(), 7)
    tab.pad_to_vocab_size(20)
    self.assertEqual(tab.get_max_id(), 20)
    tab.reset()
    for s in 'tuvwx':
      tab.insert(s)
    self.assertEqual(tab.get_max_id(), 20)
    self.assertTrue(tab.has_id('x'))
    self.assertFalse(tab.has_id('a'))


if __name__ == '__main__':
  tf.test.main()

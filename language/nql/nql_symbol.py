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
"""SymbolTable class for NQL."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function




class SymbolTable(object):
  """Birectionally map strings to integers."""

  def __init__(self):
    """Initialize a SymbolTable."""
    self._symbol_list = []
    self._next_id = 0
    self._id_dict = {}
    self._vocab_size = 0
    self._frozen = False
    self._unk = None

  def freeze(self, unknown_marker = '<UNK>'):
    """Prevent further additions to the symbol table.

    After a table is frozen, subsequent calls to insert() will not change the
    table, and calls to get_id(s) will return the id for the unknown word marker
    for any s that did not have an id when the table was frozen.

    Args:
      unknown_marker: string  Unknown words or words added after a freeze are
        mapped to the unknown marker. If None, do not use an unknown value.
    """
    if unknown_marker:
      self.insert(unknown_marker)
      self._unk = self._id_dict[unknown_marker]
    self._frozen = True

  def reset(self):
    """Reset the set of symbols allowed, but keep the same vocabulary size."""
    self._symbol_list = []
    self._next_id = 0
    self._id_dict = {}

  def pad_to_vocab_size(self, n):
    """Force the maximum vocabulary size to be n.

    Args:
      n: a positive integer

    Raises:
      IndexError: When n is smaller than the existing vocabulary size.
    """
    if n < self._next_id:
      raise IndexError(
          'Cannot reduce vocabulary from %d to %d' % (self._next_id, n))
    tmp = self._next_id
    self._next_id = tmp
    self._vocab_size = n

  def insert(self, symbol):
    """Try to insert a symbol into the table.

    No action will be taken after the table has been frozen.

    Args:
      symbol: a string

    Raises:
      IndexError: When exceeding a fixed vocabulary size.
      ValueError: When inserting an illegally named entity.
    """
    if symbol not in self._id_dict and not self._frozen:
      if self._vocab_size and self._next_id >= self._vocab_size:
        raise IndexError(
            'Cannot exceed fixed vocabluary size of %d' % self._vocab_size)
      self._id_dict[symbol] = self._next_id
      self._symbol_list.append(symbol)
      self._next_id += 1
    if symbol not in self._id_dict and self._frozen and self._unk is None:
      raise ValueError('Trying to insert an illegal entity name %s' % symbol)

  def get_symbol(self, index):
    """Map index to symbol with that id.

    Args:
      index: integer id of an entity

    Returns:
      string for that symbol

    Raises:
      RuntimeError: If the id value exceeds a fixed vocabulary size.
    """

    if index < len(self._symbol_list):
      return self._symbol_list[index]
    elif self._vocab_size and index < self._vocab_size:
      return '<PAD%03d>' % index
    else:
      raise RuntimeError(
          'Index %d is not smaller than limit %d.' % (index, self._vocab_size))

  def has_id(self, symbol):
    """Test if symbol has been inserted into the table.

    Args:
      symbol: a string

    Returns:
      boolean indicating if the symbol appears in the table.
    """
    return symbol in self._id_dict

  def get_insert_id(self, symbol):
    """Get the numeric id of a symbol.

    If the symbol is not present and the SymbolTable is not frozen, it is
    inserted. Then a valid id is returned, which will be either the
    id for this symbol (for an unfrozen symbol table), or the id of the
    unknown-symbol marker (for a frozen symbol table)>

    Args:
      symbol: string name of name.

    Returns:
      integer id for the symbol (or the id for the unknown_marker)
    """
    self.insert(symbol)
    return self._id_dict.get(symbol, self._unk)

  def get_id(self, symbol):
    """Get the numeric id of a symbol.

    This assumes that either the symbol has been inserted in the
    past - ie that self.has_id(symbol) succeeds - or that the
    symbol table has been frozen.

    Args:
      symbol: string name of symbol.

    Returns:
      integer id for the symbol (or the id for the unknown_marker)

    Raises:
      KeyError if the symbol does not map onto an id.
    """
    if self._frozen:
      return self._id_dict.get(symbol, self._unk)
    else:
      return self._id_dict[symbol]

  def get_max_id(self):
    """An integer upper bound on symbol ids.

    Returns:
      one plus largest assigned symbol id.
    """
    return self._vocab_size or self._next_id

  def get_unk_id(self):
    """Return the unknown symbol id or None if nonexistent."""
    return self._unk

  def get_symbols(self):
    """Return all the defined symbols in a list."""
    return self._symbol_list

  def to_dict(self):
    """Return this object as a simple dictionary for serialization.

    Returns:
      dict version of a SymbolTable.
    """
    return vars(self)


def create_from_dict(symbol_table_dict):
  """Create a new SymbolTable from a dictionary for deserialization.

  Args:
    symbol_table_dict: A dict created from to_dict.

  Returns:
    A newly converted SymbolTable.
  """
  symbol_table = SymbolTable()
  for (a, v) in symbol_table_dict.items():
    setattr(symbol_table, a, v)
  return symbol_table

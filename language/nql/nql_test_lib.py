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
"""Utilities and tools to assist NQL tests."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from language.nql import nql



def cell(i, j):
  """Makes a cell name from coordinates."""
  return 'cell_%d_%d' % (i, j)


def make_grid():
  """Create a grid, with relations for going n, s, e, w."""
  result = nql.NeuralQueryContext()
  result.declare_relation('n', 'place_t', 'place_t')
  result.declare_relation('s', 'place_t', 'place_t')
  result.declare_relation('e', 'place_t', 'place_t')
  result.declare_relation('w', 'place_t', 'place_t')
  result.declare_relation('color', 'place_t', 'color_t')
  result.declare_relation('distance_to', 'place_t', 'corner_t')

  kg_lines = []
  dij = {'n': (-1, 0), 's': (+1, 0), 'e': (0, +1), 'w': (0, -1)}
  for i in range(0, 4):
    for j in range(0, 4):
      cell_color = 'black' if (i % 2) == (j % 2) else 'white'
      kg_lines.append('\t'.join(['color', cell(i, j), cell_color]) + '\n')
      kg_lines.append(
          '\t'.join(['distance_to', cell(i, j), 'ul',
                     str(i + j)]) + '\n')
      for direction, (di, dj) in dij.items():
        if (0 <= i + di < 4) and (0 <= j + dj < 4):
          kg_lines.append(
              '\t'.join([direction, cell(i, j),
                         cell(i + di, j + dj)]) + '\n')
  result.load_kg(lines=kg_lines)
  return result

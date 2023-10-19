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
"""The world class."""

import itertools
import random

from GroundedScan import world

# Map converts direction to direction vector (col, row)
DIR_TO_DIR_VEC = {
    "east of": (1, 0),
    "north of": (0, -1),
    "west of": (-1, 0),
    "south of": (0, 1),
    "north east of": (1, -1),
    "south east of": (1, 1),
    "north west of": (-1, -1),
    "south west of": (-1, 1)
}


class RelationWorld(world.World):
  """The world class that considers object spatial relations.

  Similar to the original gSCAN world state but allows sampling positions
  basaed on specific conditions. See
  https://github.com/LauraRuis/groundedSCAN/blob/master/GroundedScan/world.py
  for more details.
  """

  def get_nearby_positions(self, position, exclude_locations=None):
    """Return a list of available positions that are next to the given position."""
    all_positions = set(
        itertools.product(
            set(range(self.grid_size)), list(range(self.grid_size))))
    available_positions = all_positions - self._occupied_positions
    relative_positions = set(DIR_TO_DIR_VEC.keys())
    if exclude_locations is not None:
      relative_positions = relative_positions - set(exclude_locations)
    relative_directions = [DIR_TO_DIR_VEC[loc] for loc in relative_positions]
    nearby_positions = set([(position.column - dir[0], position.row - dir[1])
                            for dir in relative_directions])
    actual_available_positions = available_positions.intersection(
        nearby_positions)
    return actual_available_positions

  def sample_nearby_position(self, position, exclude_locations=None):
    """Sample an available position that is next to the given position."""
    actual_available_positions = self.get_nearby_positions(
        position, exclude_locations=exclude_locations)
    if actual_available_positions:
      sampled_position = random.sample(
          list(actual_available_positions), 1
      ).pop()
      return world.Position(column=sampled_position[0], row=sampled_position[1])
    return None

  def sample_non_nearby_position(self, position):
    """Sample an available position that is not next to the given position."""
    all_positions = set(
        itertools.product(
            list(range(self.grid_size)), list(range(self.grid_size))))
    available_positions = all_positions - self._occupied_positions
    nearby_positions = set([(position.column - dir[0], position.row - dir[1])
                            for dir in DIR_TO_DIR_VEC.values()])
    actual_available_positions = available_positions - nearby_positions
    if actual_available_positions:
      sampled_position = random.sample(
          list(actual_available_positions), 1
      ).pop()
      return world.Position(column=sampled_position[0], row=sampled_position[1])
    return None

  def get_relative_position(self, position, location):
    """Return a position that has relative position to given position."""
    if location == "next to":
      relative_position = self.sample_nearby_position(position)
    else:
      dir_vec = DIR_TO_DIR_VEC[location]
      relative_position = world.Position(
          column=position.column - dir_vec[0], row=position.row - dir_vec[1])
    return relative_position

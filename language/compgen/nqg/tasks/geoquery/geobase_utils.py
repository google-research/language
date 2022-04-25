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
"""Utilities for extracting entities from geobase file.

geobase file is available at:
http://www.cs.utexas.edu/users/ml/nldata/geoquery.html
"""

import collections

from tensorflow.io import gfile

GeoEntity = collections.namedtuple(
    "GeoEntity",
    [
        "aliases",  # List of Strings.
        "attribute",  # String.
        "identifier",  # String.
    ])


def _add_underspecified_city_constant(city_name, identifiers_to_entities):
  identifier = "cityid('%s',_)" % city_name
  if identifier in identifiers_to_entities:
    return
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="cityid", aliases=[city_name])


def _add_city_constants(city_name, state_name, state_abbreviation,
                        identifiers_to_entities):
  """Add constants for fully and under-specified city."""
  _add_underspecified_city_constant(city_name, identifiers_to_entities)
  identifier = "cityid('%s','%s')" % (city_name, state_abbreviation)
  if identifier in identifiers_to_entities:
    return
  aliases = [
      "%s %s" % (city_name, state_name),
      "%s %s" % (city_name, state_abbreviation),
  ]
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="cityid", aliases=aliases)


def _add_state_constant(name, identifiers_to_entities):
  identifier = "stateid('%s')" % name
  if identifier in identifiers_to_entities:
    return
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="stateid", aliases=[name])


def _add_river_constant(name, identifiers_to_entities):
  """Add entities for rivers."""
  identifier = "riverid('%s')" % name
  if identifier in identifiers_to_entities:
    return
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="riverid", aliases=[name])


def _add_place_constant(name, identifiers_to_entities):
  identifier = "placeid('%s')" % name

  if identifier in identifiers_to_entities:
    return
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="placeid", aliases=[name])


def _add_usa(identifiers_to_entities):
  """Add constant for usa."""
  # Only one `country` predicate appears in geobase:
  # country('usa',307890000,9826675)
  # Special-case `usa` and add some known aliases.
  identifier = "countryid(usa)"
  aliases = [
      "america",
      "continental us",
      "united states",
      "us",
      "usa",
      "country",
  ]
  identifiers_to_entities[identifier] = GeoEntity(
      identifier=identifier, attribute="countryid", aliases=aliases)


def load_entities(geobase_file):
  """Returns list of GeoEntity tuples for geobase entities."""
  # Identifier string to GeoEntity tuple.
  identifiers_to_entities = {}

  with gfile.GFile(geobase_file, "r") as inputfile:
    for line in inputfile:
      # line = line.decode("latin1")
      if line.startswith("state"):
        splits = line.split("'")
        state_name = splits[1]
        state_abbreviation = splits[3]
        city_capital = splits[5]
        city_1 = splits[7]
        city_2 = splits[9]
        city_3 = splits[11]
        city_4 = splits[13]
        _add_state_constant(state_name, identifiers_to_entities)
        for city_name in [city_capital, city_1, city_2, city_3, city_4]:
          _add_city_constants(city_name, state_name, state_abbreviation,
                              identifiers_to_entities)

      elif line.startswith("city"):
        state_name = line.split("'")[1]
        state_abbreviation = line.split("'")[3]
        city_name = line.split("'")[5]
        _add_city_constants(city_name, state_name, state_abbreviation,
                            identifiers_to_entities)

      elif line.startswith("river"):
        river_name = line.split("'")[1]
        _add_river_constant(river_name, identifiers_to_entities)

      elif line.startswith("mountain"):
        mountain_name = line.split("'")[5]
        _add_place_constant(mountain_name, identifiers_to_entities)

      elif line.startswith("highlow"):
        lowpoint_name = line.split("'")[5]
        highpoint_name = line.split("'")[7]
        _add_place_constant(lowpoint_name, identifiers_to_entities)
        _add_place_constant(highpoint_name, identifiers_to_entities)

  # This city is not mentioned in geobase directly, but referenced by a query
  # in the train set.
  _add_city_constants("springfield", "south dakota", "sd",
                      identifiers_to_entities)
  _add_usa(identifiers_to_entities)
  return identifiers_to_entities.values()

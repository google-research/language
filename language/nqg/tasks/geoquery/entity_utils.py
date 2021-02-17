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
"""Utilities for replacing entities in FunQL."""


def _maybe_list_replace(lst, sublist, replacement):
  """Replace first occurrence of sublist in lst with replacement."""
  new_list = []
  idx = 0
  replaced = False
  while idx < len(lst):
    if not replaced and lst[idx:idx + len(sublist)] == sublist:
      new_list.append(replacement)
      replaced = True
      idx += len(sublist)
    else:
      new_list.append(lst[idx])
      idx += 1
  if not replaced:
    return None
  return new_list


def _maybe_replace_entity(funql, utterance, mention_map, geobase_entity):
  """Replace entity identifiers so that they can be generated using copy."""
  # GeoQuery has <= 2 mentions per query.
  mention_marker = "m1" if "m0" in utterance else "m0"

  # Split utterance to avoid replacing some substring of a token.
  tokens = utterance.split(" ")

  # Order aliases by longest alias, since some can be nested in others.
  aliases = sorted(geobase_entity.aliases, key=lambda x: -len(x))
  for alias in aliases:
    alias_tokens = alias.split(" ")
    new_tokens = _maybe_list_replace(tokens, alias_tokens, mention_marker)
    if new_tokens:
      normalized_identifier = geobase_entity.identifier.replace("'", "")
      new_funql = funql.replace(normalized_identifier, mention_marker)
      new_utterance = " ".join(new_tokens)
      mention_map[mention_marker] = geobase_entity.identifier
      return new_funql, new_utterance, mention_map

  # Could not find alias.
  return funql, utterance, mention_map


def replace_entities(funql, utterance, geobase_entities):
  """Replace entity references with something more copy friendly."""
  # Order entities by longest identifier, since some can be nested
  # in others.
  geobase_entities = sorted(geobase_entities, key=lambda x: -len(x.identifier))
  mention_map = {}
  for geobase_entity in geobase_entities:
    normalized_identifier = geobase_entity.identifier.replace("'", "")
    if normalized_identifier in funql:
      funql, utterance, mention_map = _maybe_replace_entity(
          funql, utterance, mention_map, geobase_entity)
  return funql, utterance, mention_map


def restore_entities(funql, mention_map):
  """Restore entities in funql."""
  for mention_mark, identifier in mention_map.items():
    funql = funql.replace(mention_mark, "%s" % identifier)
  return funql

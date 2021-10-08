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
"""Checkpoint utils."""

import logging
import os
from typing import Any, Dict, Tuple, Sequence

from flax import serialization
import jax
import jax.numpy as jnp
from tensorflow.io import gfile


def save_weights(weight_path: str, model_params: Dict[str, Any]):
  """Save serialized weight dictionary."""
  serialized_params = serialization.to_bytes(model_params)
  gfile.makedirs(os.path.dirname(weight_path))
  with gfile.GFile(weight_path, 'wb') as fp:
    fp.write(serialized_params)


def load_weights(weight_path: str) -> Dict[str, Any]:
  """Load and deserialize weight dictionary."""
  if not gfile.exists(weight_path):
    raise ValueError('Matching checkpoint not found: {}'.format(weight_path))
  else:
    logging.info('Loading weights from %s', weight_path)
    with gfile.GFile(weight_path, 'rb') as fp:
      params = serialization.from_bytes(None, fp.read())
    return jax.tree_map(jnp.asarray, params)


def flatten_nested_dict(x: Dict[str, Any],
                        join_str: str = '/',
                        prefix: str = '') -> Dict[str, Any]:
  """Transforms nested dictionary into a flat dictionary."""
  assert isinstance(x, dict)
  result = {}
  for k, v in x.items():
    key = prefix + join_str + k
    if isinstance(v, dict):
      result.update(flatten_nested_dict(v, join_str, key))
    else:
      result[key] = v
  return result


def _merge_nested_dicts_rec(original: Dict[str, Any],
                            update: Dict[str, Any],
                            prefix: str = '') -> Sequence[str]:
  """Procedure to merge two nested dictionaries."""
  unexpected = []
  for key in update:
    full_key = prefix + '/' + key
    if isinstance(update[key], dict):
      if key in original:
        assert isinstance(original[key], dict), key
        unexpected.extend(
            _merge_nested_dicts_rec(
                original[key], update[key], prefix=full_key))
      else:
        original[key] = {}
        unexpected.extend(
            _merge_nested_dicts_rec(
                original[key], update[key], prefix=full_key))
    else:
      if key not in original:
        unexpected.append(full_key)
      original[key] = update[key]

  return unexpected


def merge_nested_dicts(
    original: Dict[str, Any],
    update: Dict[str, Any]) -> Tuple[Sequence[str], Sequence[str]]:
  """Merges `update` nested dict into the `original` dict.

  Args:
    original: target nested dictionary to be updated in-place.
    update: nested dictionary which values are copied into the `original` dict.

  Returns:
    Returns a pair of lists. The first list contains "unexpected" keys,
    which existed in the `update` dictionary, but not in the `original`.
    The second list contains "missing" keys, which existed in the
    `original` one, but not in the `update`.
  """
  unexpected, missing = [], []
  original_keys = frozenset(flatten_nested_dict(original).keys())
  update_keys = frozenset(flatten_nested_dict(update).keys())
  unexpected = frozenset(_merge_nested_dicts_rec(original, update))
  missing = original_keys.difference(update_keys)
  assert unexpected == update_keys.difference(original_keys)
  return list(unexpected), list(missing)

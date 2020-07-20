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
# Lint as: python3
"""Utilities for fever project."""
import contextlib
import json
import os
import pathlib
import random
import time

import unicodedata

from absl import logging
import tensorflow.compat.v2 as tf


Path = Union[Text, pathlib.PurePath]




def safe_path(path):
  """Return a path that is safe to write to by making intermediate dirs.

  Args:
    path: Path to ensure is safe

  Returns:
    The original path, converted to Text if it was a PurePath
  """
  if isinstance(path, pathlib.PurePath):
    path = str(path)
  directory = os.path.dirname(path)
  tf.io.gfile.makedirs(directory)
  return path




def safe_copy(src, dst, overwrite=True):
  """Copy safely from src to dst by creating intermediate directories.

  Args:
    src: File to copy
    dst: Where to copy to
    overwrite: Whether to overwrite destination if it exists
  """
  if isinstance(src, pathlib.PurePath):
    src = str(src)

  dst = safe_path(dst)
  tf.io.gfile.copy(src, dst, overwrite=overwrite)


def safe_open(path, mode = 'r'):
  """Open the path safely. If in write model, make intermediate directories.

  Args:
    path: path to open
    mode: mode to use

  Returns:
    file handler
  """
  if isinstance(path, pathlib.PurePath):
    path = str(path)

  if 'w' in mode:
    directory = os.path.dirname(path)
    tf.io.gfile.makedirs(directory)
    return tf.io.gfile.GFile(path, mode)
  else:
    return tf.io.gfile.GFile(path, mode)


def read_jsonlines(path):
  """Read jsonlines file from the path.

  Args:
    path: Path to json file

  Returns:
    List of objects decoded from each line
  """
  entries = []
  with safe_open(path) as f:
    for line in f:
      entries.append(json.loads(line))
  return entries


def read_json(path):
  """Read json file from path and return.

  Args:
    path: Path of file to read

  Returns:
    JSON object from file
  """
  with safe_open(path) as f:
    return json.load(f)


def write_json(obj, path):
  """Write json object to the path.

  Args:
    obj: Object to write
    path: path to write
  """
  with safe_open(path, 'w') as f:
    json.dump(obj, f)


def random_string(*, prefix = None):
  """Return a moderately randomized string, possibly with a prefix.

  Helpful for generating random directories to write different models to

  Args:
    prefix: If not None, prefix this to the random string

  Returns:
    Random string, perhaps with a prefix
  """
  # For the use case, this is large enough (unique experiment IDs)
  postfix = str(random.randrange(1_000_000, 2_000_000))
  if prefix is None:
    return postfix
  else:
    return f'{prefix}-{postfix}'


@contextlib.contextmanager
def log_time(message):
  """Utility to easily log the runtime of a passage of code with a message.

  EG.
  with log_time('hello there'):
    time.sleep(1)
  # prints: hello there: 1 seconds

  Args:
    message: The message to prepend to the runtime of the code

  Yields:
    Nothing, but can be used with "with" statement.
  """
  start = time.time()
  yield
  end = time.time()
  logging.info('%s: %s seconds', message, end - start)


def normalize(wikipedia_url):
  """Unicode normalize the wikipedia title.

  Args:
    wikipedia_url: The original title

  Returns:
    The unicode normalized title
  """
  return unicodedata.normalize('NFC', wikipedia_url)


def tf_to_str(text):
  """Convert a string-like input to python string.

  Specifically, this is helpful when its unclear whether a function is
  expected a tf.Tensor wrapping a string, a bytes object from unwrapping
  from a tf.Tensor, or the input is already a normal python string.

  Args:
    text: A tf.Tensor containing a string, a bytes object that represents a
      utf-8 string, or a string itself.

  Returns:
    Python string of the input
  """
  if isinstance(text, tf.Tensor):
    text = text.numpy()

  if isinstance(text, bytes):
    return text.decode('utf8')
  elif isinstance(text, Text):
    return text
  else:
    input_type = type(text)
    raise TypeError(f'Unexpected type: {input_type} for input: {text}')

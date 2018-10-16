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
"""Utils for computing character embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from language.common.utils import tensor_utils
import tensorflow as tf

# The character ids are defined by ord(c) for c in word.encode("utf-8").
# Ids 0-255 are reserved for utf-8 encoding bytes.
# We also include 3 special characters.
BOW_CHAR = 256
EOW_CHAR = 257
PAD_CHAR = 258

NUM_CHARS = PAD_CHAR + 1


def word_to_char_ids(word, word_length):
  """Convert a string to a padded vector of character ids.

  If the true length of the word is less than `word_length`, padding is added.
  If the true length of the word is greater than `word_length`, additional bytes
  are ignored.

  Args:
    word: <string> []
    word_length: Number of bytes to include per word.

  Returns:
    char_ids: <int32> [word_length]
  """
  char_ids = tf.to_int32(tf.decode_raw(word, tf.uint8))[:word_length - 2]
  padding = tf.fill([word_length - tf.shape(char_ids)[0] - 2], PAD_CHAR)
  char_ids = tf.concat([[BOW_CHAR], char_ids, [EOW_CHAR], padding], 0)
  char_ids.set_shape([word_length])
  return char_ids


def batch_word_to_char_ids(words, word_length):
  """Batched version of word_to_char_ids.

  This is a deterministic function that should be computed during preprocessing.
  We pin this op to the CPU anyways to be safe, since it is slower on GPUs.

  Args:
    words: <string> [...]
    word_length: Number of bytes to include per word.

  Returns:
    char_ids: <int32> [..., word_length]
  """
  with tf.device("/cpu:0"):
    flat_words = tf.reshape(words, [-1])
    flat_char_ids = tf.map_fn(
        fn=partial(word_to_char_ids, word_length=word_length),
        elems=flat_words,
        dtype=tf.int32,
        back_prop=False)

  char_ids = tf.reshape(flat_char_ids,
                        tensor_utils.shape(words) + [word_length])
  return char_ids


def token_to_char_ids_mapper(keys_to_map, word_length=50, suffix="_cid"):
  """Creates a mapping function to augment a tf.data.Dataset with character ids.

  Suppose we have a `dataset` with outputs `str1` and `str2` of arbitrary shape.
  Here is an example usage of this function:

  dataset = dataset.map(string_to_char_ids(['str1', 'str2']))

  Now the dataset will also include outputs `str1_cid` and `str2_cid` that can
  be used as features in a model. The 'str1_cid' feature will be a int32 Tensor
  of shape str1.get_shape() + [word_length].

  Args:
    keys_to_map: List of strings that are keys for tf.string Tensors to map to
        character ids.
    word_length: Number of bytes to include per word.
    suffix: String to append to the given keys to indicate the mapped Tensors.

  Returns:
    _mapper: A mapping function that can be used with the tf.data.Dataset API.
  """
  def _mapper(dataset):
    for k in keys_to_map:
      dataset[k + suffix] = batch_word_to_char_ids(dataset[k], word_length)
    return dataset
  return _mapper

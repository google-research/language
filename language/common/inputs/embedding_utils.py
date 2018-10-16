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
"""Utilities for computing word embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.common.inputs import dataset_utils

import numpy as np
import tensorflow as tf


class PretrainedWordEmbeddings(object):
  """Class that stores pretrained word embeddings.

  This class supports both trainable and fixed embeddings.

  Important note: When you create this object, the entire pretrained vocabulary
  and pretrained word embeddings are loaded into memory. The pretrained
  embeddings are used only during initialization using a tf.train.Scaffold
  object obtained via get_params().
  """

  def __init__(self, embeddings_path, max_vocab_size, num_oov_buckets=1,
               lowercase=False, vocab_path=None):
    """Read and store embeddings for a text file up to a maximum number.

    Args:
      embeddings_path: String for path to where the pretrained embeddings are
          stored. The file should be in the typical GloVe format, e.g. lines
          each containing "word 0.1 -0.2 1.5" for pretrained embeddings of
          length 3.
      max_vocab_size: Integer indicating the maximum vocabulary size. This is
          most effective when the embeddings are sorted by frequency, making
          later embeddings more reasonable to truncate.
      num_oov_buckets: Integer indicating the number of OOV buckets. If this is
          one, then the OOV bucket will be set to a zero vector. If this is
          greater than one, then each OOV bucket is randomly sampled from a
          normal distribution.
      lowercase: Boolean indicating whether embeddings shouled be lowercased. In
          the case of duplicates due to lower caseing, we use the first one.
      vocab_path: String for path to an optional vocabulary file. If provided,
          loads embeddings only from the vocabulary. Otherwise, loads
          `max_vocab_size` embeddings from the embeddings file in order.

    Raises:
      ValueError: If embeddings_path does not contain any embeddings or
        max_vocab_size is 0.
    """
    self._idx2str = []
    self._idx2emb = []
    self._dims = None

    provided_vocab = set()
    if vocab_path is not None:
      tf.logging.info("Vocabulary was provided. Loading from %s...", vocab_path)
      with tf.gfile.Open(vocab_path) as vocab_file:
        for line in vocab_file:
          splits = line.strip().split(" ")
          word = splits[0]
          if lowercase:
            word = word.lower()
          provided_vocab.add(word)

    tf.logging.info("Loading embeddings from %s...", embeddings_path)

    vocab = set()
    with tf.gfile.Open(embeddings_path) as embeddings_file:
      for line in embeddings_file:
        if max_vocab_size is not None and len(vocab) >= max_vocab_size:
          break
        word_end_ix = line.find(" ")
        word = line[:word_end_ix]

        if lowercase:
          word = word.lower()
        if (word in vocab) or (provided_vocab and word not in provided_vocab):
          continue
        else:
          vocab.add(word)

        self._idx2str.append(word)

        word_emb = np.fromstring(line[word_end_ix+1:], np.float32, sep=" ")
        word_emb = l2_normalize(word_emb)
        if self._dims is not None:
          assert self._dims == len(word_emb)
        else:
          self._dims = len(word_emb)
        self._idx2emb.append(word_emb)

    tf.logging.info("Loaded %d embeddings.", len(self._idx2emb))
    if self._dims is None:
      raise ValueError("No embeddings found.")

    self._num_oov_buckets = num_oov_buckets

    if num_oov_buckets == 1:
      self._idx2emb.append(np.zeros([self._dims], dtype=np.float32))
    else:
      for _ in range(self._num_oov_buckets):
        self._idx2emb.append(l2_normalize(np.random.normal(size=(self._dims))))

    self._idx2str = np.array(self._idx2str)
    self._idx2emb = np.array(self._idx2emb, dtype=np.float32)

  def get_lookup_table(self):
    """Create the lookup table base on the vocabulary."""
    return tf.contrib.lookup.index_table_from_tensor(
        mapping=self._idx2str, num_oov_buckets=self._num_oov_buckets)

  def token_to_word_id_mapper(self, keys_to_map, suffix="_wid"):
    """Creates a mapping function to augment a tf.data.Dataset with word ids.

    Suppose we have a `dataset` with outputs `str1` and `str2` of arbitrary
    shape. Here is an example usage of this function:

    embeddings = PretrainedWordEmbeddings("/path/to/emb.txt", 10000)
    dataset = dataset.map(embeddings.token_to_word_id_mapper(['str1', 'str2']))

    Now the dataset will also include outputs `str1_wid` and `str2_wid` that
    can be used as features in a model.  The 'str1_wid' feature has the same
    shape as the 'str1' feature, except the string are replaced with their
    int32 word IDs.

    Args:
      keys_to_map: List of strings that are keys for tf.string Tensors to map to
          word ids.
      suffix: String to append to the given keys to indicate the mapped Tensors.

    Returns:
      _mapper: A mapping function that can be used with the tf.data.Dataset API.
    """
    return dataset_utils.string_to_int_mapper(
        keys_to_map=keys_to_map,
        mapping=self._idx2str,
        num_oov_buckets=self._num_oov_buckets,
        suffix=suffix)

  def get_params(self, trainable=False, scope="embedding", reuse=False):
    """Returns a variable with the embeddings and a scaffold to initialize them.

    Args:
      trainable: Boolean indicating whether the params should be trainable.
      scope: The name of the inner-most scope for the params.
      reuse: Boolean indicating whether to reuse params in the same scope.

    Returns:
      embedding_weights: The embedding weights.
      embedding_scaffold: A tf.train.Scaffold to be passed to the EstimatorSpec,
          which will initialize embedding_weights correctly.
    """
    with tf.variable_scope(scope, reuse=reuse):
      embedding_weights = tf.get_variable(
          "embedding_weights",
          shape=[len(self._idx2emb), self._dims],
          trainable=trainable)

    # Use a placeholder for initialization to keep the pretrained embeddings
    # out of the TensorFlow graph.
    pretrained_placeholder = tf.placeholder(
        tf.float32, [len(self._idx2emb), self._dims])
    initialize_op = tf.assign(embedding_weights, pretrained_placeholder)

    def _init_fn(_, session):
      session.run(initialize_op, feed_dict={
          pretrained_placeholder: self._idx2emb
      })

    embedding_scaffold = tf.train.Scaffold(init_fn=_init_fn)

    return embedding_weights, embedding_scaffold

  def get_initialized_params(self, trainable=False, scope="embedding",
                             reuse=False):
    """Returns a variable with the embeddings.

    Unlike `get_params` this does not require running a Scaffold to initialize
    the variable, however this method is not compatible with `tf.SavedModel`
    since it uses a `tf.py_func` to initialize the embedddings variable.

    Args:
      trainable: Boolean indicating whether the params should be trainable.
      scope: The name of the inner-most scope for the params.
      reuse: Boolean indicating whether to reuse params in the same scope.

    Returns:
      embedding_weights: The embedding weights.
    """

    # Hide `self._idx2emb` behind tf.py_func so its does not get serialized as
    # as part of the graph and blow up our log sizes.
    init_value = tf.py_func(lambda: self._idx2emb, [], tf.float32, False)
    init_value.set_shape([len(self._idx2emb), self._dims])

    with tf.variable_scope(scope, reuse=reuse):
      if trainable:
        embedding_weights = tf.get_variable(
            "embedding_weights", initializer=init_value)
      else:
        # Local variable so the embeddings won't get dumped into the checkpoints
        embedding_weights = tf.get_local_variable(
            "embedding_weights", initializer=init_value)
    return embedding_weights

  def get_vocab(self):
    """Returns a list of all the words in the vocabulary of this object.

    Words are in the same order as their embeddings in the `embedding_weights`
    object. The OOV type is not included in this list.

    Returns:
      vocab: List of strings of all word types in the vocabulary.
    """
    return self._idx2str.tolist()

  def get_dims(self):
    """Returns the dimension size of word embeddings in this object.

    Returns:
      dims: Integer dimension of word embeddings. None if no embeddings were
        loaded.
    """
    return self._dims

  def get_vocab_size_with_oov(self):
    """Returns the total number of ids including OOV buckets."""
    return self._idx2emb.shape[0]

  def oov_metric(self, word_ids):
    """Metric for checking the OOV rate.

    Args:
      word_ids: int32 Tensor of arbitrary shape with word indices.

    Returns:
      metric: A metric computing the OOV rate. For use with tf.Estimator.
    """
    return tf.metrics.mean(tf.to_float(tf.greater(
        word_ids, len(self._idx2str) - 1)))


def l2_normalize(v, axis=0):
  """Divide by the L2-norm to normalize the input vector(s).

  For the edge case of the zero vector, this simply returns zero.

  Args:
    v: Numpy vector or matrix.
    axis: Dimension along which to normalize.

  Returns:
    normalized_v: Vector(s) `v` normalized to a unit vector.
  """
  norm = np.sqrt(np.square(v).sum(axis=axis, keepdims=True))
  # Set zero-norm elements to 1 to avoid NaN when dividing.
  norm[norm == 0] = 1.
  return v / norm

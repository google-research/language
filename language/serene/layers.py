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
"""Keras layers for Fever models."""

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub


class Embedder(tf.keras.layers.Layer):
  """Abstract class for text embedders to implement."""

  def call(self, inputs):
    # inputs: Dependent on encoder, typically encoder/embedder are paired.
    # e.g.: BertEmbedder with BertEncoder
    # return: (batch_size, hidden_dim)
    raise NotImplementedError()


class ClassicEmbedder(Embedder):
  """An text embedder that uses word embeddings and recurrent networks."""

  def __init__(self,
               vocab_size,
               *,
               word_emb_size,
               use_batch_norm,
               contextualizer,
               context_num_layers,
               bidirectional,
               hidden_size,
               dropout,
               name = None,
               embeddings = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._vocab_size = vocab_size
    self._word_emb_size = word_emb_size
    self._use_batch_norm = use_batch_norm
    self._contextualizer = contextualizer
    self._context_num_layers = context_num_layers
    self._bidirectional = bidirectional
    self._hidden_size = hidden_size
    self._dropout = dropout
    # Useful for tying weights on embedding layers
    if embeddings is None:
      self.embeddings = tf.keras.layers.Embedding(
          vocab_size, word_emb_size, mask_zero=True, name='embeddings')
    else:
      self.embeddings = embeddings

    if contextualizer == 'lstm':
      context_layer = tf.keras.layers.LSTM
    elif contextualizer == 'gru':
      context_layer = tf.keras.layers.GRU
    elif contextualizer == 'rnn':
      context_layer = tf.keras.layers.RNN
    else:
      raise ValueError(
          f'The contextualizer "{contextualizer}" is not supported in ClassicEmbedder'
      )

    self._encoder = tf.keras.Sequential(name='encoder')
    self._encoder.add(self.embeddings)
    if use_batch_norm:
      self._encoder.add(tf.keras.layers.BatchNormalization())
    self._encoder.add(tf.keras.layers.Dropout(dropout))
    if bidirectional:
      for _ in range(context_num_layers - 1):
        self._encoder.add(
            tf.keras.layers.Bidirectional(
                context_layer(
                    hidden_size,
                    recurrent_dropout=dropout,
                    return_sequences=True)))
        if use_batch_norm:
          self._encoder.add(tf.keras.layers.BatchNormalization())
        self._encoder.add(tf.keras.layers.Dropout(dropout))
      self._encoder.add(
          tf.keras.layers.Bidirectional(
              context_layer(hidden_size, recurrent_dropout=dropout)))
    else:
      for _ in range(context_num_layers - 1):
        self._encoder.add(
            context_layer(
                hidden_size, recurrent_dropout=dropout, return_sequences=True))
        if use_batch_norm:
          self._encoder.add(tf.keras.layers.BatchNormalization())
        self._encoder.add(tf.keras.layers.Dropout(dropout))
      self._encoder.add(context_layer(hidden_size, recurrent_dropout=dropout))

  def call(self, inputs, **kwargs):
    # (batch_size, seq_len)
    tokens = inputs['tokens']
    # (batch_size, hidden_dim)
    context_repr = self._encoder(tokens)
    return context_repr

  def get_config(self):
    config = super().get_config()
    config.update({
        'vocab_size': self._vocab_size,
        'word_emb_size': self._word_emb_size,
        'use_batch_norm': self._use_batch_norm,
        'contextualizer': self._contextualizer,
        'context_num_layers': self._context_num_layers,
        'bidirectional': self._bidirectional,
        'hidden_size': self._hidden_size,
        'dropout': self._dropout,
    })
    return config


class BertEmbedder(Embedder):
  """Embed text with bert models."""

  def __init__(self,
               model_name,
               bert_model_path,
               trainable = True,
               bert_trainable = True,
               bert_dropout = 0.1,
               name = None,
               **kwargs):
    super().__init__(name=name, trainable=trainable, **kwargs)
    self._model_name = model_name
    self._bert_trainable = bert_trainable
    self._bert_dropout = bert_dropout
    model_checkpoint = bert_model_path
    self._bert = hub.KerasLayer(model_checkpoint, trainable=bert_trainable)
    self._regularizer = tf.keras.Sequential()
    self._regularizer.add(tf.keras.layers.LayerNormalization())
    self._regularizer.add(tf.keras.layers.Dropout(bert_dropout))

  def call(self, inputs):
    # (batch_size, seq_len)
    word_ids = inputs['word_ids']
    # (batch_size, seq_len)
    bert_mask = inputs['mask']
    # (batch_size, seq_len)
    segment_ids = inputs['segment_ids']
    # (batch_size, hidden_dim)
    pooled, _ = self._bert([word_ids, bert_mask, segment_ids])
    return self._regularizer(pooled)

  def get_config(self):
    config = super().get_config()
    config.update({'model_name': self._model_name})
    config.update({'bert_trainable': self._bert_trainable})
    config.update({'bert_dropout': self._bert_dropout})
    return config


class Bilinear(tf.keras.layers.Layer):
  """Bilinear layer similar to torch's."""

  def __init__(self, hidden_dim, name = None, **kwargs):
    super().__init__(name=name, **kwargs)
    self._hidden_dim = hidden_dim

  def build(self, input_shape):
    super().build(input_shape)
    if not isinstance(input_shape, (List, Tuple)):
      raise ValueError(
          f'Invalid type for input_shape: {type(input_shape)} val={input_shape}'
      )
    if len(input_shape) != 2:
      raise ValueError(
          f'Expected two inputs, got: {len(input_shape)} val={input_shape}')
    x1_shape, x2_shape = input_shape
    self.weight = self.add_weight(
        name='bilinear_weights',
        initializer='glorot_uniform',
        shape=(self._hidden_dim, x1_shape[-1], x2_shape[-1]),
        trainable=True,
    )
    self.bias = self.add_weight(
        name='bilinear_bias',
        initializer='glorot_uniform',
        shape=(self._hidden_dim,),
        trainable=True,
    )

  def call(self, inputs, **kwargs):
    a = inputs[0]
    b = inputs[1]
    return tf.einsum('ik,jkl,il->ij', a, self.weight, b) + self.bias

  def get_config(self):
    config = super().get_config()
    config.update({'hidden_dim': self._hidden_dim})
    return config


class Matcher(tf.keras.layers.Layer):
  """A matching layer should take two inputs and output a scalar match score."""

  def call(self, inputs, **kwargs):
    # a, b = inputs
    # a: (batch_size, ..., size)
    # b: (batch_size, ..., size)
    # out: (batch_size, ..., 1)
    raise NotImplementedError()


class BilinearMatcher(Matcher):
  """Matcher that passes inputs to a bilinear layer to produce score."""

  def __init__(self,
               *,
               hidden_size,
               use_batch_norm,
               dropout,
               n_classes,
               activation,
               name = 'bilinear_matcher',
               **kwargs):
    """Build a bilinear matcher.

    Args:
      hidden_size: hidden size of bilinear layer
      use_batch_norm: whether to use batch norm before activation
      dropout: dropout rate
      n_classes: Number of logits/probs to output from final layer
      activation: Activation function or name of activation function
      name: Name of this module
      **kwargs: Forwarded to inherited layers
    """
    super().__init__(name=name, **kwargs)
    self._hidden_size = hidden_size
    self._n_classes = n_classes
    self._use_batch_norm = use_batch_norm
    self._dropout = dropout
    self._activation = activation
    self._matcher = tf.keras.Sequential(name='matcher_sequential')
    self._matcher.add(Bilinear(hidden_size, name='bilinear'))
    if use_batch_norm:
      self._matcher.add(tf.keras.layers.BatchNormalization())
    self._matcher.add(tf.keras.layers.Activation(activation))
    self._matcher.add(tf.keras.layers.Dropout(dropout))
    self._matcher.add(tf.keras.layers.Dense(n_classes))

  def call(self, inputs, **kwargs):
    # (batch_size, ..., size)
    a = inputs[0]
    # (batch_size, ..., size)
    b = inputs[1]
    # (batch_size, ..., 1)
    return self._matcher((a, b))

  def get_config(self):
    config = super().get_config()
    config.update({
        'hidden_size': self._hidden_size,
        'use_batch_norm': self._use_batch_norm,
        'dropout': self._dropout,
        'activation': self._activation,
        'n_classes': self._n_classes
    })
    return config


class ConcatLinearMatcher(Matcher):
  """Computes a match score for two vectors by concat plus linear layer."""

  def __init__(self,
               *,
               hidden_size,
               use_batch_norm,
               dropout,
               n_classes,
               activation,
               name = 'concat_linear_matcher',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._hidden_size = hidden_size
    self._n_classes = n_classes
    self._use_batch_norm = use_batch_norm
    self._dropout = dropout
    self._activation = activation
    self._matcher = tf.keras.Sequential(name='matcher_sequential')
    self._matcher.add(tf.keras.layers.Dense(hidden_size))
    if use_batch_norm:
      self._matcher.add(tf.keras.layers.BatchNormalization())
    self._matcher.add(tf.keras.layers.Activation(activation))
    self._matcher.add(tf.keras.layers.Dropout(dropout))
    self._matcher.add(tf.keras.layers.Dense(n_classes))

  def call(self, inputs, **kwargs):
    # (batch_size, ..., size)
    a = inputs[0]
    # (batch_size, ..., size)
    b = inputs[1]
    # (batch_size, ..., 2 * size)
    a_concat_b = tf.keras.layers.concatenate([a, b], axis=-1)
    # (batch_size, ..., 1)
    return self._matcher(a_concat_b)

  def get_config(self):
    config = super().get_config()
    config.update({
        'hidden_size': self._hidden_size,
        'use_batch_norm': self._use_batch_norm,
        'dropout': self._dropout,
        'activation': self._activation,
        'n_classes': self._n_classes,
    })
    return config


def cosine_similarity(a, b):
  """Compute cosine similarity along final dimension of tensors.

  In tensor size descriptions, ... indicates an arbitrary number of dimensions.
  For example, this can compute similarity for any of these:
  - Dialogs: (batch_size, n_messages, n_words, size)
  - Text: (batch_size, n_words, size)
  - Hidden State: (batch_size, size)

  Args:
    a: Left tensor to compare, (batch_size, ..., n)
    b: Right tensor to compare (batch_size, ..., n)

  Returns:
    TF Scalar cosine similarity between vectors a and b
  """
  # (batch_size, ..., 1)
  a_norm = tf.norm(a, 2, axis=-1)
  # (batch_size, ..., 1)
  b_norm = tf.norm(b, 2, axis=-1)
  # (batch_size, ..., 1)
  return tf.reduce_sum(a * b, axis=-1) / (a_norm * b_norm)


class CosineSimilarityMatcher(Matcher):
  """Match vectors with using only cosine similarity."""

  def __init__(self,
               name = 'cosine_similarity_matcher',
               hidden_size=None,
               use_batch_norm=None,
               dropout=None,
               activation=None,
               n_classes=None,
               **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs, **kwargs):
    # (batch_size, ..., size)
    a = inputs[0]
    # (batch_size, ..., size)
    b = inputs[1]
    # (batch_size, ..., 1)
    return tf.expand_dims(cosine_similarity(a, b), -1)

  # Disabling lint is required here since there are no configurable params to
  # save, but Layer.get_config() raises an exception if this is not overridden
  # and there are extra params to the constructor (required for API compat)
  # see Layer.get_config in tensorflow/python/keras/engine/base_layer.py
  def get_config(self):  # pylint: disable=useless-super-delegation
    return super().get_config()


class ProductSimilarityMatcher(Matcher):
  """Match vectors with using only cosine similarity."""

  def __init__(self,
               name = 'product_similarity_matcher',
               hidden_size=None,
               use_batch_norm=None,
               dropout=None,
               activation=None,
               n_classes=None,
               **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs, **kwargs):
    """Compute similarity between inputs via product.

    In tensor size descriptions, ... indicates an arbitrary number of
    dimensions.
    For example, this can compute similarity for any of these:
    - Dialogs: (batch_size, n_messages, n_words, size)
    - Text: (batch_size, n_words, size)
    - Hidden State: (batch_size, size)

    Args:
      inputs: Two input tensors
      **kwargs: API Compat

    Returns:
      Product similarity tensor
    """
    # (batch_size, ..., size)
    a = inputs[0]
    # (batch_size, ..., size)
    b = inputs[1]
    # (batch_size, ..., 1)
    return tf.expand_dims(tf.reduce_sum(a * b, axis=-1), -1)

  # Disabling lint is required here since there are no configurable params to
  # save, but Layer.get_config() raises an exception if this is not overridden
  # and there are extra params to the constructor (required for API compat)
  # see Layer.get_config in tensorflow/python/keras/engine/base_layer.py
  def get_config(self):  # pylint: disable=useless-super-delegation
    return super().get_config()


matcher_registry = {
    'concat_linear_matcher': ConcatLinearMatcher,
    'bilinear_matcher': BilinearMatcher,
    'cosine_matcher': CosineSimilarityMatcher,
    'product_matcher': ProductSimilarityMatcher,
}

embedder_registry = {
    'classic_embedder': ClassicEmbedder,
    'bert_embedder': BertEmbedder,
}

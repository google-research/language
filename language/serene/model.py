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
"""Fever models built from tf.keras.Model module API (eg two tower ranker)."""

from language.serene import layers
import numpy as np
import tensorflow.compat.v2 as tf


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def parse_activation(
    name):
  if name == 'gelu':
    return gelu
  else:
    return name


class TwoTowerRanker(tf.keras.Model):
  """Build a model of two towers that are joined with a matching function."""

  # pyformat: disable
  def __init__(
      self,
      vocab_size, *,
      embedder_name,
      tied_encoders,
      matcher_name,
      bidirectional,
      contextualizer,
      context_num_layers,
      activation,
      matcher_hidden_size,
      word_emb_size,
      hidden_size,
      projection_dim,
      bert_model_name,
      bert_model_path,
      bert_trainable,
      bert_dropout,
      dropout,
      use_batch_norm,
      classify_claim):
    # pyformat: enable
    """TwoTower model.

    Args:
      vocab_size: Size of vocab
      embedder_name: Which embedder, basic or bert
      tied_encoders: Whether to tie the encoders
      matcher_name: What kind of matcher to use
      bidirectional: Whether context layer is bidirectional
      contextualizer: Which contextualizer to use (e.g., GRU)
      context_num_layers: Number of layers in contextualizer
      activation: Activation function to use for feed forward layers
      matcher_hidden_size: Hidden size of matcher, if it has one
      word_emb_size: Word embedding hidden size
      hidden_size: Hidden size of contextualizer
      projection_dim: Dimension to project embeddings to
      bert_model_name: Name of bert model (e.g., base vs large)
      bert_model_path: Path to the bert checkpoint
      bert_trainable: Whether bert part of model is trainable
      bert_dropout: Dropout rate on bert embeddings
      dropout: Dropout rate to use
      use_batch_norm: Whether to use batch norm
      classify_claim: Whether to classify the claim
    """
    super().__init__(name='two_tower_ranker')
    self._vocab_size = vocab_size
    self._embedder_name = embedder_name
    self._matcher_name = matcher_name
    self._bidirectional = bidirectional
    self._contextualizer = contextualizer
    self._context_num_layers = context_num_layers
    self._activation = activation
    activation = parse_activation(activation)
    self._matcher_hidden_size = matcher_hidden_size
    self._word_emb_size = word_emb_size
    self._hidden_size = hidden_size
    self._projection_dim = projection_dim
    self._bert_model_name = bert_model_name
    self._bert_model_path = bert_model_path
    self._bert_trainable = bert_trainable
    self._bert_dropout = bert_dropout
    self._dropout = dropout
    self._use_batch_norm = use_batch_norm
    self._classify_claim = classify_claim
    self._tied_encoders = tied_encoders

    if embedder_name == 'classic_embedder':
      if self._tied_encoders:
        self._claim_encoder = layers.ClassicEmbedder(
            vocab_size=vocab_size,
            word_emb_size=word_emb_size,
            use_batch_norm=use_batch_norm,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            contextualizer=contextualizer,
            context_num_layers=context_num_layers,
            name='tied_encoder',
        )
        self._evidence_encoder = self._claim_encoder
      else:
        self._claim_encoder = layers.ClassicEmbedder(
            vocab_size=vocab_size,
            word_emb_size=word_emb_size,
            use_batch_norm=use_batch_norm,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            contextualizer=contextualizer,
            context_num_layers=context_num_layers,
            name='claim_encoder',
        )
        self._evidence_encoder = layers.ClassicEmbedder(
            vocab_size=vocab_size,
            word_emb_size=word_emb_size,
            use_batch_norm=use_batch_norm,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            contextualizer=contextualizer,
            context_num_layers=context_num_layers,
            name='evidence_encoder',
        )
    elif embedder_name == 'bert_embedder':
      if self._tied_encoders:
        self._claim_encoder = layers.BertEmbedder(
            bert_model_name,
            bert_model_path,
            name='tied_encoder',
            bert_trainable=bert_trainable,
            bert_dropout=bert_dropout,
        )
        self._evidence_encoder = self._claim_encoder
      else:
        self._claim_encoder = layers.BertEmbedder(
            bert_model_name,
            bert_model_path,
            name='claim_encoder',
            projection_dim=projection_dim,
            bert_trainable=bert_trainable,
            bert_dropout=bert_dropout,
        )
        self._evidence_encoder = layers.BertEmbedder(
            bert_model_name,
            bert_model_path,
            name='evidence_encoder',
            projection_dim=projection_dim,
            bert_trainable=bert_trainable,
            bert_dropout=bert_dropout,
        )
    else:
      raise ValueError('Invalid embedder')

    matcher = layers.matcher_registry[matcher_name](
        hidden_size=matcher_hidden_size,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
        # Relevant or not
        n_classes=1,
    )
    if self._tied_encoders:
      self._claim_projector = tf.keras.layers.Dense(projection_dim)
      self._evidence_projector = self._claim_projector
    else:
      self._claim_projector = tf.keras.layers.Dense(projection_dim)
      self._evidence_projector = tf.keras.layers.Dense(projection_dim)
    self._evidence_classifier = tf.keras.Sequential(name='evidence_classifier')
    self._evidence_classifier.add(matcher)
    self._evidence_classifier.add(tf.keras.layers.Activation('sigmoid'))
    if classify_claim:
      # This doesn't need to preserve property of being findable with dot
      # product, so make it more powerful with bilinear matching.
      claim_hidden = layers.matcher_registry['bilinear_matcher'](
          hidden_size=matcher_hidden_size,
          dropout=dropout,
          use_batch_norm=use_batch_norm,
          activation=activation,
          # support, refute, not enough info
          n_classes=3,
      )
      self._claim_classifier = tf.keras.Sequential(name='claim_classifier')
      self._claim_classifier.add(claim_hidden)
      self._claim_classifier.add(tf.keras.layers.Activation('softmax'))
    else:
      self._claim_classifier = None

  def call(self,
           inputs,
           embed_claim=False,
           embed_evidence=False,
           training=None,
           **kwargs):
    """Model forward pass.

    Args:
      inputs: Input dictionary, dependent on type of embedder used
      embed_claim: Whether to embed the claim
      embed_evidence: Whether to embed the evidence
      training: Whether training mode is enabled
      **kwargs: Satisfies API compat

    Returns:
      Prediction of the model if claim is relevant to evidence
    """
    if embed_claim or embed_evidence:
      # (batch_size, projection_dim), (batch_size, projection_dim)
      return self.embed_only(
          inputs, embed_claim=embed_claim, embed_evidence=embed_evidence)
    else:
      if self._embedder_name == 'classic_embedder':
        # (batch_size, hidden_dim)
        encoded_claim = self._claim_encoder({'tokens': inputs['claim_text']})
        # (batch_size, hidden_dim)
        encoded_evidence = self._evidence_encoder(
            {'tokens': inputs['evidence_text']})
      elif self._embedder_name == 'bert_embedder':
        # (batch_size, hidden_dim)
        encoded_claim = self._claim_encoder({
            'word_ids': inputs['claim_text_word_ids'],
            'mask': inputs['claim_text_mask'],
            'segment_ids': inputs['claim_text_segment_ids'],
        })
        # (batch_size, hidden_dim)
        encoded_evidence = self._evidence_encoder({
            'word_ids': inputs['evidence_text_word_ids'],
            'mask': inputs['evidence_text_mask'],
            'segment_ids': inputs['evidence_text_segment_ids'],
        })
      else:
        raise ValueError('invalid embedder')
      out = {}
      # (batch_size, projection_dim)
      projected_claim = self._claim_projector(encoded_claim)
      # (batch_size, projection_dim)
      projected_evidence = self._evidence_projector(encoded_evidence)
      # (batch_size, 1)
      evidence_out = self._evidence_classifier(
          (projected_claim, projected_evidence))
      out['evidence_matching'] = tf.identity(
          evidence_out, name='evidence_matching')
      if self._classify_claim:
        # The claim takes the full size embedding, not the projected one.
        claim_out = self._claim_classifier((encoded_claim, encoded_evidence))
      else:
        # Predictions have three logits, one for each class
        claim_out = tf.fill((tf.shape(encoded_claim)[0], 3), 0.0)
      out['claim_classification'] = tf.identity(
          claim_out, name='claim_classification')
      return out

  def embed_only(
      self,
      inputs,
      *,
      embed_claim,
      embed_evidence,
      project = True):
    if self._embedder_name == 'classic_embedder':
      if embed_claim:
        # (batch_size, hidden_dim)
        encoded_claim = self._claim_encoder({'tokens': inputs['claim_text']})
      else:
        encoded_claim = None
      if embed_evidence:
        # (batch_size, hidden_dim)
        encoded_evidence = self._evidence_encoder(
            {'tokens': inputs['evidence_text']})
      else:
        encoded_evidence = None
    elif self._embedder_name == 'bert_embedder':
      if embed_claim:
        # (batch_size, hidden_dim)
        encoded_claim = self._claim_encoder({
            'word_ids': inputs['claim_text_word_ids'],
            'mask': inputs['claim_text_mask'],
            'segment_ids': inputs['claim_text_segment_ids'],
        })
      else:
        encoded_claim = None
      if embed_evidence:
        # (batch_size, hidden_dim)
        encoded_evidence = self._evidence_encoder({
            'word_ids': inputs['evidence_text_word_ids'],
            'mask': inputs['evidence_text_mask'],
            'segment_ids': inputs['evidence_text_segment_ids'],
        })
      else:
        encoded_evidence = None
    else:
      raise ValueError('invalid embedder')

    if project:
      if encoded_claim is not None:
        encoded_claim = self._claim_projector(encoded_claim)
      if encoded_evidence is not None:
        encoded_evidence = self._evidence_projector(encoded_evidence)
    return encoded_claim, encoded_evidence

  def get_config(self):
    config = super().get_config()
    config.update({
        'vocab_size': self._vocab_size,
        'embedder_name': self._embedder_name,
        'tied_encoders': self._tied_encoders,
        'matcher_name': self._matcher_name,
        'bidirectional': self._bidirectional,
        'contextualizer': self._contextualizer,
        'context_num_layers': self._context_num_layers,
        'activation': self._activation,
        'matcher_hidden_size': self._matcher_hidden_size,
        'word_emb_size': self._word_emb_size,
        'hidden_size': self._hidden_size,
        'projection_dim': self._projection_dim,
        'bert_model_name': self._bert_model_name,
        'bert_model_path': self._bert_model_path,
        'bert_trainable': self._bert_trainable,
        'bert_dropout': self._bert_dropout,
        'dropout': self._dropout,
        'use_batch_norm': self._use_batch_norm,
        'classify_claim': self._classify_claim,
    })
    return config

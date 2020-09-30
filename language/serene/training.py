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
"""Training manager for fever code."""

import json
import os


from absl import logging
import dataclasses
from language.serene import callbacks
from language.serene import fever_tfds
from language.serene import layers
from language.serene import losses
from language.serene import model
from language.serene import preprocessing
from language.serene import tokenizers
from language.serene import util
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm

from official.common import distribute_utils


@dataclasses.dataclass
class ModelConfig:
  """Typed parameters for model."""
  fever_experiment_id: int
  model_checkpoint: Text
  dataset: Text
  buffer_size: int
  batch_size: int
  word_emb_size: int
  hidden_size: int
  learning_rate: float
  positive_class_weight: Optional[float]
  max_epochs: int
  dropout: float
  activation: Text
  use_batch_norm: bool
  # Model Choice: two_tower or one_tower (not implemented yet).
  model: Text
  # Preprocessing
  tokenizer: Text  # EG: Convert strings to list of strings.
  text_encoder: Text  # EG: Convert list of strings to integers.
  basic_lowercase: bool

  # Embedder + Contextualizer
  embedder: Text
  contextualizer: Text
  context_num_layers: int
  tied_encoders: bool
  bidirectional: bool
  bert_model_name: Text
  bert_max_seq_length: int
  bert_vocab_path: Text
  bert_model_path: Text
  bert_trainable: bool
  bert_dropout: float

  # Neural Module Configuration
  matcher: Text
  matcher_hidden_size: int

  projection_dim: int

  fever_dev_path: Text
  max_evidence: int

  max_claim_tokens: int
  max_evidence_tokens: int

  # Whether to include the title/sentence_id in evidence encoding.
  include_title: bool
  include_sentence_id: bool
  n_similar_negatives: int
  n_background_negatives: int
  scrape_type: Text
  include_not_enough_info: bool
  title_in_scoring: bool

  classify_claim: bool
  claim_loss_weight: float

  def validate(self):
    """Validate that the arguments to the config are correct, error if not."""
    if self.tokenizer not in ['bert', 'basic']:
      raise ValueError(f'Invalid tokenizer: "{self.tokenizer}"')

    if self.text_encoder not in ['bert', 'basic']:
      raise ValueError(f'Invalid text encoder: "{self.text_encoder}"')

    if self.matcher not in layers.matcher_registry:
      raise ValueError(f'Invalid matcher: "{self.matcher}"')

    if self.contextualizer not in ['bert', 'rnn', 'lstm', 'gru']:
      raise ValueError(f'Invalid contextualizer: "{self.contextualizer}"')

    if self.model not in ['one_tower', 'two_tower']:
      raise ValueError(f'Invalid model: "{self.model}"')

    if self.bert_model_name not in ['base', 'large']:
      raise ValueError(f'Invalid bert model: "{self.bert_model_name}')

    if self.embedder not in ['classic_embedder', 'bert_embedder']:
      raise ValueError(f'Invalid embedder: "{self.embedder}"')

  @classmethod
  def from_dict(cls, params):
    return ModelConfig(**params)

  @classmethod
  def from_file(cls,
                file_path,
                overrides = None):
    with util.safe_open(file_path) as f:
      params: Dict[Text, Any] = json.load(f)
      if overrides is not None:
        params.update(overrides)
      return ModelConfig.from_dict(params)

  def save(self, file_path):
    with util.safe_open(file_path, 'w') as f:
      json.dump(self.asdict(), f)

  def asdict(self):
    return dataclasses.asdict(self)


class Trainer:
  """Training wrapper around keras to manage vocab/saving/dataset creation.

  The primary methods of this class are:
  - train()
  - predict()
  - embed()
  - save()
  - load()

  The intended use of this is
  > trainer = Trainer(my_config)
  > trainer.train()

  The following methods are primarily for converting TFDS to tf.data.Dataset
  for keras training
  - _build_tokenizer()
  - _build_encoder()
  - _encode_and_batch()
  - _batch_dataset()
  - _encode_dataset()
  - _build_vocab()
  - _tokenize_example()

  These are utilities for embedding different TFDSs
  - embed_wiki_dataset()
  - embed_claim_dataset()

  The following methods deal with preparing the keras model for training
  - _compile(): Compile model uner right scope, create callbacks, glue losses
    to model
  - _build_callbacks(): Keras callbacks
  """

  def __init__(
      self,
      model_config,
      debug = False,
      tpu = None,
      distribution_strategy = None,
      tb_log_dir = None):
    """Configure the trainer.

    Args:
      model_config: ModelConfig parameters for training
      debug: Enables certain debug behaviors like dataset subsampling
      tpu: The TPU to use or None otherwise
      distribution_strategy: Parallel training strategy
      tb_log_dir: The directory for Tensorboard to log to
    """
    self._debug = debug
    if debug:
      logging.info('Debug mode enabled on trainer')
    self._tpu = tpu
    self._distribution_strategy = distribution_strategy
    self._tb_log_dir = tb_log_dir
    self._strategy: Optional[tf.distribute.Strategy] = None
    self._model_config = model_config
    self._vocab: Optional[List[Text]] = None
    self._vocab_stats: Dict[Text, Any] = {}
    self._class_stats: Dict[int, int] = {0: 0, 1: 0}
    # Whitespace tokenizer
    self._tokenizer: Optional[tokenizers.Tokenizer] = None
    self._encoder: Optional[preprocessing.FeverTextEncoder] = None
    self._model: Optional[tf.keras.Model] = None
    self._inner_model: Optional[tf.keras.Model] = None

  def save(self):
    """Persist the encoder and the model to disk.
    """
    if self._model is None or self._encoder is None:
      raise ValueError('Model and encoder cannot be None')
    else:
      self._encoder.save_to_file(
          # This is a prefix, which converts to: mydir/text_encoder.tokens
          os.path.join(self._model_config.model_checkpoint, 'text_encoder'))
      self._model.save_weights(
          os.path.join(self._model_config.model_checkpoint, 'best_model.tf'))

  @classmethod
  def load(cls,
           model_checkpoint,
           model_config_overrides = None,
           **kwargs):
    """Load the model, its tokenizer, and weights from the checkpoint.

    Args:
      model_checkpoint: Checkpoint to restore from, from .save()
      model_config_overrides: Extra args for ModelConfig
      **kwargs: Passed through to trainer, used for overriding checkpoint

    Returns:
      A model in the same state as just before it was saved with .save()
    """
    # pylint: disable=protected-access
    model_config = ModelConfig.from_file(
        os.path.join(model_checkpoint, 'model_config.json'),
        overrides=model_config_overrides)
    trainer = Trainer(model_config=model_config, **kwargs)
    trainer._tokenizer = trainer._build_tokenizer()
    encoder_path = os.path.join(model_checkpoint, 'text_encoder')
    if model_config.text_encoder == 'bert':
      trainer._encoder = preprocessing.BertTextEncoder.load_from_file(
          encoder_path)
    elif model_config.text_encoder == 'basic':
      trainer._encoder = preprocessing.BasicTextEncoder.load_from_file(
          encoder_path)
    else:
      raise ValueError('Invalid text encoder')

    trainer._compile()
    if trainer._model is None:
      raise ValueError('Model does not exist despite being compiled')
    trainer._model.load_weights(os.path.join(model_checkpoint, 'best_model.tf'))
    return trainer

  def _save_model_config(self):
    """Save only the Model configuration to disk."""
    logging.info('Saving config to: %s/model_config.json',
                 self._model_config.model_checkpoint)
    self._model_config.save(
        os.path.join(self._model_config.model_checkpoint, 'model_config.json'))

  def _save_encoder(self):
    """Save only the text encoder to disk."""
    self._encoder.save_to_file(
        os.path.join(self._model_config.model_checkpoint, 'text_encoder'))

  @property
  def vocab_size(self):
    if self._encoder is None:
      raise ValueError('Model has not been build, so no vocab size')
    else:
      return self._encoder.vocab_size

  def _init_strategy(self):
    """Initialize the distribution strategy (e.g. TPU/GPU/Mirrored)."""
    if self._strategy is None:
      if self._tpu is not None:
        resolver = distribute_utils.tpu_initialize(self._tpu)
        self._strategy = tf.distribute.experimental.TPUStrategy(resolver)
      elif self._distribution_strategy is None or self._distribution_strategy == 'default':
        self._strategy = tf.distribute.get_strategy()
      elif self._distribution_strategy == 'cpu':
        self._strategy = tf.distribute.OneDeviceStrategy('/device:cpu:0')
      else:
        if self._distribution_strategy == 'mirrored':
          self._strategy = tf.distribute.MirroredStrategy()
        else:
          raise ValueError(
              f'Invalid distribution strategy="{self._distribution_strategy}"')

  def _build_tokenizer(self):
    """Build the correct tokenizer depending on model encoder.

    Returns:
      Tokenizer for model
    """
    if self._model_config.tokenizer == 'basic':
      base_tokenizer = tfds.deprecated.text.Tokenizer()
      return tokenizers.ReservedTokenizer(
          tokenizer=base_tokenizer, reserved_re=preprocessing.SEPARATOR_RE)
    elif self._model_config.tokenizer == 'bert':
      return tokenizers.BertTokenizer(
          vocab_file=self._model_config.bert_vocab_path, do_lower_case=True)
    else:
      raise ValueError('Invalid tokenizer')

  def _build_encoder(self, vocab,
                     tokenizer):
    """Build the encoder using the given vocab and tokenizer.

    Args:
      vocab: Vocab to build encoder from
      tokenizer: Tokenizer to build encoder from

    Returns:
      The built text encoder
    """
    if self._model_config.text_encoder == 'basic':
      return preprocessing.BasicTextEncoder(
          vocab_list=vocab,
          tokenizer=tokenizer,
          lowercase=self._model_config.basic_lowercase,
          include_title=self._model_config.include_title,
          include_sentence_id=self._model_config.include_sentence_id,
          max_claim_tokens=self._model_config.max_claim_tokens,
          max_evidence_tokens=self._model_config.max_evidence_tokens,
      )
    elif self._model_config.text_encoder == 'bert':
      return preprocessing.BertTextEncoder(
          tokenizer=tokenizer,
          max_seq_length=self._model_config.bert_max_seq_length,
          include_title=self._model_config.include_title,
          include_sentence_id=self._model_config.include_sentence_id,
      )

  def _encode_and_batch(self,
                        dataset,
                        train=False,
                        filter_claims=True,
                        filter_evidence=True):
    """Convert a tensorflow dataset of unbatched, text examples to TF batches.

    Args:
      dataset: TF Dataset to transform
      train: Whether to encode as training dataset
      filter_claims: Whether to filter zero length claims
      filter_evidence: Whether to filter zero length evidence

    Returns:
      encoded and batched dataset for keras fit
    """
    encoded = self._encode_dataset(
        dataset, filter_claims=filter_claims, filter_evidence=filter_evidence)
    if train:
      encoded = encoded.shuffle(
          self._model_config.buffer_size, reshuffle_each_iteration=False)
    batched = self._batch_dataset(encoded)
    return batched

  def _compile(self):
    """Compile the keras model using the correct scope."""
    # pylint: disable=protected-access
    self._init_strategy()
    with self._strategy.scope():
      if self._model_config.model == 'two_tower':
        module_model = model.TwoTowerRanker(
            self.vocab_size,
            activation=self._model_config.activation,
            matcher_name=self._model_config.matcher,
            word_emb_size=self._model_config.word_emb_size,
            hidden_size=self._model_config.hidden_size,
            dropout=self._model_config.dropout,
            use_batch_norm=self._model_config.use_batch_norm,
            contextualizer=self._model_config.contextualizer,
            context_num_layers=self._model_config.context_num_layers,
            bidirectional=self._model_config.bidirectional,
            tied_encoders=self._model_config.tied_encoders,
            embedder_name=self._model_config.embedder,
            matcher_hidden_size=self._model_config.matcher_hidden_size,
            bert_model_name=self._model_config.bert_model_name,
            bert_model_path=self._model_config.bert_model_path,
            bert_trainable=self._model_config.bert_trainable,
            bert_dropout=self._model_config.bert_dropout,
            projection_dim=self._model_config.projection_dim,
            classify_claim=self._model_config.classify_claim,
        )
        self._inner_model = module_model
        # This hackery is necessary since keras doesn't handle dictionary inputs
        # well, so we have to manually specify input/output output shapes. Since
        # this is dependent on the model (e.g., bert vs other), let the encoder
        # yield this.
        inputs = self._encoder.compute_input_shapes()
        outputs = module_model(inputs)
        module_model.input_names = sorted(inputs.keys())
        module_model._feed_input_names = sorted(inputs.keys())
        module_model.output_names = sorted(
            ['evidence_matching', 'claim_classification'])
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.input_names = sorted(inputs.keys())
        self._model._feed_input_names = sorted(inputs.keys())
        self._model.output_names = sorted(
            ['evidence_matching', 'claim_classification'])
        self._model.summary(line_length=500)
      elif self._model_config.model == 'one_tower':
        raise NotImplementedError()
      else:
        raise ValueError('Invalid model')
      metrics = {}
      evidence_metrics = [
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalsePositives(name='fn'),
      ]
      metrics['evidence_matching'] = evidence_metrics

      loss = {}
      loss['evidence_matching'] = losses.WeightedBinaryCrossentropyFromProbs(
          positive_class_weight=self._model_config.positive_class_weight)

      loss_weights = {
          'evidence_matching': 1.0,
          'claim_classification': self._model_config.claim_loss_weight
      }
      if self._model_config.classify_claim:
        # TODO(perodriguez): add claim classifier metrics
        claim_metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        ]
        metrics['claim_classification'] = claim_metrics
        loss[
            'claim_classification'] = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)
      else:
        loss['claim_classification'] = losses.ZeroLoss()
        metrics['claim_classification'] = []
      self._model.compile(
          loss=loss,
          optimizer=tf.keras.optimizers.Adam(self._model_config.learning_rate),
          metrics=metrics,
          loss_weights=loss_weights,
      )

  def train(self,
            *,
            epochs = None,
            steps_per_epoch = None,
            validation_steps = None):
    """Prepare the dataset, callbacks, and model, then train/save it.

    Args:
      epochs: The number of epochs to train for, if None then default to
        early stopping (useful for debugging)
      steps_per_epoch: How many training steps to take, if None default to
        normal training (useful for debugging)
      validation_steps: How many validation steps to take, if None defualt to
        normal training (useful for debugging)
    """
    logging.info('Preparing model with config:\n%s', self._model_config)
    with util.log_time('Initial dataset read'):
      builder = fever_tfds.FeverEvidence(
          data_dir=self._model_config.dataset,
          n_similar_negatives=self._model_config.n_similar_negatives,
          n_background_negatives=self._model_config.n_background_negatives,
          train_scrape_type=self._model_config.scrape_type,
          include_not_enough_info=self._model_config.include_not_enough_info,
          title_in_scoring=self._model_config.title_in_scoring,
      )
      # Cache here to prevent hitting remote fs again
      train_dataset = (builder.as_dataset(split='train')).cache()
    val_dataset = builder.as_dataset(split='validation').cache()
    if self._debug:
      train_dataset = train_dataset.take(1000)
    if self._debug:
      val_dataset = val_dataset.take(200)

    self._tokenizer = self._build_tokenizer()
    self._vocab = list(self._build_vocab(train_dataset))
    self._encoder = self._build_encoder(self._vocab, self._tokenizer)

    train_batched = self._encode_and_batch(train_dataset, train=True)
    val_batched = self._encode_and_batch(val_dataset, train=False)
    # Cache the batch creation, but not the batchwise shuffle.
    train_batched = train_batched.cache().shuffle(
        100,
        reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)
    # Cache the batched validation data.
    val_batched = val_batched.cache().prefetch(tf.data.experimental.AUTOTUNE)
    self._compile()
    model_callbacks = self._build_callbacks(val_batched)
    # Save enough to reconstruct anything except for the model.
    # The model itself is saved with the ModelCheckpoint callback.
    self._save_model_config()
    self._save_encoder()
    if epochs is None:
      epochs = self._model_config.max_epochs

    self._model.fit(
        train_batched,
        validation_data=val_batched,
        callbacks=model_callbacks,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)
    logging.info('Model Summary:\n%s', self._model.summary())
    # First load the best model.
    logging.info('Loading best model weights')
    self._model.load_weights(self.model_weight_path)
    logging.info('Saving dev predictions from best model')
    self._save_dev_predictions(val_batched)

  @property
  def model_weight_path(self):
    return os.path.join(self._model_config.model_checkpoint, 'best_model.tf')

  def _save_dev_predictions(self, val_batched):
    """Save model predictions for the dev set.

    This is used to compute Fever F1 as stopping metric

    Args:
      val_batched: The batched validation set.
    """
    unbatched = val_batched.unbatch()
    model_predictions = self._model.predict(val_batched)
    claim_probs = model_predictions['claim_classification']
    evidence_probs = model_predictions['evidence_matching']
    predictions = []
    # Extra _ is the label, which we don't need
    for (ex, _), claim_prob, evidence_prob in tqdm.tqdm(
        zip(unbatched, claim_probs, evidence_probs), mininterval=5):
      predictions.append({
          'claim_prob': claim_prob.tolist(),
          'evidence_prob': evidence_prob.tolist(),
          'metadata': json.loads(ex['metadata'].numpy().decode('utf8'))
      })
    pred_path = os.path.join(self._model_config.model_checkpoint,
                             'val_predictions.json')
    with util.safe_open(pred_path, 'w') as f:
      json.dump({'predictions': predictions}, f)


  def predict(self, examples):
    """Given examples in JSON format, predict evidence relevance.

    Args:
      examples: List of claim/evidence pairs to rank

    Returns:
      Scalar scores for each pair
    """
    stacked = {
        'claim_text': [],
        'evidence_text': [],
        'metadata': [],
        'label': [],
    }
    for ex in examples:
      stacked['claim_text'].append(ex['claim_text'])
      stacked['evidence_text'].append(ex['evidence_text'])
      stacked['metadata'].append(ex['metadata'])
      stacked['label'].append(ex['label'])

    dataset = tf.data.Dataset.from_tensor_slices((stacked,))
    batched_examples = self._encode_and_batch(
        dataset, filter_claims=False, filter_evidence=False)
    preds = []
    for batch in batched_examples:
      # model.predict() is broken after model load so we have to do this
      # manually.
      preds.append(self._model(batch))
    return np.vstack(preds).reshape(-1).tolist()

  def embed(self, examples, *, as_claim,
            as_evidence):  # Checker .tolist() -> Any
    """Embed a list of evidence text.

    Args:
      examples: A list of evidence text to embed.
      as_claim: Whether to embed examples as claims
      as_evidence: Whether to embed examples as evidence

    Returns:
      A list of embeddings, one for each evidence text.

    """
    stacked = {
        'claim_text': [],
        'evidence_text': [],
        'metadata': [],
        'label': [],
    }
    for text in examples:
      # Dummie value to make sure tokenizing works.
      if as_claim:
        stacked['claim_text'].append(text)
      else:
        stacked['claim_text'].append('a')
      if as_evidence:
        stacked['evidence_text'].append(text)
      else:
        stacked['evidence_text'].append('a')
      stacked['metadata'].append('')
      stacked['label'].append(tf.constant(0, dtype=tf.int64))

    dataset = tf.data.Dataset.from_tensor_slices((stacked,))
    batched_examples = self._encode_and_batch(
        dataset, filter_claims=False, filter_evidence=False)
    claim_preds = []
    ev_preds = []
    for batch in batched_examples:
      # model.predict() is broken after model load due to missing shapes, so
      # have to do our own batching/unbatching.
      inputs, _ = batch
      claim_encoding, ev_encoding = self._model(
          inputs, embed_claim=as_claim, embed_evidence=as_evidence)
      claim_preds.append(claim_encoding)
      ev_preds.append(ev_encoding)
    return np.vstack(claim_preds).tolist(), np.vstack(ev_preds).tolist()

  def embed_wiki_dataset(self, dataset):
    """Embed the wikipedia/evidence only dataset.

    Args:
      dataset: The wikipedia only dataset (e.g. wiki_tfds.py)

    Returns:
      Aligned wikipedia_urls, sentence_ids, and embeddings of model
    """

    # map_fn and tf_map_fn transform the dataset to the same format as
    # tfds_evidence/the one the model expects
    def map_fn(text, wikipedia_url, sentence_id):
      return ('a', text, wikipedia_url, str(sentence_id),
              json.dumps({
                  'sentence_id': int(sentence_id.numpy()),
                  'wikipedia_url': wikipedia_url.numpy().decode('utf8')
              }))

    def tf_map_fn(example):
      tensors = tf.py_function(
          map_fn,
          inp=[
              example['text'], example['wikipedia_url'], example['sentence_id']
          ],
          Tout=(tf.string, tf.string, tf.string, tf.string, tf.string))
      return {
          'claim_text': tensors[0],
          'evidence_text': tensors[1],
          'wikipedia_url': tensors[2],
          'sentence_id': tensors[3],
          'claim_label': tf.constant(0, dtype=tf.int64),
          'evidence_label': tf.constant(0, dtype=tf.int64),
          'metadata': tensors[4]
      }

    formatted_ds = dataset.map(tf_map_fn)
    batched_examples = self._encode_and_batch(
        formatted_ds, filter_claims=False, filter_evidence=False)
    preds = []
    wikipedia_urls = []
    sentence_ids = []
    for batch in tqdm.tqdm(batched_examples, mininterval=5):
      # model.predict() is broken after model load due to missing shapes, so
      # have to do our own batching/unbatching.
      inputs, _ = batch
      _, ev_encoding = self._inner_model(
          inputs, embed_claim=False, embed_evidence=True)
      for m in inputs['metadata'].numpy():
        key = json.loads(m.decode('utf8'))
        wikipedia_urls.append(key['wikipedia_url'])
        sentence_ids.append(key['sentence_id'])
      preds.append(ev_encoding)

    return np.array(wikipedia_urls), np.array(sentence_ids), np.vstack(preds)

  def embed_claim_dataset(self, dataset):
    """Embed the claim only dataset and save them with claim_ids.

    Args:
      dataset: The claims only dataset (e.g. claim_tfds.py)

    Returns:
      Aligned claim ids and embeddings from the model
    """
    batched_examples = self._encode_and_batch(
        dataset, filter_claims=False, filter_evidence=False)
    claim_ids = []
    embeddings = []
    for batch in tqdm.tqdm(batched_examples, mininterval=5):
      # model.predict() is broken after model load due to missing shapes, so
      # have to do our own batching/unbatching.
      inputs, _ = batch
      # Cannot use self._model since it does not take extra arguments. Since
      # we're not using the keras API (namey .predict()), we can just use the
      # underlying model stored in self._inner_model.
      claim_encoding, _ = self._inner_model(
          inputs, embed_claim=True, embed_evidence=False)
      for m in inputs['metadata'].numpy():
        key = json.loads(m.decode('utf8'))
        claim_ids.append(int(key['claim_id']))
      embeddings.append(claim_encoding)

    return np.array(claim_ids), np.vstack(embeddings)

  def _build_callbacks(self, val_batched):
    """Build the callbacks used during training."""
    cns_model_checkpoint = util.safe_path(
        os.path.join(self._model_config.model_checkpoint, 'best_model.tf'))
    model_callbacks = [
        # Note: Order matters here, particularly that FeverMetricsCallback
        # comes before tensorboard so it can write to the log dictionary
        # and TB picks it up.
        callbacks.FeverMetricsCallback(
            validation_batched=val_batched,
            debug=self._debug,
            fever_dev_path=self._model_config.fever_dev_path,
            max_evidence=self._model_config.max_evidence,
            checkpoint_dir=self._model_config.model_checkpoint,
        ),
        # TODO(perodriguez): Determine a better thing to stop on
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=.001,
            patience=3,
            verbose=1,
            mode='min'),
        # TODO(perodriguez): Determine a better thing to save on
        # Checkpointing also needs to know about fever recall.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cns_model_checkpoint,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1,
            # There is no support for GRU/LSTM Dropout with normal save
            save_weights_only=True,
        ),
    ]

    if self._tb_log_dir is not None:
      model_callbacks.append(
          tf.keras.callbacks.TensorBoard(log_dir=self._tb_log_dir))
    return model_callbacks

  def _batch_dataset(self, dataset):
    """Batch the dataset depending on what model is used.

    Args:
      dataset: A dataset to batch

    Returns:
      A batched dataset with correct padding shapes.
    """
    return dataset.padded_batch(
        batch_size=self._model_config.batch_size,
        padded_shapes=(
            self._encoder.padded_shapes(),
            # Must match losses in training.py
            {
                'claim_classification': [],
                'evidence_matching': []
            }))

  def _encode_dataset(self,
                      dataset,
                      filter_claims=True,
                      filter_evidence=True):
    """Convert the tfds dataset to numbers by tokenizing/embedding."""
    encode = self._encoder.build_encoder_fn()
    encoded_data = dataset.map(
        encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if filter_claims:
      encoded_data = encoded_data.filter(preprocessing.filter_claim_fn)
    if filter_evidence:
      encoded_data = encoded_data.filter(preprocessing.filter_evidence_fn)

    return encoded_data

  def _build_vocab(self, dataset):
    """Build the vocabulary and encoder from the dataset.

    Args:
      dataset: The dataset to build vocab from.

    Returns:
      The vocabulary in the dataset, or empty vocab if using bert
    """
    # If we are using bert, then we do not need to build the vocab
    # since its already defined
    if self._model_config.tokenizer == 'bert' and self._model_config.text_encoder == 'bert':
      logging.info('Using bert, skipping vocabulary creation')
      return set()

    if self._tokenizer is None:
      raise ValueError('Cannot build vocab without a tokenizer.')
    claim_lengths = []
    evidence_lengths = []
    vocab = set()
    for example in tqdm.tqdm(dataset, mininterval=5):
      tokenized_claim, tokenized_evidence = self._tokenize_example(example)
      claim_lengths.append(len(tokenized_claim))
      evidence_lengths.append(len(tokenized_evidence))
      vocab.update(tokenized_claim)
      vocab.update(tokenized_evidence)
    logging.info('Build vocab of size (without padding): %s', len(vocab))
    logging.info('Claim length statistics')
    logging.info('Max: %s', max(claim_lengths))
    logging.info('Min: %s', min(claim_lengths))
    claim_percentiles = np.percentile(claim_lengths, [50, 90, 95, 99]).tolist()
    logging.info('50/90/95/99: %s', str(claim_percentiles))
    logging.info('Evidence length statistics')
    logging.info('Max: %s', max(evidence_lengths))
    logging.info('Min: %s', min(evidence_lengths))
    evidence_percentiles = np.percentile(evidence_lengths,
                                         [50, 90, 95, 99]).tolist()
    logging.info('50/90/95/99: %s', str(evidence_percentiles))
    self._vocab_stats['claim_max'] = max(claim_lengths)
    self._vocab_stats['claim_min'] = min(claim_lengths)
    self._vocab_stats['claim_percentiles'] = claim_percentiles
    self._vocab_stats['evidence_max'] = max(evidence_lengths)
    self._vocab_stats['evidence_min'] = min(evidence_lengths)
    self._vocab_stats['evidence_percentiles'] = evidence_percentiles
    return vocab

  def _tokenize_example(self, example):
    tokenized_claim = self._tokenizer.tokenize(
        example['claim_text'].numpy().decode('utf8'))
    tokenized_evidence = self._tokenizer.tokenize(
        example['evidence_text'].numpy().decode('utf8'))
    return tokenized_claim, tokenized_evidence

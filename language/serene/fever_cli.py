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
"""Fever top level CLI tool."""

import os
import pathlib


from absl import app
from absl import flags
from absl import logging
from apache_beam.runners.direct import direct_runner
from language.serene import boolq_tfds
from language.serene import claim_tfds
from language.serene import config
from language.serene import constants
from language.serene import fever_tfds
from language.serene import layers
from language.serene import scrape_db
from language.serene import text_matcher
from language.serene import training
from language.serene import util
from language.serene import wiki_db
from language.serene import wiki_tfds
import matplotlib
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
matplotlib.use('TKAgg')



def train_model(
    *,
    model_config,
    debug = False,
    tb_log_dir = None,
    distribution_strategy = None,
    tpu = None):
  """Train an evidence matching model.

  Args:
    model_config: Set of parameters used for ModelConfig
    debug: Whether to enable debug features
    tb_log_dir: Where, if any, to log to tb
    distribution_strategy: CPU/GPU/TPU
    tpu: TPU config, if using TPU
  """
  trainer = training.Trainer(
      model_config,
      debug=debug,
      tpu=tpu,
      distribution_strategy=distribution_strategy,
      tb_log_dir=tb_log_dir)
  if debug:
    steps_per_epoch = 5
    validation_steps = 5
    epochs = 1
  else:
    steps_per_epoch = None
    validation_steps = None
    epochs = None
  trainer.train(
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps)


def embed_wiki(*, data_dir, model_checkpoint, shard,
               num_shards):
  """Embed wikipedia using the given model checkpoint.

  Args:
    data_dir: Data directory for Wiki TFDS dataset
    model_checkpoint: Checkpoint of model to use
    shard: The shard to embed, intention is to run this in parallel
    num_shards: Number of shards to write.
  """
  trainer = training.Trainer.load(model_checkpoint)
  wiki_builder = wiki_tfds.WikipediaText(data_dir=data_dir)
  wiki_sents = wiki_builder.as_dataset(split='validation').shard(
      num_shards=num_shards, index=shard)
  wikipedia_urls, sentence_ids, encodings = trainer.embed_wiki_dataset(
      wiki_sents)
  enc_path = os.path.join(model_checkpoint, 'wikipedia',
                          f'embeddings_{shard}_{num_shards}.npy')
  with util.safe_open(enc_path, 'wb') as f:
    np.save(f, encodings, allow_pickle=False)
  wiki_url_path = os.path.join(model_checkpoint, 'wikipedia',
                               f'urls_{shard}_{num_shards}.npy')
  with util.safe_open(wiki_url_path, 'wb') as f:
    np.save(f, wikipedia_urls, allow_pickle=False)
  sentence_id_path = os.path.join(model_checkpoint, 'wikipedia',
                                  f'sentence_ids_{shard}_{num_shards}.npy')
  with util.safe_open(sentence_id_path, 'wb') as f:
    np.save(f, sentence_ids, allow_pickle=False)


def embed_claims(*, claim_tfds_data, train_claim_ids_npy_filename,
                 train_embeddings_npy_filename,
                 val_claim_ids_npy_filename,
                 val_embeddings_npy_filename, model_checkpoint):
  """Embed the claims using the given model checkpoint.

  Args:
    claim_tfds_data: path to claim tfds data
    train_claim_ids_npy_filename: Path to write train claim ids to
    train_embeddings_npy_filename: Path to write train embeddings to
    val_claim_ids_npy_filename: Path to write validation ids to
    val_embeddings_npy_filename: Path to write validation embeddings to
    model_checkpoint: The checkpoint of the model to use for embedding
  """
  logging.info('Loading model')
  trainer = training.Trainer.load(model_checkpoint)
  logging.info('Building claim datasets')
  claim_builder = claim_tfds.ClaimDataset(data_dir=claim_tfds_data)
  train_claims = claim_builder.as_dataset(split='train')
  val_claims = claim_builder.as_dataset(split='validation')
  logging.info('Embedding claims')
  train_claim_ids, train_embeddings = trainer.embed_claim_dataset(train_claims)
  val_claim_ids, val_embeddings = trainer.embed_claim_dataset(val_claims)

  out_dir = pathlib.Path(model_checkpoint) / 'claims'
  train_claim_id_path = out_dir / train_claim_ids_npy_filename
  train_emb_path = out_dir / train_embeddings_npy_filename
  val_claim_id_path = out_dir / val_claim_ids_npy_filename
  val_emb_path = out_dir / val_embeddings_npy_filename

  with util.safe_open(train_claim_id_path, 'wb') as f:
    np.save(f, train_claim_ids, allow_pickle=False)

  with util.safe_open(train_emb_path, 'wb') as f:
    np.save(f, train_embeddings, allow_pickle=False)

  with util.safe_open(val_claim_id_path, 'wb') as f:
    np.save(f, val_claim_ids, allow_pickle=False)

  with util.safe_open(val_emb_path, 'wb') as f:
    np.save(f, val_embeddings, allow_pickle=False)


def preprocess(
    *,
    common_config,
    scrape_type,
    data_dir,
    download_dir):
  """Preprocess the fever data to the TFDS Fever data.

  Args:
    common_config: Common configuration from config.Config
    scrape_type: Which scrape to use, drqa/lucene/ukp, in training
    data_dir: Where to write data to
    download_dir: Where to download data to, unused but required by TFDS API
  """
  logging.info('Creating fever dataset builder')
  text_matcher_params_path = common_config.text_matcher_params
  fever_train_path = common_config.fever_train
  fever_dev_path = common_config.fever_dev
  fever_test_path = common_config.fever_test

  ukp_docs_train = common_config.ukp_docs_train
  ukp_docs_dev = common_config.ukp_docs_dev
  ukp_docs_test = common_config.ukp_docs_test
  builder = fever_tfds.FeverEvidence(
      wiki_db_path=common_config.wikipedia_db_path,
      text_matcher_params_path=text_matcher_params_path,
      fever_train_path=fever_train_path,
      fever_dev_path=fever_dev_path,
      fever_test_path=fever_test_path,
      drqa_db_path=common_config.drqa_scrape_db_path,
      lucene_db_path=common_config.lucene_scrape_db_path,
      data_dir=data_dir,
      n_similar_negatives=common_config.n_similar_negatives,
      n_background_negatives=common_config.n_background_negatives,
      ukp_docs_train=ukp_docs_train,
      ukp_docs_dev=ukp_docs_dev,
      ukp_docs_test=ukp_docs_test,
      train_scrape_type=scrape_type,
      title_in_scoring=common_config.title_in_scoring,
      n_inference_candidates=common_config.n_inference_candidates,
      include_not_enough_info=common_config.include_not_enough_info,
      n_inference_documents=common_config.n_inference_documents,
      max_inference_sentence_id=common_config.max_inference_sentence_id,
  )

  logging.info('Preparing fever evidence dataset')
  beam_runner = direct_runner.DirectRunner()
  download_config = tfds.download.DownloadConfig(beam_runner=beam_runner,)
  builder.download_and_prepare(
      download_dir=download_dir, download_config=download_config)


def wiki_preprocess(*,
                    common_config,
                    data_dir,
                    download_dir,
                    max_sentence_id = 30):
  """Preprocess wikipedia dump to TFDS format.

  Args:
    common_config: Configuration
    data_dir: Where to write data to
    download_dir: Where to download data to, unused but required by TFDS API
    max_sentence_id: The max sentence_id to take on each wikipedia page
  """
  logging.info('Creating wikipedia dataset builder')
  wiki_db_path = common_config.wikipedia_db_path
  builder = wiki_tfds.WikipediaText(
      wiki_db_path=wiki_db_path,
      data_dir=data_dir,
      max_sentence_id=max_sentence_id,
  )

  logging.info('Preparing wikipedia dataset')
  download_config = tfds.download.DownloadConfig(
      beam_runner=runner.FlumeRunner(),)
  builder.download_and_prepare(
      download_dir=download_dir, download_config=download_config)


def claim_preprocess(*, common_config, data_dir,
                     download_dir):
  """Preprocess only claims TFDS format.

  Args:
    common_config: Common global config
    data_dir: Where to write data to
    download_dir: Where to download data to, unused but required by TFDS API
  """
  logging.info('Creating claim dataset builder')
  builder = claim_tfds.ClaimDataset(
      fever_train_path=common_config.fever_train,
      fever_dev_path=common_config.fever_dev,
      data_dir=data_dir,
  )
  download_config = tfds.download.DownloadConfig()
  builder.download_and_prepare(
      download_dir=download_dir, download_config=download_config)


def boolq_preprocess(*, common_config, data_dir,
                     download_dir):
  """Preprocess boolq as fever-like claims TFDS format.

  Args:
    common_config: Common global config
    data_dir: Where to write data to
    download_dir: Where to download data to, unused but required by TFDS API
  """
  logging.info('Creating claim dataset builder')
  builder = boolq_tfds.BoolQClaims(
      boolq_train_path=common_config.boolq_train,
      boolq_dev_path=common_config.boolq_dev,
      data_dir=data_dir,
  )
  download_config = tfds.download.DownloadConfig()
  builder.download_and_prepare(
      download_dir=download_dir, download_config=download_config)




FLAGS = flags.FLAGS
flags.DEFINE_enum('command', None, [
    'preprocess',
    'train_model',
    'wiki_preprocess',
    'claim_preprocess',
    'boolq_preprocess',
    'embed_wiki',
    'embed_claims',
    'cache_wikipedia',
    'load_wikipedia',
    'cache_scrapes',
], 'The sub-command to run')
flags.DEFINE_bool('debug', False,
                  'Enable debug mode for functions that support it')
flags.DEFINE_integer('seed', None, 'random seed to set')
# train flags
flags.DEFINE_string('tpu', None, 'TPU configuration')
flags.DEFINE_string('tb_log_root', None, 'Tensorboard logging directory')
flags.DEFINE_string('distribution_strategy', None,
                    'TF distribution, cpu/gpu/tpu')
# model configuration flags
flags.DEFINE_string('model_checkpoint_root', '', 'The root for saving models.')
flags.DEFINE_integer('buffer_size', 1_000, 'Buffer size for tf.data.Dataset')
flags.DEFINE_integer('experiment_id', None, '')
flags.DEFINE_integer('work_unit_id', None, '')

# model hyper parameters
flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_integer('word_emb_size', 300, 'Word embedding size')
flags.DEFINE_integer('hidden_size', 100, 'LSTM hidden state size')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_float('positive_class_weight', None, 'Weight for positive class')
flags.DEFINE_integer('max_epochs', 50, 'Max number of epochs to train for')
flags.DEFINE_float('dropout', .5, 'Dropout rate')
flags.DEFINE_float('bert_dropout', .1,
                   'Dropout rate on final embeddings computed by BERT')
flags.DEFINE_enum('activation', 'gelu', ['gelu', 'relu', 'elu'],
                  'Activation function for non-linear layers')
flags.DEFINE_bool('use_batch_norm', True, 'Whether to use batch norm')
flags.DEFINE_enum('tokenizer', 'basic', ['basic', 'bert'],
                  'Which tokenizer to use')
flags.DEFINE_enum('text_encoder', 'basic', ['basic', 'bert'],
                  'Which text encoder (text to int) to use')
flags.DEFINE_bool('basic_lowercase', True,
                  'Whether basic encoder should lowercase input')
flags.DEFINE_enum('matcher', 'product_matcher',
                  list(layers.matcher_registry.keys()),
                  'How to compare claims and evidence for evidence matching')
flags.DEFINE_integer('matcher_hidden_size', 200,
                     'Size of hidden size in matcher, if it has one')
flags.DEFINE_enum('embedder', 'classic_embedder',
                  ['classic_embedder', 'bert_embedder'],
                  'Which embedder to use (word indices -> embeddings)')
flags.DEFINE_enum(
    'contextualizer', 'gru', ['gru', 'rnn', 'lstm', 'bert'],
    'What type of contextualizer to use in claim/evidence encoder')
flags.DEFINE_integer('context_num_layers', 1,
                     'Number of GRU/LSTM/RNN layers in contextualizer')
flags.DEFINE_bool('tied_encoders', True,
                  'Whether to tie claim/evidence encoder parameters')
flags.DEFINE_bool('bidirectional', True,
                  'Whether to make GRU/LSTM/RNN bidirectional.')
flags.DEFINE_enum(
    'model', 'two_tower', ['one_tower', 'two_tower'],
    'Which type of model to use, a no-op since one_tower is not implemented')
flags.DEFINE_enum('bert_model_name', 'base', ['base', 'large'],
                  'Type of bert to use')
flags.DEFINE_integer('bert_max_seq_length', 100, 'max seq length for bert')
flags.DEFINE_string('inference_model_checkpoint', None,
                    'Checkpoint of model to run inference with')
flags.DEFINE_integer('inference_shard', None, 'Inference shard for current job')
flags.DEFINE_integer('inference_num_shards', None,
                     'Total number of inference shards')
flags.DEFINE_integer(
    'projection_dim', -1,
    'Dimension to project output of embedder to. If -1, do not project')
flags.DEFINE_bool('include_title', True, 'Whether to prepend title to evidence')
flags.DEFINE_bool('include_sentence_id', False,
                  'Whether to prepend sentence_id to evidence')
flags.DEFINE_bool('bert_trainable', True, 'Whether bert params are trainable.')
flags.DEFINE_enum('scrape_type', constants.UKP_PRED, constants.DOC_TYPES, '')
flags.DEFINE_bool(
    'classify_claim', False,
    'Whether to classify claims as support/refute/not enough info')
flags.DEFINE_integer(
    'n_inference_candidates', None,
    'The maximum number of sentences to return for each claim during inference')
flags.DEFINE_integer(
    'n_inference_documents', None,
    'The maximum number of documents to generate sentences from during inference'
)
flags.DEFINE_bool('include_not_enough_info', None,
                  'Whether to include not enough information claims')
flags.DEFINE_bool('title_in_scoring', None,
                  'Whether to concat titles to evidence in tfidf scoring')

# These are intentionally set to None, they must all be defined and consistent
# Across command invocations. The default values are set in config.py
flags.DEFINE_string('fever_train', None, '')
flags.DEFINE_string('fever_dev', None, '')
flags.DEFINE_string('fever_test', None, '')
flags.DEFINE_string('boolq_train', None, '')
flags.DEFINE_string('boolq_dev', None, '')
flags.DEFINE_string('lucene_train_scrapes', None, '')
flags.DEFINE_string('lucene_dev_scrapes', None, '')
flags.DEFINE_string('lucene_test_scrapes', None, '')
flags.DEFINE_string('drqa_train_scrapes', None, '')
flags.DEFINE_string('drqa_dev_scrapes', None, '')
flags.DEFINE_string('drqa_test_scrapes', None, '')
flags.DEFINE_string('text_matcher_params', None, '')
flags.DEFINE_string('bert_checkpoint', None, '')
flags.DEFINE_string('fever_evidence_tfds_data', None, '')
flags.DEFINE_string('fever_evidence_tfds_download', None, '')
flags.DEFINE_string('wiki_tfds_data', None, '')
flags.DEFINE_string('wiki_tfds_download', None, '')
flags.DEFINE_string('claim_tfds_data', None, '')
flags.DEFINE_string('claim_tfds_download', None, '')
flags.DEFINE_string('boolq_tfds_data', None, '')
flags.DEFINE_string('boolq_tfds_download', None, '')
flags.DEFINE_string('bert_base_uncased_model', None, '')
flags.DEFINE_string('bert_large_uncased_model', None, '')
flags.DEFINE_string('bert_base_uncased_vocab', None, '')
flags.DEFINE_string('bert_large_uncased_vocab', None, '')
flags.DEFINE_string('train_claim_ids_npy', None, '')
flags.DEFINE_string('train_embeddings_npy', None, '')
flags.DEFINE_string('val_claim_ids_npy', None, '')
flags.DEFINE_string('val_embeddings_npy', None, '')
flags.DEFINE_integer('max_claim_tokens', None, '')
flags.DEFINE_integer('max_evidence_tokens', None, '')
flags.DEFINE_integer('max_evidence', None, '')
flags.DEFINE_integer('n_similar_negatives', None, '')
flags.DEFINE_integer('n_background_negatives', None, '')
flags.DEFINE_string('wikipedia_db_path', None, '')
flags.DEFINE_string('lucene_scrape_db_path', None, '')
flags.DEFINE_string('drqa_scrape_db_path', None, '')
flags.DEFINE_string('ukp_docs_train', None, '')
flags.DEFINE_string('ukp_docs_dev', None, '')
flags.DEFINE_string('ukp_docs_test', None, '')
flags.DEFINE_integer('max_inference_sentence_id', None, '')
flags.DEFINE_float('claim_loss_weight', None, '')


def main(_):
  flags.mark_flag_as_required('command')
  tf.enable_v2_behavior()
  # Parse the common flags from FLAGS
  common_flags = {
      'fever_train': FLAGS.fever_train,
      'fever_dev': FLAGS.fever_dev,
      'fever_test': FLAGS.fever_test,
      'lucene_train_scrapes': FLAGS.lucene_train_scrapes,
      'lucene_dev_scrapes': FLAGS.lucene_dev_scrapes,
      'lucene_test_scrapes': FLAGS.lucene_test_scrapes,
      'lucene_scrape_db_path': FLAGS.lucene_scrape_db_path,
      'drqa_train_scrapes': FLAGS.drqa_train_scrapes,
      'drqa_dev_scrapes': FLAGS.drqa_dev_scrapes,
      'drqa_test_scrapes': FLAGS.drqa_test_scrapes,
      'drqa_scrape_db_path': FLAGS.drqa_scrape_db_path,
      'text_matcher_params': FLAGS.text_matcher_params,
      'bert_checkpoint': FLAGS.bert_checkpoint,
      'fever_evidence_tfds_data': FLAGS.fever_evidence_tfds_data,
      'fever_evidence_tfds_download': FLAGS.fever_evidence_tfds_download,
      'wiki_tfds_data': FLAGS.wiki_tfds_data,
      'wiki_tfds_download': FLAGS.wiki_tfds_download,
      'claim_tfds_data': FLAGS.claim_tfds_data,
      'claim_tfds_download': FLAGS.claim_tfds_download,
      'boolq_tfds_data': FLAGS.boolq_tfds_data,
      'boolq_tfds_download': FLAGS.boolq_tfds_download,
      'bert_base_uncased_model': FLAGS.bert_base_uncased_model,
      'bert_large_uncased_model': FLAGS.bert_large_uncased_model,
      'bert_base_uncased_vocab': FLAGS.bert_base_uncased_vocab,
      'bert_large_uncased_vocab': FLAGS.bert_large_uncased_vocab,
      'train_claim_ids_npy': FLAGS.train_claim_ids_npy,
      'train_embeddings_npy': FLAGS.train_embeddings_npy,
      'val_claim_ids_npy': FLAGS.val_claim_ids_npy,
      'val_embeddings_npy': FLAGS.val_embeddings_npy,
      'max_claim_tokens': FLAGS.max_claim_tokens,
      'max_evidence_tokens': FLAGS.max_evidence_tokens,
      'max_evidence': FLAGS.max_evidence,
      'n_similar_negatives': FLAGS.n_similar_negatives,
      'n_background_negatives': FLAGS.n_background_negatives,
      'wikipedia_db_path': FLAGS.wikipedia_db_path,
      'ukp_docs_train': FLAGS.ukp_docs_train,
      'ukp_docs_dev': FLAGS.ukp_docs_dev,
      'ukp_docs_test': FLAGS.ukp_docs_test,
      'boolq_train': FLAGS.boolq_train,
      'boolq_dev': FLAGS.boolq_dev,
      'n_inference_candidates': FLAGS.n_inference_candidates,
      'n_inference_documents': FLAGS.n_inference_documents,
      'include_not_enough_info': FLAGS.include_not_enough_info,
      'title_in_scoring': FLAGS.title_in_scoring,
      'max_inference_sentence_id': FLAGS.max_inference_sentence_id,
      'claim_loss_weight': FLAGS.claim_loss_weight,
  }
  # Remove anything that is not defined and fallback to defaults
  common_flags = {k: v for k, v in common_flags.items() if v is not None}
  # Create configuration from non-None flags and passthrough everywhere.
  common_config = config.Config(**common_flags)
  if FLAGS.command == 'preprocess':
    preprocess(
        common_config=common_config,
        data_dir=common_config.fever_evidence_tfds_data,
        download_dir=common_config.fever_evidence_tfds_download,
        scrape_type=FLAGS.scrape_type,
    )
  elif FLAGS.command == 'embed_wiki':
    embed_wiki(
        data_dir=common_config.wiki_tfds_data,
        model_checkpoint=FLAGS.inference_model_checkpoint,
        shard=FLAGS.inference_shard,
        num_shards=FLAGS.inference_num_shards,
    )
  elif FLAGS.command == 'embed_claims':
    embed_claims(
        claim_tfds_data=common_config.claim_tfds_data,
        train_claim_ids_npy_filename=common_config.train_claim_ids_npy,
        train_embeddings_npy_filename=common_config.train_embeddings_npy,
        val_claim_ids_npy_filename=common_config.val_claim_ids_npy,
        val_embeddings_npy_filename=common_config.val_embeddings_npy,
        model_checkpoint=FLAGS.inference_model_checkpoint)
  elif FLAGS.command == 'wiki_preprocess':
    wiki_preprocess(
        common_config=common_config,
        data_dir=common_config.wiki_tfds_data,
        download_dir=common_config.wiki_tfds_download,
    )
  elif FLAGS.command == 'claim_preprocess':
    claim_preprocess(
        data_dir=common_config.claim_tfds_data,
        download_dir=common_config.claim_tfds_download,
        common_config=common_config,
    )
  elif FLAGS.command == 'boolq_preprocess':
    boolq_preprocess(
        data_dir=common_config.boolq_tfds_data,
        download_dir=common_config.boolq_tfds_download,
        common_config=common_config,
    )
  elif FLAGS.command == 'train_model':
    if FLAGS.tb_log_root is None:
      tb_log_dir = None
    else:
      tb_parts = [FLAGS.tb_log_root, FLAGS.fever_experiment_id]
      if FLAGS.experiment_id is not None and FLAGS.work_unit_id is not None:
        tb_parts.append(str(FLAGS.experiment_id))
        tb_parts.append(str(FLAGS.work_unit_id))
      tb_log_dir = os.path.join(*tb_parts)

    model_checkpoint_parts = [
        FLAGS.model_checkpoint_root,
        FLAGS.fever_experiment_id,
    ]
    if FLAGS.experiment_id is not None and FLAGS.work_unit_id is not None:
      model_checkpoint_parts.append(str(FLAGS.experiment_id))
      model_checkpoint_parts.append(str(FLAGS.work_unit_id))
    model_checkpoint = os.path.join(*model_checkpoint_parts)

    if FLAGS.projection_dim == -1:
      projection_dim = None
    else:
      projection_dim = FLAGS.projection_dim

    if FLAGS.bert_model_name == 'base':
      bert_vocab = common_config.bert_base_uncased_vocab
      bert_model_path = common_config.bert_base_uncased_model
    elif FLAGS.bert_model_name == 'large':
      bert_vocab = common_config.bert_large_uncased_vocab
      bert_model_path = common_config.bert_large_uncased_model
    else:
      raise ValueError('Invalid bert model')
    # These values must be json serializable, which is why
    # common_config: config.Config is not passed in
    model_config = training.ModelConfig(
        fever_experiment_id=FLAGS.fever_experiment_id,
        model_checkpoint=model_checkpoint,
        dataset=common_config.fever_evidence_tfds_data,
        buffer_size=FLAGS.buffer_size,
        batch_size=FLAGS.batch_size,
        word_emb_size=FLAGS.word_emb_size,
        hidden_size=FLAGS.hidden_size,
        learning_rate=FLAGS.learning_rate,
        positive_class_weight=FLAGS.positive_class_weight,
        max_epochs=FLAGS.max_epochs,
        dropout=FLAGS.dropout,
        activation=FLAGS.activation,
        use_batch_norm=FLAGS.use_batch_norm,
        tokenizer=FLAGS.tokenizer,
        text_encoder=FLAGS.text_encoder,
        basic_lowercase=FLAGS.basic_lowercase,
        embedder=FLAGS.embedder,
        contextualizer=FLAGS.contextualizer,
        tied_encoders=FLAGS.tied_encoders,
        bidirectional=FLAGS.bidirectional,
        matcher=FLAGS.matcher,
        matcher_hidden_size=FLAGS.matcher_hidden_size,
        model=FLAGS.model,
        bert_model_name=FLAGS.bert_model_name,
        bert_max_seq_length=FLAGS.bert_max_seq_length,
        bert_model_path=bert_model_path,
        bert_vocab_path=bert_vocab,
        bert_trainable=FLAGS.bert_trainable,
        bert_dropout=FLAGS.bert_dropout,
        context_num_layers=FLAGS.context_num_layers,
        projection_dim=projection_dim,
        fever_dev_path=common_config.fever_dev,
        max_evidence=common_config.max_evidence,
        max_claim_tokens=common_config.max_claim_tokens,
        max_evidence_tokens=common_config.max_evidence_tokens,
        include_title=FLAGS.include_title,
        include_sentence_id=FLAGS.include_sentence_id,
        n_similar_negatives=common_config.n_similar_negatives,
        n_background_negatives=common_config.n_background_negatives,
        include_not_enough_info=common_config.include_not_enough_info,
        scrape_type=FLAGS.scrape_type,
        classify_claim=FLAGS.classify_claim,
        title_in_scoring=common_config.title_in_scoring,
        claim_loss_weight=common_config.claim_loss_weight,
    )
    # Intentionally *not* passing in common_config, the trainer serializes for
    # save/load based on model_parameters, so everything must go  in there.
    train_model(
        model_config=model_config,
        debug=FLAGS.debug,
        tpu=FLAGS.tpu,
        distribution_strategy=FLAGS.distribution_strategy,
        tb_log_dir=tb_log_dir,
    )
  else:
    raise ValueError('Incorrect command')


if __name__ == '__main__':
  app.run(main)

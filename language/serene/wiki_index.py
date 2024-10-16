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
# pylint: disable=line-too-long
r"""Build index from wikipedia embeddings from model and save to disk.

"""
# pylint: enable=line-too-long
from concurrent import futures
import os
import time

from absl import app
from absl import flags
from absl import logging
import dataclasses
from fever_scorer.scorer import fever_score
from language.serene import config
from language.serene import constants
from language.serene import util
import numpy as np
import tensorflow.compat.v2 as tf
import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('wiki_embedding_dir', None,
                    'Directory for embedding files.')
flags.DEFINE_string('claim_id_path', None, 'File for claim ids')
flags.DEFINE_string('claim_embedding_path', None, 'File for claim embeddings')
flags.DEFINE_integer('n_shards', None, 'Number of shards in embedding_dir.')
flags.DEFINE_string('out_path', None, 'Output location for predictions.')
flags.DEFINE_bool('l2_norm', False, 'Whether to apply L2 Norm to embeddings')
flags.DEFINE_bool('copy_to_tmp', False,
                  'Whether to copy embeddings to tmp before reading them.')
flags.DEFINE_string('device', '/CPU:0', 'TF Device')
flags.DEFINE_integer('batch_size', 256, 'batch size for matrix ops')


@dataclasses.dataclass(frozen=True, eq=True)
class IndexKey:
  wikipedia_url: Text
  sentence_id: int


def copy_file_to_tmp(path_pair):
  orig_path, tmp_path = path_pair
  with util.log_time(f'Copy: {orig_path} To: {tmp_path}'):
    util.safe_copy(orig_path, tmp_path)


class ShardedEmbPaths:
  """Utility class for managing sharded embedding paths."""

  def __init__(self, *, embedding_dir, n_shards):
    """Initialize sharded paths.

    Args:
      embedding_dir: Directory of embeddings
      n_shards: Number of shards in directory
    """
    self._embedding_dir = embedding_dir
    self._n_shards = n_shards
    self._tmp_dir = os.path.join('/tmp',
                                 util.random_string(prefix='embedding-tmp'))

  def files(self, tmp = False):
    """Return the files for the sharded embeddings.

    Args:
      tmp: Whether to return paths in /tmp or not

    Returns:
      Paths of all sharded files.
    """
    sharded_files = []
    for shard in range(self._n_shards):
      sharded_files.extend(self.shard_files(shard, tmp=tmp))

    return sharded_files

  def orig_tmp_file_pairs(self):
    """Creates a list of tuples of (original_path, tmp_path).

    Returns:
      A list of tuples where first path is original and second is tmp location
    """
    pairs = []
    for shard in range(self._n_shards):
      orig_paths = self.shard_files(shard, tmp=False)
      tmp_paths = self.shard_files(shard, tmp=True)
      pairs.extend(list(zip(orig_paths, tmp_paths)))
    return pairs

  def shard_files(self, shard, tmp = False):
    """Return file paths that correspond to the given shard.

    Args:
      shard: Shard to make paths for
      tmp: Whether to return tmp version or not

    Returns:
      A list of files for the shard
    """
    return [
        self.shard_emb(shard, tmp=tmp),
        self.shard_urls(shard, tmp=tmp),
        self.shard_sentence_ids(shard, tmp=tmp),
    ]

  def _file_dir(self, tmp):
    if tmp:
      return self._tmp_dir
    else:
      return self._embedding_dir

  def shard_emb(self, shard, tmp = False):
    """Get the path for the sharded embedding.

    Args:
      shard: Shard to get embedding path for
      tmp: Whether to return /tmp version of path

    Returns:
      Path to embedding for given shard
    """
    return os.path.join(
        self._file_dir(tmp), f'embeddings_{shard}_{self._n_shards}.npy')

  def shard_urls(self, shard, tmp = False):
    """Get the path for the sharded urls.

    Args:
      shard: Shard to get urls path for
      tmp: Whether to return /tmp version of path

    Returns:
      Path to urls for given shard
    """
    return os.path.join(
        self._file_dir(tmp), f'urls_{shard}_{self._n_shards}.npy')

  def shard_sentence_ids(self, shard, tmp = False):
    """Get the path for the sharded sentence_ids.

    Args:
      shard: Shard to get sentence_ids path for
      tmp: Whether to return /tmp version of path

    Returns:
      Path to sentence_ids for given shard
    """
    return os.path.join(
        self._file_dir(tmp), f'sentence_ids_{shard}_{self._n_shards}.npy')

  def to_tmp(self):
    path_pairs = self.orig_tmp_file_pairs()
    with futures.ThreadPoolExecutor(max_workers=len(path_pairs)) as executor:
      list(tqdm.tqdm(executor.map(copy_file_to_tmp, path_pairs)))


def read_examples(
    *, embedding_dir, n_shards,
    copy_to_tmp):
  """Read and yield examples from embeddings in directory.

  Args:
    embedding_dir: The directory of .npy embedding files
    n_shards: Number of shards used to create the data
    copy_to_tmp: Whether to copy embeddings to /tmp before reading, this can
      significantly improve throughput compared to remote filesystem reads

  Yields:
    Tuples of an integer identifier, wikipedia_url, sentence_id, and embedding
  """
  idx = 0
  sharded_paths = ShardedEmbPaths(
      embedding_dir=embedding_dir, n_shards=n_shards)
  logging.info('Copying files to tmp')
  if copy_to_tmp:
    sharded_paths.to_tmp()
  logging.info('Starting example read')
  for shard in tqdm.trange(n_shards):
    emb_path = sharded_paths.shard_emb(shard, tmp=copy_to_tmp)
    urls_path = sharded_paths.shard_urls(shard, tmp=copy_to_tmp)
    sentence_ids_path = sharded_paths.shard_sentence_ids(shard, tmp=copy_to_tmp)
    logging.info('Emb path: %s', emb_path)
    logging.info('Urls path: %s', urls_path)
    logging.info('Sent path: %s', sentence_ids_path)

    with \
        util.safe_open(emb_path, 'rb') as emb_f,\
        util.safe_open(urls_path, 'rb') as url_f,\
        util.safe_open(sentence_ids_path, 'rb') as sid_f:
      load_start = time.time()
      embeddings = np.load(emb_f)
      wikipedia_urls = np.load(url_f)
      sentence_ids = np.load(sid_f)
      load_end = time.time()
      logging.info('Reading shard %s, Seconds: %s', shard,
                   load_end - load_start)
      for wiki_url, sid, emb in zip(wikipedia_urls, sentence_ids, embeddings):
        yield idx, wiki_url, sid, emb
        idx += 1


class Index:
  """Index that can be used for brute for neighbor search."""

  def __init__(self,
               *,
               claim_embedding_path,
               claim_id_path,
               wiki_embedding_dir,
               n_shards,
               copy_to_tmp,
               device = '/CPU:0',
               batch_size = 256,
               l2_norm=False):
    """Configure index.

    Claim ids and embeddings are related through their position (eg, first
    embedding corresponds to first id).

    Args:
      claim_embedding_path: Path to claim embeddings
      claim_id_path: Path to claim ids
      wiki_embedding_dir: Directory of .npy embedding files
      n_shards: Number of shards to read
      copy_to_tmp: Whether to copy embeddings to tmp (eg, maybe they are on slow
        network file system)
      device: Tensorflow device to use for batched matrix multiplies
      batch_size: Batch size for batched matrix multiplies
      l2_norm: Whether to impose the L2 norm on vectors
    """
    self._claim_embedding_path = claim_embedding_path
    self._claim_id_path = claim_id_path
    self._wiki_embedding_dir = wiki_embedding_dir
    self._n_shards = n_shards
    self._batch_size = batch_size
    self._l2_norm = l2_norm
    self._copy_to_tmp = copy_to_tmp
    self._device = device
    self._wiki_embeddings: Optional[tf.Tensor] = None
    self._key_to_idx: Optional[Dict[IndexKey, int]] = None
    self._idx_to_key: Optional[Dict[int, IndexKey]] = None
    self._claim_embeddings: Optional[tf.Tensor] = None
    self._claim_ids: Optional[tf.Tensor] = None
    self._claim_id_to_idx: Optional[Dict[int, int]] = None

  def build(self, load_wiki=True, load_fever=True):
    """Build the index in memory.

    Args:
      load_wiki: Whether to load wiki embeddings
      load_fever: Whether to load fever claim embeddings
    """
    if load_wiki:
      self._load_wiki_embeddings()
    if load_fever:
      self._load_fever_embeddings()

  def _load_fever_embeddings(self):
    """Load fever claim embeddings and ids."""
    with util.safe_open(self._claim_embedding_path, 'rb') as f:
      claim_embeddings = np.load(f)
    with util.safe_open(self._claim_id_path, 'rb') as f:
      claim_ids = np.load(f)
    self._claim_embeddings = tf.convert_to_tensor(claim_embeddings)
    if self._l2_norm:
      self._claim_embeddings = tf.math.l2_normalize(
          self._claim_embeddings, axis=-1)
    self._claim_ids = claim_ids
    self._claim_id_to_idx = {}
    for idx, claim_id in enumerate(self._claim_ids):
      self._claim_id_to_idx[claim_id] = idx

  def _load_wiki_embeddings(self):
    """Build an index from embeddings."""
    logging.info('Read: %s', self._wiki_embedding_dir)
    logging.info('N Shards: %s', self._n_shards)
    examples = read_examples(
        embedding_dir=self._wiki_embedding_dir,
        n_shards=self._n_shards,
        copy_to_tmp=self._copy_to_tmp,
    )
    logging.info('Starting indexing')
    embeddings = []
    self._key_to_idx = {}
    self._idx_to_key = {}
    for idx, wiki_url, sid, emb in tqdm.tqdm(examples, mininterval=10):
      embeddings.append(emb)
      self._key_to_idx[IndexKey(wiki_url, sid)] = idx
      self._idx_to_key[idx] = IndexKey(wiki_url, sid)
    self._wiki_embeddings = tf.convert_to_tensor(np.vstack(embeddings))
    if self._l2_norm:
      self._wiki_embeddings = tf.math.l2_normalize(
          self._wiki_embeddings, axis=-1)
    logging.info('Embedding Shape: %s', self._wiki_embeddings.shape)

  def score_claim_to_wiki(self, n = 5):
    """Score all claims to wikipedia and return the predictions.

    Args:
      n: Number of predictions to make per claim

    Returns:
      Top ranked wikipedia sentences per claim
    """
    logging.info('TF Initializing')
    if self._claim_embeddings is None:
      raise ValueError('Claim embeddings are not loaded')
    with tf.device(self._device):
      idx = 0
      top_idx = []
      top_scores = []
      bar = tqdm.tqdm(total=self._claim_embeddings.shape[0])
      # batch_size: over n_claims
      while idx < self._claim_embeddings.shape[0]:
        # (batch_size, emb_dim)
        batch = self._claim_embeddings[idx:idx + self._batch_size, :]
        # (n_wiki_embeddings, batch_size)
        batch_scores = tf.linalg.matmul(
            # wiki_embeddings: (n_wiki_embeddings, emb_dim)
            self._wiki_embeddings,
            batch,
            transpose_b=True)
        # <float>(batch_size, n_wiki_embeddings)
        batch_scores = tf.transpose(batch_scores)
        # <float>(batch_size, n), <long>(batch_size, n)
        batch_top_scores, batch_top_idx = tf.nn.top_k(batch_scores, k=n)
        top_idx.append(batch_top_idx.numpy())
        top_scores.append(batch_top_scores.numpy())
        idx += self._batch_size
        bar.update(self._batch_size)
      bar.close()
      top_idx = np.vstack(top_idx)
      top_scores = np.vstack(top_scores)

    claim_id_to_scored_keys = {}
    for claim_index in range(top_idx.shape[0]):
      row = top_idx[claim_index]
      wiki_keys = []
      for idx in row:
        wiki_keys.append(self._idx_to_key[idx])
      claim_id = self._claim_ids[claim_index]
      claim_id_to_scored_keys[claim_id] = {
          'wiki_keys': wiki_keys,
          'scores': top_scores[claim_index]
      }

    return claim_id_to_scored_keys


def main(_):
  flags.mark_flag_as_required('out_path')
  flags.mark_flag_as_required('wiki_embedding_dir')
  flags.mark_flag_as_required('claim_id_path')
  flags.mark_flag_as_required('claim_embedding_path')
  flags.mark_flag_as_required('n_shards')

  tf.enable_v2_behavior()

  conf = config.Config()
  logging.info('wiki_embedding_dir: %s', FLAGS.wiki_embedding_dir)
  logging.info('n_shards: %s', FLAGS.n_shards)
  logging.info('l2_norm: %s', FLAGS.l2_norm)
  logging.info('claim_id_path: %s', FLAGS.claim_id_path)
  logging.info('claim_embedding_path: %s', FLAGS.claim_embedding_path)
  logging.info('copy_to_tmp: %s', FLAGS.copy_to_tmp)
  logging.info('batch_size: %s', FLAGS.batch_size)

  with util.log_time('Building index'):
    index = Index(
        wiki_embedding_dir=FLAGS.wiki_embedding_dir,
        n_shards=FLAGS.n_shards,
        l2_norm=FLAGS.l2_norm,
        claim_id_path=FLAGS.claim_id_path,
        claim_embedding_path=FLAGS.claim_embedding_path,
        copy_to_tmp=FLAGS.copy_to_tmp,
        batch_size=FLAGS.batch_size,
        device=FLAGS.device,
    )
    index.build()

  logging.info('Reading claims from: %s', conf.fever_dev)
  dev = [
      c for c in util.read_jsonlines(conf.fever_dev)
      if c['label'] != constants.NOT_ENOUGH_INFO
  ]

  logging.info('Making predictions')
  claim_id_to_scored_keys = index.score_claim_to_wiki(n=5)

  formatted_predictions = []
  actual = []
  for claim in tqdm.tqdm(dev):
    claim_id = claim['id']
    predicted_evidence = []
    scored_keys = claim_id_to_scored_keys[claim_id]
    for index_key in scored_keys['wiki_keys']:
      # sentence_id is a numpy int, and fever scoring script only
      # accepts python int.
      predicted_evidence.append([
          index_key.wikipedia_url, int(index_key.sentence_id)])

    formatted_predictions.append({
        'id': claim_id,
        'predicted_label': constants.SUPPORTS,
        'predicted_evidence': predicted_evidence,
    })
    actual.append({'evidence': claim['evidence'], 'label': claim['label']})

  logging.info('FEVER Metrics')
  strict_score, accuracy_score, precision, recall, f1 = fever_score(
      formatted_predictions, actual)
  logging.info('Strict Score: %s', strict_score)
  logging.info('Accuracy Score: %s', accuracy_score)
  logging.info('Precision: %s', precision)
  logging.info('Recall: %s', recall)
  logging.info('F1: %s', f1)

  logging.info('Saving predictions and metrics to: %s', FLAGS.out_path)
  util.write_json(
      {
          'predictions': formatted_predictions,
          'metrics': {
              'strict_score': strict_score,
              'accuracy_score': accuracy_score,
              'precision': precision,
              'recall': recall,
              'f1': f1,
          }
      }, FLAGS.out_path)


if __name__ == '__main__':
  app.run(main)

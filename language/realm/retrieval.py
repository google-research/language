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
"""Utilities for performing retrieval."""
import abc
from concurrent import futures
import time


from absl import logging
from language.realm import featurization
from language.realm import parallel
from language.realm import profile
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


class Retriever(abc.ABC):
  """Retrieves documents for a query."""

  @abc.abstractmethod
  def retrieve(self, query_batch):
    """Retrieves candidates for a batch of queries.

    Args:
      query_batch (list[Query]): a list of queries.

    Returns:
      a batch of lists, where each list is a list of Documents for the
      corresponding query.
    """
    raise NotImplementedError()


class DummyRetriever(Retriever):
  """Dummy retriever for testing."""

  def __init__(self, num_neighbors):
    self._num_neighbors = num_neighbors
    self.total_candidates = 13353718
    self.embed_dim = 128
    with tf.device('/CPU:0'):
      self._doc_embeds = tf.zeros((self.total_candidates, self.embed_dim))

  def retrieve(self, query_batch):
    # [batch_size, embed_dim]
    query_embeds = tf.zeros((len(query_batch), self.embed_dim))

    with tf.device('/CPU:0'):
      # [batch_size, total_candidates]
      cand_scores = tf.matmul(query_embeds, self._doc_embeds, transpose_b=True)
      _, top_ids_batch = tf.math.top_k(cand_scores, k=self._num_neighbors)

    title_ids = np.zeros(10, dtype=np.int32)
    body_ids = np.zeros(280, dtype=np.int32)

    retrievals_batch = []
    for top_ids in top_ids_batch:
      retrievals = [
          featurization.Document(0, title_ids, body_ids) for i in top_ids
      ]
      retrievals_batch.append(retrievals)
    return retrievals_batch


class BruteForceRetriever(Retriever):
  """Retrieves documents using brute force matrix multiplication."""

  def __init__(self, query_embedder, documents, doc_embeds_or_path,
               num_neighbors):
    """Constructs BruteForceRetriever.

    Args:
      query_embedder: an instance of QueryEmbedder.
      documents: a list of Document objects.
      doc_embeds_or_path: either a [num_docs, embed_dim] TF Tensor, or a path to
        load it.
      num_neighbors: number of neighbors to retrieve.
    """
    total_candidates = len(documents)

    self._query_embedder = query_embedder
    self._num_neighbors = num_neighbors
    self._documents = documents

    # Load embeddings.
    if isinstance(doc_embeds_or_path, str):
      with tf.device('/CPU:0'):
        ckpt_reader = tf.train.load_checkpoint(doc_embeds_or_path)
        self._doc_embeds = ckpt_reader.get_tensor('block_emb')
    else:
      self._doc_embeds = doc_embeds_or_path
    logging.info('Loaded document embeddings.')

    # Check shapes.
    if self._doc_embeds.shape[0] != total_candidates:
      raise ValueError('Did not load the right number of embeddings.')

  @profile.profiled_function
  def retrieve(self, query_batch):
    # [batch_size, embed_dim]
    query_embeds = self._query_embedder.embed(query_batch)

    with tf.device('/CPU:0'):
      # [batch_size, total_candidates]
      cand_scores = tf.matmul(query_embeds, self._doc_embeds, transpose_b=True)
      _, top_ids_batch = tf.math.top_k(cand_scores, k=self._num_neighbors)

    retrievals_batch = []
    for top_ids in top_ids_batch:
      retrievals = [self._documents[i] for i in top_ids]
      retrievals_batch.append(retrievals)
    return retrievals_batch


def count_tf_records(file_path):
  """Counts the number of records in a GZIP'd TFRecord file."""
  gzip_option = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  count = 0
  for _ in tf.python_io.tf_record_iterator(file_path, gzip_option):
    count += 1
  return count


def count_tf_records_parallel_helper(args):
  """Just a helper function for count_tf_records_parallel."""
  file_idx, file_path = args
  return (file_idx, count_tf_records(file_path))


def count_tf_records_parallel(file_paths, num_processes=None):
  """Counts number of records in TFRecord files in parallel.

  Args:
    file_paths: a list of paths, where each path points to a GZIP-ed TFRecord
      file.
    num_processes: number of Python processes to use in parallel. If None, will
      use all available CPUs.

  Returns:
    shard_sizes: a list of ints.
  """
  num_files = len(file_paths)
  with parallel.Executor(
      create_worker=lambda: count_tf_records_parallel_helper,
      queue_size=num_files,
      num_workers=num_processes) as executor:
    for file_idx, file_path in enumerate(file_paths):
      executor.submit((file_idx, file_path))

    counts = [None] * num_files
    results = executor.results(max_to_yield=num_files)
    for i, (file_idx, count) in enumerate(results):
      counts[file_idx] = count
      logging.info('Counted %d / %d files.', i + 1, num_files)

  return counts


def load_documents(path):
  """Loads Documents from a GZIP-ed TFRecords file into a Python list."""
  gzip_option = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)

  def get_bytes_feature(ex, name):
    return list(ex.features.feature[name].bytes_list.value)

  def get_ints_feature(ex, name):
    # 32-bit Numpy arrays are more memory-efficient than Python lists.
    return np.array(ex.features.feature[name].int64_list.value, dtype=np.int32)

  docs = []
  for val in tf.python_io.tf_record_iterator(path, gzip_option):
    ex = tf.train.Example.FromString(val)
    title = get_bytes_feature(ex, 'title')[0]
    body = get_bytes_feature(ex, 'body')[0]

    doc_uid = featurization.get_document_uid(title, body)
    title_token_ids = get_ints_feature(ex, 'title_token_ids')
    body_token_ids = get_ints_feature(ex, 'body_token_ids')

    doc = featurization.Document(
        uid=doc_uid,
        title_token_ids=title_token_ids,
        body_token_ids=body_token_ids)
    docs.append(doc)

  return docs


def load_documents_from_shard(args):
  """A helper function for load_documents_from_shards."""
  shard_idx, shard_path = args
  docs = load_documents(shard_path)
  return (shard_idx, docs)


@profile.profiled_function
def load_documents_from_shards(shard_paths, num_processes=None):
  """Loads Documents from a sharded, GZIP-ed TFRecords file into a Python list.

  Uses multiple processes to perform IO in parallel.

  Args:
    shard_paths: a list of paths, where each path points to a GZIP-ed TFRecords
      file. Documents loaded from each shard will be concatenated in the order
      of shard_paths.
    num_processes: number of Python processes to use in parallel. If None, will
      use all available CPUs.

  Returns:
    a list of Document instances.
  """
  num_shards = len(shard_paths)

  with parallel.Executor(
      create_worker=lambda: load_documents_from_shard,
      queue_size=num_shards,
      num_workers=num_processes) as executor:
    for shard_idx, shard_path in enumerate(shard_paths):
      executor.submit((shard_idx, shard_path))

    results = []
    for shard_idx, docs in executor.results(max_to_yield=num_shards):
      results.append((shard_idx, docs))
      logging.info('Loaded %d of %d document shards.', len(results), num_shards)

    # Sorts results by shard_idx.
    results.sort()

    logging.info('Combining data from all document shards.')
    all_docs = []
    for shard_idx, docs in results:
      all_docs.extend(docs)

    logging.info('Finished loading all shards.')
    return all_docs


class QueryEmbedder(object):
  """Embeds queries."""

  def __init__(self, embedder_model_or_path, featurizer):
    if isinstance(embedder_model_or_path, str):
      # Assume it is a path to a SavedModel
      self._model = tf.saved_model.load_v2(embedder_model_or_path, tags={})
    else:
      # Assume it is an already loaded SavedModel
      self._model = embedder_model_or_path
    logging.info('Loaded query embedder.')

    self._featurizer = featurizer

  def embed(self, query_batch):
    """Embeds a batch of queries.

    Args:
      query_batch: a list of Query instances.

    Returns:
      embeds: a [batch_size, embed_dim] float Tensor.
    """
    with profile.Timer('embed_featurize'):
      feature_dicts = [self._featurizer.featurize_query(q) for q in query_batch]

      # Concatenate features into a single dict with the following structure:
      #   input_ids: [batch_size, seq_len] <int32>
      #   input_mask: [batch_size, seq_len] <int32>
      #   segment_ids: [batch_size, seq_len] <int32>
      model_inputs = featurization.batch_feature_dicts(feature_dicts)

    with profile.Timer('embed_tf'):
      return self._model.signatures['projected'](**model_inputs)['default']


class DocumentEmbedder(object):
  """Embeds documents using TF Estimator.

  Note: this only works with the REALM Hub modules. An ICT Hub module won't work
  because it has a different set of signatures.
  """

  def __init__(self, hub_module_spec, featurizer, use_tpu, run_config=None):
    """Constructs the DocumentEmbedder."""
    if run_config is None:
      if use_tpu:
        raise ValueError('Must supply a RunConfig if use_tpu.')
      else:
        run_config = tf.estimator.tpu.RunConfig()  # Just supply a default.

    self._hub_module_spec = hub_module_spec
    self._featurizer = featurizer
    self._use_tpu = use_tpu
    self._run_config = run_config
    self._log_interval = 10  # When embedding, log every 10 seconds.

  def embed(self, get_documents_dataset, total_docs, batch_size):
    """Embeds a Dataset of documents using Estimator.

    Args:
      get_documents_dataset: a function that returns a TF Dataset, where each
        element is a dict with attributes described below.
      total_docs: total number of documents in the Dataset.
      batch_size: number of documents to embed in each batch.  Each element in
        the Dataset returned by get_documents_dataset should be a dict with the
        attributes described below.

    get_documents_dataset should return a Dataset over dicts, each containing at
    least the following attributes:
    - title_token_ids: a 1-D int Tensor.
    - body_token_ids: a 1-D int Tensor.

    Yields:
      a [embed_dim] Numpy array, one for each document.
    """
    if total_docs < 1:
      raise ValueError('Must embed at least 1 document.')

    # These hyperparams are passed to Estimator.
    params = {
        'vocab_path':
            self._featurizer.tokenizer.vocab_path,
        'do_lower_case':
            self._featurizer.tokenizer.do_lower_case,
        'query_seq_len':
            self._featurizer.query_seq_len,
        'candidate_seq_len':
            self._featurizer.candidate_seq_len,
        'num_candidates':
            self._featurizer.num_candidates,
        'max_masks':
            self._featurizer.max_masks,
        'separate_candidate_segments':
            self._featurizer.separate_candidate_segments,
    }

    def input_fn(params):
      """Constructs the dataset fed to Estimator."""
      # We cannot access self._featurizer via closure, because this function is
      # passed to another device. Hence, we need to reconstruct the featurizer
      # from its hyerparameters (passed through `params`).

      tokenizer = featurization.Tokenizer(
          vocab_path=params['vocab_path'],
          do_lower_case=params['do_lower_case'])

      featurizer = featurization.Featurizer(
          query_seq_len=params['query_seq_len'],
          candidate_seq_len=params['candidate_seq_len'],
          num_candidates=params['num_candidates'],
          max_masks=params['max_masks'],
          tokenizer=tokenizer,
          separate_candidate_segments=params['separate_candidate_segments'])

      dataset = get_documents_dataset()

      def featurize(doc_dict):
        return featurizer.featurize_document_tf(doc_dict['title_token_ids'],
                                                doc_dict['body_token_ids'])

      dataset = dataset.map(
          featurize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Add a document index variable.
      dataset = dataset.enumerate()

      def _enumerate_to_dict(result_idx, tensor_dict):
        return dict(tensor_dict, result_idx=result_idx)

      dataset = dataset.map(
          _enumerate_to_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Pad the end of the dataset with one full extra batch.
      # This ensures that we don't drop the remainder.
      if total_docs % batch_size != 0:
        # Pad using the first value of the dataset, repeated batch_size times.
        pad_vals = dataset.take(1).repeat(batch_size)
        dataset = dataset.concatenate(pad_vals)

      # Batch the dataset.
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.prefetch(2)  # Prefetch for efficiency.
      return dataset

    def model_fn(features, labels, mode, params):
      """Constructs the model used by Estimator."""
      del labels, params
      embedder_module = hub.Module(
          spec=self._hub_module_spec, name='embedder', trainable=False)

      # Remove the result_idx before feeding features to the module.
      result_idx = features.pop('result_idx')

      # [batch_size, embed_dim]
      embeds = embedder_module(inputs=features, signature='projected')

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions={
              'embeds': embeds,
              'result_idx': result_idx
          })

    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=self._use_tpu,
        model_fn=model_fn,
        model_dir=None,  # Don't persist model.
        config=self._run_config,
        params=params,
        train_batch_size=batch_size,
        predict_batch_size=batch_size)

    logging.info('Embedding %d documents total.', total_docs)
    predictions = estimator.predict(
        input_fn=input_fn, yield_single_examples=True)
    for result in yield_predictions_from_estimator(
        predictions, total=total_docs, log_interval=self._log_interval):
      yield result['embeds']


def yield_predictions_from_estimator(predictions, total, log_interval=10):
  """Yields predictions from Estimator.predict, with added error correction.

  This function handles the case of Estimator.predict occasionally restarting,
  causing results to be yielded out of order.

  Args:
    predictions: the return value of Estimator.predict. An iterable of dicts.
      Each dict MUST have a 'result_idx' attribute, used to track result order.
    total (int): total expected number of elements to yield from predictions.
    log_interval: log every this many seconds.

  Yields:
    the same dicts yielded from Estimator.predict, but in the right order. The
    result_idx element is removed from every dict.
  """
  predictions_iter = iter(predictions)
  total_yielded = 0

  start_time = time.time()
  last_log_timestamp = time.time()

  while total_yielded < total:
    try:
      result = next(predictions_iter)
    except StopIteration:
      raise ValueError(
          'Estimator.predict terminated before we got all results.')

    result_idx = result.pop('result_idx')

    if result_idx == total_yielded:
      # If results are always emitted from Estimator.predict in the same
      # order that they were fed into the Estimator, then we should always
      # expect result_idx to equal total_yielded. However, this does not always
      # happen, so we handle that in the `else` case below.
      yield result
      total_yielded += 1

      # Log progress.
      current_time = time.time()
      if current_time - last_log_timestamp > log_interval:
        total_time = current_time - start_time
        log_msg = 'Yielded {} results in {:.2f} secs.'.format(
            total_yielded, total_time)
        logging.info(log_msg)
        last_log_timestamp = current_time
    else:
      # If results start to arrive out of order, something has gone wrong.

      if result_idx < total_yielded:
        # This can happen if the TPU worker dies, causing Estimator.predict to
        # restart from the beginning. In this case, we just don't yield
        # anything on this step. Instead, we keep pulling things from the
        # iterator until we are back to where we were.
        if result_idx == 0:
          logging.warning('TPU worker seems to have restarted.')
      elif result_idx > total_yielded:
        # Something has gone really wrong.
        raise ValueError('Estimator.predict has somehow missed a result.')


def embed_documents_using_multiple_tpu_workers(
    shard_paths, shard_sizes, hub_module_spec,
    featurizer, tpu_workers,
    batch_size, num_tpu_cores_per_worker):
  """Embeds documents using multiple TPU workers.

  Args:
    shard_paths: a list of file paths, each specifying a GZIP'd TFRecord file
      containing documents stored as TF Examples. Doc embeddings will be
      concatenated in the order of shard_paths.
    shard_sizes: a list parallel to shard_paths, specifying the number of
      documents in each shard.
    hub_module_spec: path to the Hub module that will be used to embed the
      documents.
    featurizer: a Featurizer used to convert documents into Tensor features.
    tpu_workers: list of addresses of available TPU workers.
    batch_size: each TPU worker embeds documents in batches of this size.
    num_tpu_cores_per_worker: number of cores to use on each TPU worker.

  Returns:
    a [total_docs, embed_dim] Numpy array.
  """
  num_shards = len(shard_paths)
  num_tpu_workers = len(tpu_workers)

  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=1,  # This seems to be ignored by predict().
      num_shards=num_tpu_cores_per_worker)

  # Distribute the data shards as evenly as possible among the workers.
  num_shards_per_worker = [num_shards // num_tpu_workers] * num_tpu_workers
  for worker_idx in range(num_shards % num_tpu_workers):
    num_shards_per_worker[worker_idx] += 1

  worker_kwargs = []
  shards_assigned = 0
  for k, num_shards_k in enumerate(num_shards_per_worker):
    worker_kwargs.append({
        'tpu_run_config':
            tf.estimator.tpu.RunConfig(
                master=tpu_workers[k], tpu_config=tpu_config),
        'shard_paths':
            shard_paths[shards_assigned:shards_assigned + num_shards_k],
        'shard_sizes':
            shard_sizes[shards_assigned:shards_assigned + num_shards_k],
        'hub_module_spec': hub_module_spec,
        'featurizer': featurizer,
        'batch_size': batch_size,
    })
    shards_assigned += num_shards_k

  # All shards should be assigned.
  assert shards_assigned == num_shards

  # Run all workers in parallel via separate threads.
  with futures.ThreadPoolExecutor(max_workers=num_tpu_workers) as executor:
    # A list of [num_docs_per_worker, embed_dim] Numpy arrays.
    embeds_list = list(
        executor.map(lambda kwargs: embed_documents(**kwargs), worker_kwargs))

    # A [total_docs, embed_dim] Numpy array.
    embeds = np.concatenate(embeds_list, axis=0)

  return embeds


def embed_documents(
    shard_paths,
    shard_sizes,
    hub_module_spec,
    featurizer,
    batch_size,
    tpu_run_config = None):
  """Embeds documents either locally (CPU/GPU) or with a TPU worker.

  Note: TPUEstimator.predict currently requires the TPU worker to have a single
  "host" (a machine running TensorFlow that is physically connected to the TPU
  chips). This is not true for all TPU topologies -- some have multiple hosts.

  Args:
    shard_paths: a list of file paths, each specifying a GZIP'd TFRecord file
      containing documents stored as TF Examples. Doc embeddings will be
      concatenated in the order of shard_paths.
    shard_sizes: a list parallel to shard_paths, specifying the number of
      documents in each shard.
    hub_module_spec: path to the Hub module that will be used to embed the
      documents.
    featurizer: a Featurizer used to convert documents into Tensor features.
    batch_size: embed documents in batches of this size.
    tpu_run_config: configures the TPU worker. If None, run on CPU/GPU.

  Returns:
    a [total_docs, embed_dim] Numpy array.
  """
  embedder = DocumentEmbedder(
      hub_module_spec=hub_module_spec,
      featurizer=featurizer,
      use_tpu=(tpu_run_config is not None),
      run_config=tpu_run_config)

  def parse_tf_example(serialized):
    # FixedLenSequenceFeature requires allow_missing to be True, even though we
    # can't actually handle those cases.
    feature_spec = {
        'title':
            tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'text':
            tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'title_token_ids':
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'body_token_ids':
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    features = tf.parse_single_example(serialized, feature_spec)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(features.keys()):
      tensor = features[name]
      if tensor.dtype == tf.int64:
        tensor = tf.cast(tensor, tf.int32)
      features[name] = tensor

    return features

  def get_documents_dataset():
    # Note: num_parallel_reads should be None to guarantee that shard_paths
    # are visited sequentially, not in parallel.
    dataset = tf.data.TFRecordDataset(
        shard_paths,
        compression_type='GZIP',
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=None)

    return dataset.map(
        parse_tf_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  embeds = embedder.embed(
      get_documents_dataset=get_documents_dataset,
      total_docs=sum(shard_sizes),
      batch_size=batch_size)
  # A list of [embed_dim] Numpy arrays.
  embeds_list = list(embeds)

  # A [total_docs, embed_dim] Numpy array.
  return np.stack(embeds_list, axis=0)

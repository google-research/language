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
r"""Generates examples for REALM.

To load the entire retrieval index and relevant models, you will need a machine
with at least 50 GB of RAM.

Repeatedly performs the following steps:
- Randomly choose a sentence to be the query.
- Randomly select a salient span in the sentence to mask out.
- Retrieve a set of candidates for the query (using MIPS).
- Perform some additional processing on the candidates (e.g. add a null cand).
- Form a TF Example from the query and candidates.
- Push the TF Example to a queue.

"""
import collections
import os
import random
import re
import time

from absl import app
from absl import flags
from absl import logging
from language.common.utils import experiment_utils
from language.common.utils import export_utils
from language.realm import featurization
from language.realm import preprocessing
from language.realm import profile
from language.realm import retrieval
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

# ==============================================================================
# Model configuration
# ==============================================================================

flags.DEFINE_string('initial_embedder_module', None,
                    'Path to an initial embedder module.')

flags.DEFINE_integer('query_seq_len', None,
                     'Maximum sequence length of the query text.')

flags.DEFINE_integer(
    'candidate_seq_len', None,
    'Maximum sequence length of the retrieved candidate text.')

flags.DEFINE_integer('max_masks', None,
                     'Maximum number of tokens that can be masked out.')

flags.DEFINE_string('vocab_path', None, 'Path to vocabulary file.')

flags.DEFINE_boolean('do_lower_case', True, 'Whether to lowercase text.')

flags.DEFINE_integer('retrieval_batch_size', 64,
                     'Retrieval is performed in batches of this size.')

flags.DEFINE_boolean('share_embedders', True,
                     'Whether we use the same embedders for queries and docs')

flags.DEFINE_boolean('separate_candidate_segments', True,
                     'Whether titles and bodies have separate segment IDs.')

# ==============================================================================
# Data configuration
# ==============================================================================
flags.DEFINE_string('pretrain_corpus_path', None,
                    'Glob path to the pre-training corpus.')

flags.DEFINE_string('retrieval_corpus_path', None,
                    'Glob path to the sharded retrieval corpus.')

flags.DEFINE_integer('num_candidates', None,
                     'Number of candidates to retrieve per example.')

flags.DEFINE_boolean('is_train', True, 'Use the training or heldout split.')

flags.DEFINE_integer(
    'num_shards_per_mips_refresh', 4,
    'Each time the MIPS index is refreshed, randomly select this number of '
    'shards and generate examples from those shards. Don\'t refresh again '
    'until we have exhausted those shards.')

# ==============================================================================
# Server configuration
# ==============================================================================

flags.DEFINE_bool(
    'local_debug', False,
    'In local debug mode, examples are printed to stdout, rather'
    'than being pushed to the example queue.')

flags.DEFINE_integer('port', 8080, 'Example queue listens on this port.')

flags.DEFINE_integer(
    'max_queue_size', 1000,
    'Blocks until the queue has less than `max_queue_size` files.')

# Tracks various example generation statistics.
STATS = collections.Counter()


def generate_queries(featurizer):
  """Returns a generator over Query objects."""
  # Load the pre-training corpus.
  all_shard_paths = sorted(tf.gfile.Glob(FLAGS.pretrain_corpus_path))
  num_eval_shards = 5  # Hold out the first K for eval.
  eval_shard_paths = all_shard_paths[:num_eval_shards]
  train_shard_paths = all_shard_paths[num_eval_shards:]

  if FLAGS.is_train:
    pretrain_dataset = load_train_dataset(train_shard_paths,
                                          FLAGS.num_shards_per_mips_refresh)
  else:
    pretrain_dataset = load_eval_dataset(eval_shard_paths)

  def deserialize_example(tensor):
    return tf.train.Example.FromString(tensor.numpy())

  pretrain_dataset = map(deserialize_example, pretrain_dataset)

  # An iterable of Query objects.
  queries = (text_features_to_query(x, featurizer) for x in pretrain_dataset)
  queries = filter(lambda q: q is not None, queries)
  return queries


def generate_queries_and_candidates(featurizer, retriever):
  """Returns a generator over (Query, List[Document]) pairs."""
  queries = generate_queries(featurizer)

  for query_batch in batch(queries, FLAGS.retrieval_batch_size):
    # candidates_batch has shape [batch_size, num_candidates]
    # Each element is a Document.
    candidates_batch = retriever.retrieve(query_batch)

    for query, cands in zip(query_batch, candidates_batch):
      # If the retrieval index contains duplicate docs, we may end up retrieving
      # duplicates. Skip such examples.
      cand_uids = [c.uid for c in cands]
      if len(cand_uids) != len(set(cand_uids)):
        STATS['duplicate_candidates'] += 1
        continue

      yield (query, postprocess_candidates(cands, query))


def generate_realm_examples(featurizer, retriever, model_timestamp):
  """Generates examples for REALM pre-training.

  Args:
    featurizer: instance of featurization.Featurizer
    retriever: instance of retrieval.Retriever
    model_timestamp (int): integer tracking the time when model was saved.

  Yields:
    query: a Query object.
    docs: a list of Documents. These are the candidates retrieved for the query.
    example: a TF Example holding the featurized representation of the query
      and the docs.
  """
  query_cand_pairs = generate_queries_and_candidates(featurizer, retriever)
  for i, (query, cands) in enumerate(query_cand_pairs):
    ex = featurizer.query_and_docs_to_tf_example(query, cands, model_timestamp)
    yield query, cands, ex

    if i % 100 == 0:
      logging.info('Example generation stats @ %d: %s', i, str(STATS))


def batch(iterable, batch_size):
  """Groups examples into batches, dropping remainder."""
  if batch_size < 1:
    raise ValueError('batch_size must be a positive integer.')
  item_batch = []
  for item in iterable:
    item_batch.append(item)
    if len(item_batch) == batch_size:
      yield item_batch
      item_batch = []


@profile.profiled_function
def text_features_to_query(ex, featurizer):
  """Converts a dict of text features to a Query.

  Args:
    ex: a TF Example containing the features described below.
    featurizer: an instance of featurization.Featurizer

  Returns:
    a Query

  Each Example has the following features:
  - title: title of the document (just a bytes string).
  - text: raw text of the document (just a bytes string).
  - sentence_byte_start: byte offset for the start of each sentence (inclusive).
  - sentence_byte_limit: byte offset for the end of each sentence (exclusive).
  - span_byte_start: byte offset for the start of each salient span (inclusive).
  - span_byte_limit: byte offset for the end of each salient span (exclusive).
  """
  title = get_bytes_feature(ex, 'title')[0]
  body_text = get_bytes_feature(ex, 'text')[0]
  sentence_starts = get_ints_feature(ex, 'sentence_byte_start')
  sentence_limits = get_ints_feature(ex, 'sentence_byte_limit')
  span_starts = get_ints_feature(ex, 'span_byte_start')
  span_limits = get_ints_feature(ex, 'span_byte_limit')

  # List of (start, stop) byte offsets for each sentence (right-exclusive).
  sentence_boundaries = list(zip(sentence_starts, sentence_limits))

  # List of (start, stop) byte offsets for each salient span (right-exclusive).
  spans = list(zip(span_starts, span_limits))

  # Map spans to sentences.
  # Spans that do not strictly fall within a single sentence are omitted.
  span_to_sentence_boundaries = {}
  for span_start, span_stop in spans:
    for sent_start, sent_stop in sentence_boundaries:
      if span_start >= sent_start and span_stop <= sent_stop:
        span_to_sentence_boundaries[(span_start, span_stop)] = (sent_start,
                                                                sent_stop)
        break

  if not span_to_sentence_boundaries:
    # If there are no valid spans, skip this example.
    STATS['no_valid_spans'] += 1
    return None

  # Randomly sample a span.
  selected_span, selected_sentence_boundaries = random.choice(
      list(span_to_sentence_boundaries.items()))

  # Shift the span offsets to be relative to the sentence.
  selected_span = [
      offset - selected_sentence_boundaries[0] for offset in selected_span
  ]

  # Extract the sentence from the passage.
  sentence_text = body_text[
      selected_sentence_boundaries[0]:selected_sentence_boundaries[1]]

  try:
    sentence_tokens = featurizer.tokenizer.tokenize(sentence_text)
  except featurization.TokenizationError:
    # Tokenization errors can occur if we are unable to recover the byte offset
    # of a token in the original string. If so, skip this query.
    STATS['tokenization_error'] += 1
    return None

  doc_uid = featurization.get_document_uid(title, body_text)

  query = featurization.Query(
      text=sentence_text,
      tokens=sentence_tokens,
      mask_spans=[selected_span],
      orig_doc_uid=doc_uid)

  try:
    featurizer.mask_query(query)
  except featurization.MaskingError:
    # If the masks cannot be appropriately applied, skip this query.
    STATS['masking_error'] += 1
    return None

  return query


def get_bytes_feature(ex, name):
  return list(ex.features.feature[name].bytes_list.value)


def get_ints_feature(ex, name):
  return list(ex.features.feature[name].int64_list.value)


NULL_DOCUMENT_UID = featurization.get_document_uid(b'', b'')
NULL_DOCUMENT = featurization.Document(NULL_DOCUMENT_UID, [], [])


def postprocess_candidates(candidates, query):
  """Perform additional processing on the candidates retrieved for a query.

  Args:
    candidates (list[Document]): a list of retrieved documents.
    query (Query): the query used to retrieve the documents.

  Returns:
    new_candidates (list[Document]): a list of the same size as candidates.
  """
  # If the query's originating document appears among candidates, remove it.
  candidates = [c for c in candidates if c.uid != query.orig_doc_uid]

  # We shouldn't have lost more than 1 candidate.
  assert len(candidates) >= FLAGS.num_candidates - 1

  # Prepend a null candidate.
  candidates.insert(0, NULL_DOCUMENT)
  candidates = candidates[:FLAGS.num_candidates]
  return candidates


def load_train_dataset(shard_paths, num_shards_to_sample):
  # Sample without replacement.
  shard_paths = list(shard_paths)
  random.shuffle(shard_paths)
  sample_shard_paths = shard_paths[:num_shards_to_sample]
  return load_dataset(sample_shard_paths, randomize=True)


def load_eval_dataset(shard_paths):
  return load_dataset(shard_paths, randomize=False)


def load_dataset(shard_paths, randomize):
  """Loads a dataset from a set of sharded TFRecord files."""
  tf.logging.info('Reading data from these shards:')
  for shard_path in shard_paths:
    tf.logging.info(shard_path)

  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(shard_paths))

  def tfrecord_dataset(path):
    buffer_size = 16 * 1024 * 1024  # Max number of bytes to store.
    return tf.data.TFRecordDataset(
        path, compression_type='GZIP', buffer_size=buffer_size)

  # `sloppy` mode means that the interleaving is not exact. This adds more
  # randomness to the training pipeline.
  dataset = dataset.apply(
      tf.data.experimental.parallel_interleave(
          map_func=tfrecord_dataset,
          sloppy=randomize,
          cycle_length=len(shard_paths)))

  if randomize:
    dataset = dataset.shuffle(buffer_size=10000)

  dataset = dataset.prefetch(16 * 1024)
  return dataset


@profile.profiled_function
def load_featurizer():
  """Loads a Featurizer."""
  tokenizer = featurization.Tokenizer(
      vocab_path=FLAGS.vocab_path, do_lower_case=FLAGS.do_lower_case)

  featurizer = featurization.Featurizer(
      query_seq_len=FLAGS.query_seq_len,
      candidate_seq_len=FLAGS.candidate_seq_len,
      num_candidates=FLAGS.num_candidates,
      max_masks=FLAGS.max_masks,
      tokenizer=tokenizer,
      separate_candidate_segments=FLAGS.separate_candidate_segments)

  logging.info('Loaded featurizer.')
  return featurizer


@profile.profiled_function
def load_retriever(query_embedder_path, docs, doc_embeds_path, featurizer):
  """Constructs a Retriever based on the specified embedder."""
  query_embedder = retrieval.QueryEmbedder(
      embedder_model_or_path=query_embedder_path, featurizer=featurizer)
  retriever = retrieval.BruteForceRetriever(
      query_embedder=query_embedder,
      documents=docs,
      doc_embeds_or_path=doc_embeds_path,
      num_neighbors=FLAGS.num_candidates)
  logging.info(
      'Loaded retriever with query embedder from %s and doc '
      'embeddings from %s', query_embedder_path, doc_embeds_path)
  return retriever


def load_latest_retriever(docs, featurizer):
  """Returns the latest Retriever and a model timestamp."""
  latest_embedder_path = export_utils.tfhub_export_path(
      model_dir=experiment_utils.FLAGS.model_dir,
      hub_prefix='encoded',
      module_prefix='embedder')
  if FLAGS.share_embedders:
    latest_query_embedder_path = latest_embedder_path
  else:
    latest_query_embedder_path = export_utils.tfhub_export_path(
        model_dir=experiment_utils.FLAGS.model_dir,
        hub_prefix='encoded',
        module_prefix='query_embedder')

  if latest_embedder_path is None:
    # The initial embedder module comes from the ICT codebase
    latest_embedder_path = FLAGS.initial_embedder_module
    latest_query_embedder_path = FLAGS.initial_embedder_module
    model_timestamp = 0
  else:
    model_timestamp_match = re.match('.+/export/.+/([0-9]+)/.+',
                                     latest_embedder_path)
    assert model_timestamp_match
    model_timestamp = int(model_timestamp_match.group(1))

  doc_embeds_path = os.path.join(latest_embedder_path, 'encoded/encoded.ckpt')

  retriever = load_retriever(
      query_embedder_path=latest_query_embedder_path,
      docs=docs,
      doc_embeds_path=doc_embeds_path,
      featurizer=featurizer)

  return (retriever, model_timestamp)


def generate_realm_examples_with_model_refresh():
  """Generates REALM examples, checking for new models periodically."""
  featurizer = load_featurizer()

  # Load a list of Documents (candidates for retrieval).
  doc_shard_paths = tf.io.matching_files(FLAGS.retrieval_corpus_path).numpy()
  doc_shard_paths = sorted([p.decode() for p in doc_shard_paths])
  docs = retrieval.load_documents_from_shards(doc_shard_paths, num_processes=12)

  while True:
    retriever, model_timestamp = load_latest_retriever(docs, featurizer)
    examples = generate_realm_examples(featurizer, retriever, model_timestamp)
    for ex in examples:
      yield ex


def main(unused_argv):
  tf.enable_eager_execution()
  # WARNING: do not set tf.debugging.set_log_device_placement(True)
  # This can cause an error when using Eager Execution and SavedModels.

  examples = generate_realm_examples_with_model_refresh()

  if FLAGS.local_debug:
    featurizer = load_featurizer()

    profile.reset()
    start_time = time.time()
    for i, (query, cands, _) in enumerate(examples):
      if i % 50 == 0:
        print('Example', i)
        print('Total time elapsed: {}'.format(time.time() - start_time))
        print(query)
        print('Originating doc UID: {}'.format(query.orig_doc_uid))
        for cand in cands:
          print('Doc UID: {}'.format(cand.uid))
          print(featurizer.tokenizer.token_ids_to_str(cand.title_token_ids))
          print(featurizer.tokenizer.token_ids_to_str(cand.body_token_ids))
        print()
      if i == 1000:
        break
    profile.print_report()
  else:
    preprocessing.push_examples(
        example_generator=(tf_ex for query, cands, tf_ex in examples),
        port=FLAGS.port,
        max_queue_size=FLAGS.max_queue_size,
        queue_timeout=30.0)


# Note: internal version of the code overrides this function.
def run_main():
  app.run(main)



if __name__ == '__main__':
  run_main()

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
r"""Script to generate a retrieval corpus.

On a machine with 12 CPU cores, this processes Wikipedia in 3 hours.
It needs roughly 10 GB of RAM.

"""
import json
import math
import time

from absl import app
from absl import flags

from language.realm import featurization
from language.realm import parallel
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'Path to documents in JSONL format.')

flags.DEFINE_string(
    'output_prefix', None,
    'Sharded output files will have the following format: '
    '{output_prefix}-{shard_idx}-of-{num_output_shards}')

flags.DEFINE_integer('num_output_shards', 50,
                     'Shard the output file into this many shards.')

flags.DEFINE_integer('total_documents', None, 'Total number of documents.')

flags.DEFINE_string('vocab_path', None, 'Path to vocabulary file.')

flags.DEFINE_boolean('do_lower_case', True, 'Whether to lowercase text.')

flags.DEFINE_boolean('parallel', True,
                     'Whether to process docs in parallel (multiprocessing).')


class DocumentProcessor(object):
  """Formats a document as a TF Example."""

  def __init__(self, vocab_path, do_lower_case):
    self._tokenizer = featurization.Tokenizer(
        vocab_path=vocab_path, do_lower_case=do_lower_case)

  def __call__(self, input_data):
    doc_idx, json_serialized = input_data
    doc_dict = json.loads(json_serialized)
    title = doc_dict['title']
    body = doc_dict['body']

    # Tokenize. Don't bother computing token boundaries, because they are
    # not used, and the process is not perfect anyway.
    title_tokens = self._tokenizer.tokenize(
        title, compute_token_boundaries=False)
    body_tokens = self._tokenizer.tokenize(body, compute_token_boundaries=False)

    features_dict = {}
    features_dict['title'] = bytes_feature([title.encode()])
    features_dict['body'] = bytes_feature([body.encode()])
    features_dict.update(tokens_to_features_dict(title_tokens, 'title'))
    features_dict.update(tokens_to_features_dict(body_tokens, 'body'))

    features = tf.train.Features(feature=features_dict)
    example = tf.train.Example(features=features)
    return doc_idx, example.SerializeToString()


def tokens_to_features_dict(tokens, name_prefix):
  """Converts a list of Tokens into a dict of TF Features."""
  as_tuples = [(t.text.encode(), t.start, t.stop, t.id) for t in tokens]
  features_dict = {}
  for i, (name_suffix, feature_type) in enumerate([('texts', bytes_feature),
                                                   ('starts', int_feature),
                                                   ('stops', int_feature),
                                                   ('ids', int_feature)]):
    feature_name = '{}_token_{}'.format(name_prefix, name_suffix)
    features_dict[feature_name] = feature_type([x[i] for x in as_tuples])
  return features_dict


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def int_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def load_json_data(input_path):
  with tf.gfile.Open(input_path) as input_file:
    for doc_idx, json_serialized in enumerate(input_file):
      yield doc_idx, json_serialized
      if doc_idx >= FLAGS.total_documents:
        raise ValueError('Got more documents than expected.')


def generate_examples():
  """Generates serialized TF Examples."""
  doc_processor = DocumentProcessor(FLAGS.vocab_path, FLAGS.do_lower_case)
  for doc_idx, json_serialized in load_json_data(FLAGS.input_path):
    yield doc_processor((doc_idx, json_serialized))
    if doc_idx >= FLAGS.total_documents:
      raise ValueError('Got more documents than expected.')


def generate_examples_parallel():
  """Same as `generate_examples`, but runs in parallel."""
  create_worker = DocumentProcessor
  worker_kwargs = {
      'vocab_path': FLAGS.vocab_path,
      'do_lower_case': FLAGS.do_lower_case,
  }

  with parallel.Executor(
      create_worker=create_worker,
      queue_size=10000,
      worker_kwargs=worker_kwargs) as executor:

    # Feed inputs to the DocumentProcessors.
    executor.submit_from_generator(load_json_data, input_path=FLAGS.input_path)

    for i, result in enumerate(executor.results()):
      yield result
      if i == FLAGS.total_documents - 1:
        break


def main(unused_argv):
  if FLAGS.parallel:
    results = generate_examples_parallel()
  else:
    results = generate_examples()

  start_time = time.time()
  docs_processed = 0

  examples = [None] * FLAGS.total_documents
  for doc_idx, example in results:
    examples[doc_idx] = example
    docs_processed += 1
    if docs_processed % 10000 == 0:
      print('Processed {} documents.'.format(docs_processed))
      print('Seconds elapsed:', time.time() - start_time)

  # Write to disk.
  gzip_option = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)

  num_docs_per_shard = math.ceil(FLAGS.total_documents /
                                 FLAGS.num_output_shards)

  for shard_idx in range(FLAGS.num_output_shards):
    # Shard path will look like '{output_prefix}-00012-of-00050'.
    shard_path = '{}-{}-of-{}'.format(FLAGS.output_prefix,
                                      str(shard_idx).zfill(5),
                                      str(FLAGS.num_output_shards).zfill(5))

    # Write to shard
    with tf.io.TFRecordWriter(shard_path, gzip_option) as output_writer:
      start_idx = num_docs_per_shard * shard_idx
      stop_idx = min(start_idx + num_docs_per_shard, FLAGS.total_documents)
      for doc_idx in range(start_idx, stop_idx):
        output_writer.write(examples[doc_idx])

    print('Finished writing: {}'.format(shard_path))


# Note: internal version of the code overrides this function.
def run_main():
  app.run(main)



if __name__ == '__main__':
  run_main()

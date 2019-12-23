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
"""Defines some common utilties for input pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from language.xsp.model import constants
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.gfile as gfile


def get_source_len_fn(key):

  def get_source_len(keys_to_tensors):
    shape = tf.shape(keys_to_tensors[key])
    return shape[0]

  return get_source_len


def get_data_files(data_sources):
  """Get list of data files from data_sources.

  Args:
    data_sources: a list or tuple of data file paths.

  Returns:
    A list of file paths.

  Raises:
    ValueError: if not data files are not found
  """
  # TODO(alanesuhr): Verify this is necessary for sharded TFRecord files?
  data_files = []
  for source in data_sources:
    if source.endswith('@*'):
      data_files += gfile.Glob(source[:-2] + '*')
    elif '@' in source:
      data_files += gfile.GenerateShardedFilenames(source)
    elif '*' in source or '?' in source or '[' in source:
      data_files += gfile.Glob(source)
    else:
      data_files.append(source)
  if not data_files:
    raise ValueError('No data files found in %s' % data_sources)
  return data_files


# Explicitly pad integers with special padding symbol.
def get_padded_value(datatype):
  if datatype == tf.string:
    return tf.constant('', dtype=tf.string)
  elif datatype == tf.int64:
    return tf.constant(constants.PAD_SYMBOL_ID, dtype=tf.int64)
  else:
    return tf.constant(0, dtype=datatype)


def get_dataset(data_sources, num_epochs, batch_size, decode_record,
                static_padded_shapes, shuffle, drop_remainder):
  """Converts a dataset file to a dataset tensor."""
  # Get sharded files from sstable path.
  tf.logging.info('Reading from ' + str(data_sources))

  data_files = get_data_files(
      [source_name + '@*' for source_name in data_sources])
  tf.logging.info('First file: ' + data_files[0])

  num_of_files = len(data_files)
  tf.logging.info('# of files: ' + str(num_of_files))

  num_reader_threads = min(num_of_files, multiprocessing.cpu_count())

  ds_filenames = tf.data.Dataset.from_tensor_slices(data_files)

  if num_epochs is None:
    ds_filenames = ds_filenames.shuffle(buffer_size=num_of_files)
  dataset = ds_filenames.apply(
      tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset,
          cycle_length=num_reader_threads,
          sloppy=num_epochs is None))

  capacity = 16 * batch_size
  dataset = dataset.map(
      decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if static_padded_shapes is None:
    # Let all shapes be dynamic.
    padded_shapes = dict([(name, [None] * len(shape))
                          for name, shape in dataset.output_shapes.items()])
  else:
    padded_shapes = static_padded_shapes

  if shuffle:
    dataset = dataset.shuffle(capacity)

  # Loop inifinitely if training, just once otherwise.
  dataset = dataset.repeat(num_epochs)

  padded_values = dict([(name, get_padded_value(datatype))
                        for name, datatype in dataset.output_types.items()])

  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      padding_values=padded_values,
      drop_remainder=drop_remainder)

  return dataset


def decode_features_and_labels(sequence_decoder,
                               feature_keys,
                               label_keys,
                               data_sources,
                               batch_size,
                               static_padded_shapes=None,
                               drop_remainder=False,
                               shuffle=False,
                               num_epochs=None,
                               scope='input_fn'):
  """Creates a dataset tensor out of a dataset given placeholders."""
  with tf.variable_scope(scope):

    def decode_record(record):
      """(key, serialized SequenceExample) to dict of <feature name, Tensor>."""
      decode_items = feature_keys + label_keys
      decoded = sequence_decoder.decode(record, items=decode_items)
      return dict(zip(decode_items, decoded))

    tf.logging.info('Reading from sources: ' + str(data_sources))

    # Load the datasets, and construct batches alternating between them.
    datasets = [
        get_dataset([source], num_epochs, batch_size, decode_record,
                    static_padded_shapes, shuffle, drop_remainder)
        for source in data_sources
    ]
    choices = tf.data.Dataset.range(len(datasets)).repeat()
    dataset = tf.data.experimental.choose_from_datasets(datasets, choices)

    # Separate features and labels.
    batched_examples = dataset.make_one_shot_iterator().get_next()
    features_batch = {k: batched_examples[k] for k in feature_keys}
    labels_batch = {k: batched_examples[k] for k in label_keys}

    return features_batch, labels_batch

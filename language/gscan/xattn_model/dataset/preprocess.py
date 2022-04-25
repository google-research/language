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
"""Convert gSCAN data to TFRecord."""

import collections
import json
import os
import re

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

flags.DEFINE_string('data_dir', None, 'The output data directory.')
flags.DEFINE_enum(
    'split', None,
    ['compositional_splits', 'target_length_split', 'spatial_relation_splits'],
    'The data split.')
flags.DEFINE_integer('k', 0, 'The number of examples sampled from adverb_1.')

FLAGS = flags.FLAGS

MAX_INPUT_LENGTH = {
    'compositional_splits': 10,
    'target_length_split': 10,
    'spatial_relation_splits': 15,
}

MAX_TARGET_LENGTH = {
    'compositional_splits': 105,
    'target_length_split': 50,
    'spatial_relation_splits': 25,
}

SPLIT_MAP = {
    'compositional_splits': [
        'train', 'dev', 'test', 'visual_easier', 'visual', 'situational_1',
        'situational_2', 'contextual', 'adverb_1', 'adverb_2'
    ],
    'target_length_split': ['train', 'dev', 'test', 'target_lengths'],
    'spatial_relation_splits': [
        'train',
        'dev',
        'test',
        'visual',
        'relation',
        'referent',
        'relative_position_1',
        'relative_position_2',
    ]
}


class Vocabulary(object):
  """Object that maps words in string form to indices.

  Code adapted from https://github.com/LauraRuis/multimodal_seq2seq_gSCAN/blob/
    master/seq2seq/gSCAN_dataset.py
  """

  def __init__(self, sos_token='<SOS>', eos_token='<EOS>', pad_token='<PAD>'):
    """NB: <PAD> token is by construction idx 0."""
    self.sos_token = sos_token
    self.eos_token = eos_token
    self.pad_token = pad_token
    self.idx2word = [pad_token, sos_token, eos_token]
    self.word2idx = {}
    self.word2idx[pad_token] = 0
    self.word2idx[sos_token] = 1
    self.word2idx[eos_token] = 2
    self.word_frequencies = collections.Counter()

  def __call__(self, word):
    if self.contains_word(word):
      return self.word2idx[word]
    else:
      raise RuntimeError(f'Word {word} does not exist in vocabulary')

  def __len__(self):
    return len(self.idx2word)

  def contains_word(self, word):
    if word in self.word2idx:
      return True
    else:
      return False

  def add_sentence(self, sentence):
    for word in sentence:
      if word not in self.word2idx:
        self.word2idx[word] = self.size
        self.idx2word.append(word)
      self.word_frequencies[word] += 1

  def most_common(self, n=10):
    return self.word_frequencies.most_common(n=n)

  @property
  def pad_idx(self):
    return self.word2idx[self.pad_token]

  @property
  def start_idx(self):
    return self.word2idx[self.sos_token]

  @property
  def end_idx(self):
    return self.word2idx[self.eos_token]

  @property
  def size(self):
    return len(self.idx2word)

  @classmethod
  def load(cls, path):
    """Load vocabulary."""
    with tf.io.gfile.GFile(path, 'r') as infile:
      all_data = json.load(infile)
      sos_token = all_data['sos_token']
      eos_token = all_data['eos_token']
      pad_token = all_data['pad_token']
      vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
      vocab.idx2word = all_data['idx_to_word']
      vocab.word2idx = collections.defaultdict(int)
      for word, idx in all_data['word_to_idx'].items():
        vocab.word2idx[word] = idx
      vocab.word_frequencies = collections.Counter(all_data['word_frequencies'])
    return vocab

  def to_dict(self):
    return {
        'sos_token': self.sos_token,
        'eos_token': self.eos_token,
        'pad_token': self.pad_token,
        'idx_to_word': self.idx2word,
        'word_to_idx': self.word2idx,
        'word_frequencies': self.word_frequencies
    }

  def save(self, path):
    with tf.io.gfile.GFile(path, 'w') as outfile:
      json.dump(self.to_dict(), outfile, indent=2)
    return path


class Tokenizer(object):
  """A simple tokenizer."""

  def __init__(self, vocab, max_seq_len):
    self.vocab = vocab
    self.max_seq_len = max_seq_len

  def tokenize(self, tokens):
    """Tokenize one sentence."""

    # Add <SOS> and <EOS>.
    tokens = [self.vocab.sos_token] + tokens + [self.vocab.eos_token]
    if len(tokens) > self.max_seq_len:
      tokens = tokens[:self.max_seq_len]
    token_ids = np.asarray([self.vocab(token) for token in tokens])
    token_ids = list(token_ids)
    token_mask = [1] * len(token_ids)

    if len(token_ids) < self.max_seq_len:
      padded_len = self.max_seq_len - len(token_ids)
      token_ids += [self.vocab.pad_idx] * padded_len
      token_mask += [0] * padded_len
    assert len(token_ids) == len(token_mask) == self.max_seq_len
    return token_ids, token_mask


def get_vocabularies(data_dir):
  """Get input and target vocabularies.

  If will read vocabularies from file if exists. Otherwise,
  it will generating vocabulary from examples.

  Args:
    data_dir: Path to data dir.

  Returns:
    Input and target vocabularies.
  """
  input_vocab_file = os.path.join(data_dir, 'training_input_vocab.txt')
  target_vocab_file = os.path.join(data_dir, 'training_target_vocab.txt')
  if tf.io.gfile.exists(input_vocab_file) and tf.io.gfile.exists(
      target_vocab_file):
    logging.info('Loading input vocabulary from %s.', data_dir)
    input_vocab = Vocabulary.load(input_vocab_file)
    target_vocab = Vocabulary.load(target_vocab_file)
  else:
    logging.info('Generating input vocabulary.')
    input_vocab = Vocabulary()
    target_vocab = Vocabulary()
    examples = tfds.load(f'grounded_scan/{FLAGS.split}', split='train')
    for example in examples:
      commands = _parse_commands(example['command'])
      target_commands = _parse_commands(example['target_commands'])
      if FLAGS.split == 'spatial_relation_splits':
        # The spatial relation split also uses whitespace as separtor,
        # therefore it will tokenize south east as two words.
        commands = re.split(',| ', ','.join(commands))
      input_vocab.add_sentence(commands)
      target_vocab.add_sentence(target_commands)
    input_vocab.save(input_vocab_file)
    target_vocab.save(target_vocab_file)
    logging.info('Saved input vocabulary to %s', data_dir)
  return input_vocab, target_vocab


def _parse_commands(commands):
  return [c.numpy().decode('utf-8') for c in commands]


def _parse_vector(vector):
  return [int(bit) for bit in vector.numpy().decode('utf-8')]


def parse_sparse_situation(situation_representation):
  """Parse sparse situation to dense grid.

  Code adapted from https://github.com/LauraRuis/multimodal_seq2seq_gSCAN/
    blob/master/read_gscan/read_gscan.py
  Each grid cell in a situation is fully specified by a vector:
  [_ _ _ _ _ _ _   _       _      _       _   _ _ _ _]
   1 2 3 4 r g b circle square cylinder agent E S W N
   _______ _____ ______________________ _____ _______
     size  color        shape           agent agent dir.

  Args:
    situation_representation: data from dataset.txt at key "situation".

  Returns:
    grid to be parsed by computational models.
  """
  grid_size = situation_representation['grid_size']
  num_object_attributes = len(
      _parse_vector(situation_representation['target_object']['vector']))
  # Object representation + agent bit + agent direction bits (see docstring).
  num_grid_channels = num_object_attributes + 1 + 4

  # Initialize the grid.
  grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

  # Place the agent.
  agent_row = int(situation_representation['agent_position']['row'])
  agent_column = int(situation_representation['agent_position']['column'])
  agent_direction = int(situation_representation['agent_direction'])
  agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
  agent_representation[-5] = 1
  agent_representation[-4 + agent_direction] = 1
  grid[agent_row, agent_column, :] = agent_representation

  # Loop over the objects in the world and place them.
  placed_objects = situation_representation['placed_objects']
  for idx, object_vector in enumerate(placed_objects['vector']):
    object_vector = np.array(_parse_vector(object_vector), dtype=np.int32)
    object_row = int(placed_objects['position']['row'][idx])
    object_column = int(placed_objects['position']['column'][idx])
    grid[object_row, object_column, :] = np.concatenate(
        [object_vector, np.zeros([5], dtype=np.int32)])
  return grid


def get_int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_bytes_feature(value):
  value_bytes = [tf.compat.as_bytes(element) for element in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_bytes))


def raw_data_to_tfrecord(item):
  """Process data into TFRecord compatible format to prepare for writing."""
  image = tf.io.serialize_tensor(tf.convert_to_tensor(item['image'])).numpy()
  context_features = {
      'index': get_int64_feature([item['index']]),
      'token': get_int64_feature(item['token']),
      'txt_mask': get_int64_feature(item['txt_mask']),
      'target_token': get_int64_feature(item['target_token']),
      'target_txt_mask': get_int64_feature(item['target_txt_mask']),
      'image': get_bytes_feature([image]),
  }
  ex = tf.train.Example(features=tf.train.Features(feature=context_features))

  return ex


def process_data(data_dir,
                 output_path,
                 split,
                 max_seq_len=10,
                 max_target_seq_len=105,
                 k=0,
                 k_split='adverb_1'):
  """Process the raw data."""

  input_vocab, target_vocab = get_vocabularies(data_dir)
  input_tokenizer = Tokenizer(input_vocab, max_seq_len=max_seq_len)
  target_tokenizer = Tokenizer(target_vocab, max_seq_len=max_target_seq_len)

  def process_one_example(example, index):
    """Process one example."""
    input_commands = _parse_commands(example['command'])
    target_commands = _parse_commands(example['target_commands'])
    if FLAGS.split == 'spatial_relation_splits':
      input_commands = re.split(',| ', ','.join(input_commands))
    situation = example['situation']
    situation_image = parse_sparse_situation(situation)
    input_ids, input_mask = input_tokenizer.tokenize(input_commands)
    target_ids, target_mask = target_tokenizer.tokenize(target_commands)

    return {
        'index': index,
        'token': np.array(input_ids, dtype=np.int32),
        'txt_mask': np.array(input_mask, dtype=np.int32),
        'target_token': np.array(target_ids, dtype=np.int32),
        'target_txt_mask': np.array(target_mask, dtype=np.int32),
        'image': situation_image.astype(np.float32),
    }

  count = 0
  examples = tfds.load(f'grounded_scan/{FLAGS.split}', split=split)
  with tf.io.TFRecordWriter(output_path) as writer:
    for i, example in enumerate(examples):
      item = process_one_example(example, i)
      tf_example = raw_data_to_tfrecord(item)
      writer.write(tf_example.SerializeToString())
      count += 1

    if k > 0 and split in ['train', 'dev']:
      logging.info('Sampling %d examples from %s to %s.', k, k_split, split)
      k_examples = tfds.load(
          f'grounded_scan/{FLAGS.split}', split=k_split, shuffle_files=True)
      for i, example in enumerate(k_examples.take(k)):
        item = process_one_example(example, len(examples) + i)
        tf_example = raw_data_to_tfrecord(item)
        writer.write(tf_example)
        count += 1
  logging.info('Processing %d items for split %s.', count, split)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.io.gfile.makedirs(FLAGS.data_dir)

  for data_split in SPLIT_MAP[FLAGS.split]:
    logging.info('Processing data for split %s', data_split)
    process_data(
        FLAGS.data_dir,
        os.path.join(FLAGS.data_dir, f'{data_split}.tfrecord'),
        data_split,
        max_seq_len=MAX_INPUT_LENGTH[FLAGS.split],
        max_target_seq_len=MAX_TARGET_LENGTH[FLAGS.split],
        k=FLAGS.k)


if __name__ == '__main__':
  app.run(main)

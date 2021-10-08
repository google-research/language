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
"""Prepare HoVeR dataset for evaluation."""

import json
import os


from absl import app
from absl import flags
from absl import logging
import language.bert.tokenization as bert_tokenization
from language.mentionmemory.utils import tokenization_utils
import numpy as np
import spacy
import tensorflow.compat.v2 as tf
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'save directory')
flags.DEFINE_string('data_dir', None, 'Directory where raw data is located.')
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('max_mentions', 32, 'max nr of mentions')

flags.DEFINE_list('split_file_names', ['train', 'dev'],
                  'List of split file names.')

LABEL_DICT = {
    'NOT_SUPPORTED': 0,
    'SUPPORTED': 1,
}


def process_data(data, spacy_model,
                 tokenizer):
  """Processes HoVeR splits.

  Given claims and accompanying labels, tokenizes the claims and annotates
  claim with mention positions corresponding to noun chunks.

  Args:
    data: List of raw HoVeR samples. Each raw HoVeR sample contains: claim, text
      that describes the claim to be classified. label, string label, SUPPORTED
      or NOT_SUPPORTED.
    spacy_model: Spacy model object.
    tokenizer: BERT tokenizer.

  Returns:
    List of processed HoVeR samples. Each sample contains:
      text_ids: dense token ids.
      text_mask: dense padding mask.
      mention_start_positions: sparse array of mention starting positions.
      mention_end_positions: sparse array of mention end positions.
      mention_mask: mention padding mask.
      target: string labels converted to 0 or 1.
  """
  processed_data = []
  for sample in tqdm.tqdm(data):
    new_sample = {}
    parsed_claim = spacy_model(sample['claim'])
    mention_char_spans = []
    for chunk in parsed_claim.noun_chunks:
      span_start_char = parsed_claim[chunk.start].idx
      span_last_token = parsed_claim[chunk.end - 1]
      span_end_char = span_last_token.idx + len(span_last_token.text) - 1
      mention_char_spans.append((span_start_char, span_end_char))

    _, text_ids, text_mask, mention_spans, _ = tokenization_utils.tokenize_with_mention_spans(
        tokenizer=tokenizer,
        sentence=sample['claim'],
        spans=mention_char_spans,
        max_length=FLAGS.max_length,
        add_bert_tokens=True,
        allow_truncated_spans=True)

    new_sample['text_ids'] = text_ids
    new_sample['text_mask'] = text_mask

    mention_spans = np.array(mention_spans)
    if mention_spans.size == 0:
      mention_start_positions = np.array([], dtype=np.int64)
      mention_end_positions = np.array([], dtype=np.int64)
    else:
      mention_start_positions = mention_spans[:, 0]
      mention_end_positions = mention_spans[:, 1]

    mention_start_positions = mention_start_positions[:FLAGS.max_mentions]
    mention_end_positions = mention_end_positions[:FLAGS.max_mentions]

    mention_pad_shape = (0, FLAGS.max_mentions - len(mention_start_positions))

    mention_mask = np.ones(len(mention_start_positions), dtype=np.int64)
    mention_mask = np.pad(mention_mask, mention_pad_shape, mode='constant')
    mention_start_positions = np.pad(
        mention_start_positions, mention_pad_shape, mode='constant')
    mention_end_positions = np.pad(
        mention_end_positions, mention_pad_shape, mode='constant')

    new_sample['mention_start_positions'] = mention_start_positions
    new_sample['mention_end_positions'] = mention_end_positions
    new_sample['mention_mask'] = mention_mask
    new_sample['target'] = [LABEL_DICT[sample['label']]]
    processed_data.append(new_sample)

  return processed_data


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tokenizer = bert_tokenization.FullTokenizer(
      FLAGS.vocab_path, do_lower_case=True)

  spacy_model = None
  if spacy_model is None:
    spacy_model = spacy.load('en_core_web_md')

  raw_data = {}
  for split_file_name in FLAGS.split_file_names:
    path = os.path.join(FLAGS.data_dir, split_file_name + '.json')
    with tf.io.gfile.GFile(path, 'rb') as data_file:
      raw_data[split_file_name] = json.load(data_file)

  processed_data = {}
  for split_name, split_data in raw_data.items():
    logging.info('Processing %s split', split_name)
    processed_split_data = process_data(split_data, spacy_model, tokenizer)
    processed_data[split_name] = processed_split_data

  # Create TFRecords
  tf.io.gfile.makedirs(FLAGS.save_dir)
  for split_name, split_data in processed_data.items():
    file_path = os.path.join(FLAGS.save_dir, split_name)
    logging.info('Writing %s split to %s', split_name, file_path)
    writer = tf.io.TFRecordWriter(file_path)
    for sample in split_data:
      features = tf.train.Features(
          feature={
              key: tf.train.Feature(int64_list=tf.train.Int64List(value=value))
              for key, value in sample.items()
          })

      record_bytes = tf.train.Example(features=features).SerializeToString()
      writer.write(record_bytes)


if __name__ == '__main__':
  app.run(main)

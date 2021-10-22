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
"""Prepare fever dataset for evaluation."""

import json
import os

from urllib import request

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
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('max_mentions', 32, 'max nr of mentions')

URL_DICT = {
    'train':
        'https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl',
    'dev':
        'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl'
}

LABEL_DICT = {
    'REFUTES': 0,
    'SUPPORTS': 1,
    'NOT ENOUGH INFO': 2,
}


def process_data(data, spacy_model,
                 tokenizer):
  """Processes fever splits.

  Given claims and accompanying labels, tokenizes the claims and annotates
  claim with mention positions corresponding to noun chunks.

  Args:
    data: List of raw FEVER samples. Each raw FEVER sample contains: claim -
      text that describes the claim to be classified. label - string label,
      REFUTES, SUPPORTS or NOT ENOUGH INFO.
    spacy_model: Spacy model object.
    tokenizer: BERT tokenizer.

  Returns:
    List of processed FEVER samples. Each sample contains:
      text_ids: dense token ids.
      text_mask: dense padding mask.
      mention_start_positions: sparse array of mention starting positions.
      mention_end_positions: sparse array of mention end positions.
      mention_mask: mention padding mask.
      target: string labels converted to 0, 1, 2.
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
    )

    new_sample['text_ids'] = text_ids
    new_sample['text_mask'] = text_mask

    mention_spans = np.array(mention_spans)
    if len(mention_spans) == 0:  # pylint: disable=g-explicit-length-test
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

  # Download FEVER data
  raw_data = {}
  for split_name, url in URL_DICT.items():
    logging.info('Downloading %s split', split_name)
    with request.urlopen(url) as open_url:
      sample_list = []
      lines = open_url.readlines()
      for line in lines:
        sample_list.append(json.loads(line))
      raw_data[split_name] = sample_list

  # Process FEVER data
  tokenizer = bert_tokenization.FullTokenizer(
      FLAGS.vocab_path, do_lower_case=True)

  spacy_model = None
  if spacy_model is None:
    spacy_model = spacy.load('en_core_web_md')

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
